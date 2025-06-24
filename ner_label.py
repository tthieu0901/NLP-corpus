import os
import xml.etree.ElementTree as ET
import spacy

# Load SpaCy model for Chinese
nlp = spacy.load("zh_core_web_lg")

# Map SpaCy labels to desired labels
SPACY_TO_CUSTOM = {
    "PERSON": "PER",
    "GPE": "LOC",  # Geopolitical entity (locations)
    "LOC": "LOC",
    "ORG": "ORG",
    "DATE": "TME",
    "CARDINAL": "NUM",
    "QUANTITY": "NUM"
}

def tag_entities(text):
    """Tag entities in the text using SpaCy."""
    doc = nlp(text)
    tagged_text = text

    # Sort entities by start position in reverse to avoid offset issues when replacing
    entities = sorted([(ent.start_char, ent.end_char, ent.label_, ent.text) for ent in doc.ents], reverse=True)

    for start, end, label, entity_text in entities:
        if label in SPACY_TO_CUSTOM:
            custom_tag = SPACY_TO_CUSTOM[label]
            tagged_text = tagged_text[:start] + f"<{custom_tag}>{entity_text}</{custom_tag}>" + tagged_text[end:]

    return tagged_text

def tag_entities_to_xml(stc_elem, text):
    """Replace the text of stc_elem with a sequence of text and entity subelements."""
    doc = nlp(text)
    last_idx = 0
    # Remove all children first
    for child in list(stc_elem):
        stc_elem.remove(child)
    stc_elem.text = ''
    for ent in doc.ents:
        # Add text before the entity
        if ent.start_char > last_idx:
            if stc_elem.text == '':
                stc_elem.text = text[last_idx:ent.start_char]
            else:
                # Add as tail to the last child
                if len(stc_elem):
                    stc_elem[-1].tail = (stc_elem[-1].tail or '') + text[last_idx:ent.start_char]
                else:
                    stc_elem.text += text[last_idx:ent.start_char]
        # Add the entity as a subelement if it's a mapped label
        if ent.label_ in SPACY_TO_CUSTOM:
            tag = SPACY_TO_CUSTOM[ent.label_]
            ent_elem = ET.Element(tag)
            ent_elem.text = ent.text
            stc_elem.append(ent_elem)
        else:
            # If not mapped, treat as plain text
            if len(stc_elem):
                stc_elem[-1].tail = (stc_elem[-1].tail or '') + ent.text
            else:
                stc_elem.text += ent.text
        last_idx = ent.end_char
    # Add any remaining text after the last entity
    if last_idx < len(text):
        if len(stc_elem):
            stc_elem[-1].tail = (stc_elem[-1].tail or '') + text[last_idx:]
        else:
            stc_elem.text += text[last_idx:]
    # Remove leading empty text if present
    if stc_elem.text == '':
        stc_elem.text = None

def process_xml_file(input_path, output_dir):
    """Process an XML file and apply NER tagging as real XML tags."""
    tree = ET.parse(input_path)
    root = tree.getroot()
    for stc in root.findall(".//STC"):
        if stc.text:
            tag_entities_to_xml(stc, stc.text)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

# Hard-coded input files and output directory
input_files = ["xml/phong_data.xml", "xml/hieu_data.xml"]  # Replace with actual second file if needed
output_dir = "output_ner"

def main():
    # Process each input file
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"File {input_file} does not exist.")
            continue
        try:
            process_xml_file(input_file, output_dir)
            print(f"Processed {input_file} -> {os.path.join(output_dir, os.path.basename(input_file))}")
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    main()