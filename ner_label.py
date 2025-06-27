import os
import json
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import Dataset
import torch
import numpy as np
import re

# Load Hugging Face model for Chinese NER (more robust for historical texts)
try:
    # Use a Chinese NER model that's better suited for classical Chinese
    tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese-ner")
    model = AutoModelForTokenClassification.from_pretrained("ckiplab/bert-base-chinese-ner")
    hf_ner = pipeline("ner", 
                      model=model, 
                      tokenizer=tokenizer, 
                      aggregation_strategy="simple",
                      device=0 if torch.cuda.is_available() else -1)
    print("Using Hugging Face Chinese NER model for ancient Chinese support")
except Exception as e:
    print(f"Failed to load Hugging Face model: {e}")
    raise RuntimeError("Cannot load NER model. Please install transformers and torch.")

# Enhanced mapping for ancient Chinese context
HF_TO_CUSTOM = {
    "PER": "PER",
    "PERSON": "PER", 
    "LOC": "LOC",
    "GPE": "LOC",
    "ORG": "ORG",
    "MISC": "MSC",
    "DATE": "TME",
    "TIME": "TME"
}

def process_texts_batch(texts):
    """Process multiple texts in batch for better GPU efficiency."""
    if not texts:
        return []
    
    # Create dataset for batch processing
    dataset = Dataset.from_dict({"text": texts})
    
    # Process in batches
    batch_size = 8 if torch.cuda.is_available() else 4
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_results = hf_ner(batch_texts)
        
        # Handle both single and batch results
        if not isinstance(batch_results[0], list):
            batch_results = [batch_results]
            
        results.extend(batch_results)
    
    return results

def is_chinese_char(char):
    """Check if a character is Chinese."""
    return '\u4e00' <= char <= '\u9fff'

def preprocess_chinese_text(text):
    """Remove spaces between Chinese characters while preserving structure."""
    if not text:
        return text
    
    chars = list(text)
    cleaned_chars = []
    i = 0
    
    while i < len(chars):
        current_char = chars[i]
        
        if current_char == ' ':
            # Check if this space is between Chinese characters
            prev_is_chinese = i > 0 and is_chinese_char(chars[i-1])
            next_is_chinese = i < len(chars)-1 and is_chinese_char(chars[i+1])
            
            if prev_is_chinese and next_is_chinese:
                # Skip this space (don't add to cleaned text)
                i += 1
                continue
        
        # Add character to cleaned text
        cleaned_chars.append(current_char)
        i += 1
    
    cleaned_text = ''.join(cleaned_chars)
    return cleaned_text

def tag_entities_hf(text):
    """Tag entities using Hugging Face model with preprocessing."""
    # Preprocess text to remove spaces between Chinese characters
    cleaned_text = preprocess_chinese_text(text)
    
    # Apply NER to cleaned text
    entities = hf_ner(cleaned_text)
    
    tagged_text = cleaned_text
    results = []
    
    # Sort entities by start position in reverse to avoid offset issues
    entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    for ent in entities:
        start = ent['start']
        end = ent['end']
        label = ent['entity_group'] if 'entity_group' in ent else ent['entity']
        entity_text = cleaned_text[start:end]
        confidence = ent.get('score', 1.0)
        
        # Only include high-confidence entities for ancient texts
        if confidence > 0.7:
            if label in HF_TO_CUSTOM:
                custom_tag = HF_TO_CUSTOM[label]
                tagged_text = tagged_text[:start] + f"<{custom_tag}>{entity_text}</{custom_tag}>" + tagged_text[end:]
                results.append({
                    'text': entity_text,
                    'label': custom_tag,
                    'start': start,
                    'end': end,
                    'confidence': float(confidence)
                })
    
    return tagged_text, results

def tag_entities_batch(texts):
    """Tag entities for multiple texts using batch processing with preprocessing."""
    # Preprocess all texts
    cleaned_texts = [preprocess_chinese_text(text) for text in texts]
    
    # Apply NER to cleaned texts
    batch_entities = process_texts_batch(cleaned_texts)
    results = []
    
    for i, (cleaned_text, entities) in enumerate(zip(cleaned_texts, batch_entities)):
        tagged_text = cleaned_text
        text_results = []
        
        # Sort entities by start position in reverse to avoid offset issues
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        for ent in entities:
            start = ent['start']
            end = ent['end']
            label = ent['entity_group'] if 'entity_group' in ent else ent['entity']
            entity_text = cleaned_text[start:end]
            confidence = ent.get('score', 1.0)
            
            # Only include high-confidence entities for ancient texts
            if confidence > 0.7:
                if label in HF_TO_CUSTOM:
                    custom_tag = HF_TO_CUSTOM[label]
                    tagged_text = tagged_text[:start] + f"<{custom_tag}>{entity_text}</{custom_tag}>" + tagged_text[end:]
                    text_results.append({
                        'text': entity_text,
                        'label': custom_tag,
                        'start': start,
                        'end': end,
                        'confidence': float(confidence)
                    })
        
        results.append((tagged_text, text_results))
    
    return results

def tag_entities(text):
    """Main entity tagging function."""
    return tag_entities_hf(text)

def tag_entities_to_xml(stc_elem, text):
    """Replace the text of stc_elem with a sequence of text and entity subelements."""
    # Preprocess text to remove spaces between Chinese characters
    cleaned_text = preprocess_chinese_text(text)
    
    # Apply NER to cleaned text
    entities = hf_ner(cleaned_text)
    entities = [ent for ent in entities if ent.get('score', 1.0) > 0.7]
    
    last_idx = 0
    # Remove all children first
    for child in list(stc_elem):
        stc_elem.remove(child)
    stc_elem.text = ''
    
    for ent in entities:
        start = ent['start']
        end = ent['end'] 
        label = ent['entity_group'] if 'entity_group' in ent else ent.get('entity', '')
        entity_text = cleaned_text[start:end]
        
        # Add text before the entity
        if start > last_idx:
            if stc_elem.text == '':
                stc_elem.text = cleaned_text[last_idx:start]
            else:
                if len(stc_elem):
                    stc_elem[-1].tail = (stc_elem[-1].tail or '') + cleaned_text[last_idx:start]
                else:
                    stc_elem.text += cleaned_text[last_idx:start]
        
        # Add the entity as a subelement if it's a mapped label
        if label in HF_TO_CUSTOM:
            tag = HF_TO_CUSTOM[label]
            ent_elem = ET.Element(tag)
            ent_elem.text = entity_text
            stc_elem.append(ent_elem)
        else:
            # If not mapped, treat as plain text
            if len(stc_elem):
                stc_elem[-1].tail = (stc_elem[-1].tail or '') + entity_text
            else:
                stc_elem.text += entity_text
        last_idx = end
    
    # Add any remaining text after the last entity
    if last_idx < len(cleaned_text):
        if len(stc_elem):
            stc_elem[-1].tail = (stc_elem[-1].tail or '') + cleaned_text[last_idx:]
        else:
            stc_elem.text += cleaned_text[last_idx:]
    # Remove leading empty text if present
    if stc_elem.text == '':
        stc_elem.text = None

def process_xml_file(input_path, output_dir):
    """Process an XML file and apply NER tagging as real XML tags."""
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # Collect all texts for batch processing
    stc_elements = []
    texts = []
    
    for stc in root.findall(".//STC"):
        if stc.text:
            stc_elements.append(stc)
            texts.append(stc.text)
    
    # Process all texts in batch for better GPU efficiency
    if texts:
        batch_results = tag_entities_batch(texts)
    else:
        batch_results = []
    
    # Store results for JSON output
    file_results = {
        'filename': os.path.basename(input_path),
        'sentences': []
    }
    
    # Apply results to XML elements and collect for JSON
    for stc_elem, original_text, (tagged_text, entities) in zip(stc_elements, texts, batch_results):
        # Store for JSON
        file_results['sentences'].append({
            'original_text': original_text,
            'cleaned_text': preprocess_chinese_text(original_text),
            'tagged_text': tagged_text,
            'entities': entities
        })
        
        # Apply XML tagging
        tag_entities_to_xml(stc_elem, original_text)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save XML output
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    
    # Save JSON output for readability
    json_output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '_ner.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(file_results, f, ensure_ascii=False, indent=2)
    
    return output_path, json_output_path

def get_xml_files(xml_dir):
    """Get all XML files from the xml directory."""
    xml_files = []
    if os.path.exists(xml_dir):
        for file in os.listdir(xml_dir):
            if file.endswith('.xml'):
                xml_files.append(os.path.join(xml_dir, file))
    return xml_files

# Configuration
xml_dir = "xml"
output_dir = "output_ner"

def main():
    # Get all XML files from the xml directory
    input_files = get_xml_files(xml_dir)
    
    if not input_files:
        print(f"No XML files found in {xml_dir} directory.")
        return
    
    print(f"Found {len(input_files)} XML files to process.")
    
    # Process each input file
    for input_file in input_files:
        try:
            xml_output, json_output = process_xml_file(input_file, output_dir)
            print(f"Processed {input_file}")
            print(f"  -> XML: {xml_output}")
            print(f"  -> JSON: {json_output}")
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    main()