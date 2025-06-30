import os
import re
from pathlib import Path

def merge_consecutive_tags(content):
    """
    Merge consecutive NER tags of the same type in XML content.
    
    Args:
        content (str): XML content with NER tags
    
    Returns:
        str: XML content with merged consecutive tags
    """
    # Define the entity types to process
    entity_types = ['PER', 'LOC', 'ORG', 'MSC', 'TME']
    
    # Process each entity type
    for entity_type in entity_types:
        # Pattern to match consecutive tags of the same type
        # This will match patterns like <PER>text1</PER><PER>text2</PER>...
        pattern = rf'(<{entity_type}>([^<]+)</{entity_type}>)+'
        
        def merge_match(match):
            # Extract all text between the consecutive tags
            full_match = match.group(0)
            # Find all text content within the tags
            texts = re.findall(rf'<{entity_type}>([^<]+)</{entity_type}>', full_match)
            # Merge all texts and wrap in single tag
            merged_text = ''.join(texts)
            return f'<{entity_type}>{merged_text}</{entity_type}>'
        
        # Replace all consecutive tags with merged version
        content = re.sub(pattern, merge_match, content)
    
    return content

def process_xml_files(input_folder='results/output_ner', output_folder='refine'):
    """
    Process all XML files in the input folder and save refined versions to output folder.
    
    Args:
        input_folder (str): Path to folder containing XML files with NER annotations
        output_folder (str): Path to folder where refined XML files will be saved
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all XML files in the input folder
    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]
    
    if not xml_files:
        print(f"No XML files found in {input_folder}")
        return
    
    print(f"Found {len(xml_files)} XML files to process")
    
    # Process each XML file
    for xml_file in xml_files:
        input_path = os.path.join(input_folder, xml_file)
        output_path = os.path.join(output_folder, xml_file)
        
        try:
            # Read the XML content
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Merge consecutive tags
            refined_content = merge_consecutive_tags(content)
            
            # Write the refined content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(refined_content)
            
            print(f"Processed: {xml_file}")
            
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
    
    print(f"\nProcessing complete! Refined files saved to '{output_folder}' folder")

def main():
    """Main function to execute the XML refinement process."""
    # Process files from results/output_ner to refine folder
    process_xml_files()

if __name__ == "__main__":
    main()
