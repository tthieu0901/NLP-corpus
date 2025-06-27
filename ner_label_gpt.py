import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET
from lxml import etree
from openai import AzureOpenAI
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChineseNERPipeline:
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str = "2025-01-01"):
        """
        Initialize the NER pipeline with Azure OpenAI credentials.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version to use
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        # Entity label mapping from OpenAI/standard to custom labels
        self.label_mapping = {
            'PERSON': 'PER',
            'PER': 'PER',
            'LOCATION': 'LOC',
            'LOC': 'LOC',
            'ORGANIZATION': 'ORG',
            'ORG': 'ORG',
            'MISCELLANEOUS': 'MSC',
            'MISC': 'MSC',
            'MSC': 'MSC',
            'TIME': 'TME',
            'DATE': 'TME',
            'TME': 'TME'
        }
        
        self.input_dir = Path("xml")
        self.output_dir = Path("output_ner")
        self.output_dir.mkdir(exist_ok=True)
    
    def normalize_chinese_text(self, spaced_text: str) -> str:
        """
        Remove spaces between Chinese characters for OpenAI processing.
        
        Args:
            spaced_text: Chinese text with spaces between characters
            
        Returns:
            Normalized text without spaces between Chinese characters
        """
        # Remove spaces between Chinese characters but preserve punctuation spacing
        normalized = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', spaced_text)
        return normalized.strip()
    
    def call_azure_openai_ner(self, text: str, max_retries: int = 3) -> str:
        """
        Call Azure OpenAI GPT-4 to perform NER on Chinese text and return tagged text.
        
        Args:
            text: Chinese text to process
            max_retries: Maximum number of retry attempts
            
        Returns:
            Text with NER tags inserted, or original text if processing fails
        """
        system_prompt = """你是一个专业的学术研究助手，专门处理中国古代历史文献的命名实体识别任务。

任务说明：
- 这是一项纯学术研究，涉及中国古代历史文献的语言学分析
- 文本来源于历史典籍，包含古代人名、地名、官职等学术研究内容
- 请以客观、学术的角度进行实体识别，不涉及任何现代政治或敏感话题

请识别以下实体类型并在原文中直接标注：
- PER: 古代人名（如历史人物、官员等）
- LOC: 古代地名（如城市、国家、地区等）
- ORG: 古代机构名（如官署、学校等）
- MSC: 其他重要实体（如器物、典籍等）
- TME: 时间或日期表达

输出格式要求：
请直接在原文中插入XML标签，格式如下：
<PER>赵简子</PER>有两个儿子，长子名叫<PER>赵伯鲁</PER>，次子名叫<PER>赵无恤</PER>。

注意：
1. 只输出标注后的文本，不要添加任何解释
2. 保持原文的所有标点符号和格式
3. 这是历史文献研究，所有内容均为学术用途"""

        user_prompt = f"""请对以下中国古代历史文献片段进行命名实体识别并直接在文本中标注：

{text}"""
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4.1-nano",  # Use your deployed model name
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                logger.info(f"OpenAI response: {response.choices[0].message.content}")
                result_text = response.choices[0].message.content.strip()
                
                # Validate that the response contains NER tags
                if re.search(r'<(PER|LOC|ORG|MSC|TME)>', result_text):
                    logger.info(f"NER tagging successful, found tags in response")
                    return result_text
                else:
                    logger.warning(f"No NER tags found in response, returning original text")
                    return text
                    
            except Exception as e:
                error_message = str(e)
                # Handle content filtering specifically
                if "content_filter" in error_message or "ResponsibleAIPolicyViolation" in error_message:
                    logger.warning(f"Content filter triggered on attempt {attempt + 1} for text: {text[:100]}...")
                    logger.warning("This is likely a false positive for historical Chinese text. Returning original text.")
                    return text
                else:
                    logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        return text
        
        return text
    
    def process_xml_file(self, xml_file_path: Path) -> None:
        """
        Process a single XML file and generate NER-annotated output.
        
        Args:
            xml_file_path: Path to the input XML file
        """
        try:
            logger.info(f"Processing file: {xml_file_path}")
            
            # Parse the XML file using lxml
            parser = etree.XMLParser(encoding='utf-8', remove_blank_text=False)
            tree = etree.parse(str(xml_file_path), parser)
            root = tree.getroot()

            # Find all <STC> elements and modify them in-place
            stc_elements = root.xpath('.//STC')

            for stc in stc_elements:
                if stc.text:
                    original_text = stc.text.strip()
                    if original_text:
                        logger.info(f"Original STC text: '{original_text[:100]}...'")

                        # Normalize Chinese text (e.g., remove unnecessary spaces)
                        normalized_text = self.normalize_chinese_text(original_text)
                        logger.info(f"Normalized text: '{normalized_text[:100]}...'")

                        # Call your Azure OpenAI model for NER annotation
                        annotated_text = self.call_azure_openai_ner(normalized_text)
                        logger.info(f"Annotated text: '{annotated_text[:100]}...'")

                        # Replace STC content with annotated XML (parsed correctly)
                        self._replace_stc_content_with_xml(stc, annotated_text)
            
            # Write the modified XML tree to the output file
            output_file_path = self.output_dir / xml_file_path.name
            tree.write(str(output_file_path), encoding='utf-8', pretty_print=True, xml_declaration=True)

            logger.info(f"Saved annotated XML to: {output_file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {xml_file_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _replace_stc_content_with_xml(self, stc_element, annotated_text: str) -> None:
        """
        Replace the text content of an STC element with annotated XML content.

        Args:
            stc_element: The <STC> element from the XML tree.
            annotated_text: The annotated string containing inline XML tags.
        """
        try:
            # 🧼 Sanitize malformed tags like < TME> → <TME>
            annotated_text = re.sub(r'<\s+(/?\s*\w+)\s*>', r'<\1>', annotated_text)

            # Wrap in dummy root for XML parsing
            wrapped = f"<dummy>{annotated_text}</dummy>"
            parsed = etree.fromstring(wrapped)

            # Clear current content
            stc_element.clear()

            # Set text before first tag
            stc_element.text = parsed.text

            # Append all child elements (NER tags)
            for node in parsed:
                stc_element.append(node)

        except Exception as e:
            logger.error(f"Error replacing STC content with annotated XML: {e}")
            logger.error(f"Annotated text was: {annotated_text}")
            import traceback
            logger.error(traceback.format_exc())
    
    def write_xml_with_ner_tags(self, tree: etree._ElementTree, output_path: Path) -> None:
        """
        Write XML file while preserving NER tags as literal angle brackets.
        (This method is now deprecated in favor of direct text processing)
        
        Args:
            tree: lxml ElementTree object
            output_path: Path to write the output file
        """
        # This method is kept for backward compatibility but not used
        logger.warning("write_xml_with_ner_tags is deprecated - using direct text processing instead")
        try:
            tree.write(str(output_path), encoding='utf-8', xml_declaration=True, pretty_print=True)
        except Exception as e:
            logger.error(f"Error writing XML file: {e}")
    
    def process_all_files(self) -> None:
        """
        Process all XML files in the input directory with specific ordering.
        """
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        xml_files = list(self.input_dir.glob("*.xml"))
        if not xml_files:
            logger.warning(f"No XML files found in {self.input_dir}")
            return
        
        # Define priority order for specific files
        priority_files = ["phong_data.xml", "hieu_data.xml"]
        
        # Separate priority files from others
        priority_file_paths = []
        other_file_paths = []
        
        for xml_file in xml_files:
            if xml_file.name in priority_files:
                priority_file_paths.append(xml_file)
            else:
                other_file_paths.append(xml_file)
        
        # Sort priority files according to specified order
        ordered_priority_files = []
        for priority_name in priority_files:
            for file_path in priority_file_paths:
                if file_path.name == priority_name:
                    ordered_priority_files.append(file_path)
                    break
        
        # Combine ordered priority files with remaining files
        ordered_files = ordered_priority_files + sorted(other_file_paths, key=lambda x: x.name)
        
        logger.info(f"Found {len(xml_files)} XML files to process")
        logger.info(f"Processing order: {[f.name for f in ordered_files]}")
        
        for xml_file in ordered_files:
            self.process_xml_file(xml_file)
            time.sleep(1)  # Rate limiting between files
        
        logger.info("Processing completed!")
    
    def create_test_file(self, test_filename: str = "test_single_stc.xml") -> Path:
        """
        Create a test XML file with a single STC element for testing.
        
        Args:
            test_filename: Name of the test file to create
            
        Returns:
            Path to the created test file
        """
        # Sample Chinese historical text for testing
        test_text = "赵 简 子 有 两 个 儿子 ， 长 子 名 叫 赵 伯 鲁 ， 次 子 名 叫 赵 无 恤 。"
        
        # Create test XML structure
        test_xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<root>
    <PAGE>
        <SECT>
            <FILE>
                <STC>{test_text}</STC>
            </FILE>
        </SECT>
    </PAGE>
</root>"""
        
        # Write test file to xml directory
        test_file_path = self.input_dir / test_filename
        
        try:
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_xml_content)
            
            logger.info(f"Created test file: {test_file_path}")
            logger.info(f"Test content: {test_text}")
            return test_file_path
            
        except Exception as e:
            logger.error(f"Error creating test file: {e}")
            raise
    
    def test_single_stc(self, test_filename: str = "test_single_stc.xml") -> None:
        """
        Test the NER pipeline on a single STC element.
        
        Args:
            test_filename: Name of the test file to process
        """
        try:
            logger.info("=" * 50)
            logger.info("STARTING SINGLE STC TEST")
            logger.info("=" * 50)
            
            # Create test file
            test_file_path = self.create_test_file(test_filename)
            
            # Process the test file
            logger.info(f"Processing test file: {test_file_path}")
            self.process_xml_file(test_file_path)
            
            # Read and display results
            output_file_path = self.output_dir / test_filename
            if output_file_path.exists():
                logger.info("=" * 50)
                logger.info("TEST RESULTS:")
                logger.info("=" * 50)
                
                # Read the output file as text to show the actual NER tags
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract STC content using regex (since it now contains NER tags)
                stc_matches = re.findall(r'<STC>(.*?)</STC>', content, re.DOTALL)
                
                for i, stc_content in enumerate(stc_matches):
                    stc_content = stc_content.strip()
                    logger.info(f"STC {i+1} Result: {stc_content}")
                    
                    # Count entities found
                    entity_count = len(re.findall(r'<(PER|LOC|ORG|MSC|TME)>', stc_content))
                    logger.info(f"Entities found: {entity_count}")
                    
                    # Show each entity type
                    for entity_type in ['PER', 'LOC', 'ORG', 'MSC', 'TME']:
                        entities = re.findall(f'<{entity_type}>(.*?)</{entity_type}>', stc_content)
                        if entities:
                            logger.info(f"  {entity_type}: {entities}")
                
                # Also show a sample of the raw XML
                logger.info("=" * 50)
                logger.info("SAMPLE RAW XML:")
                logger.info("=" * 50)
                sample_lines = content.split('\n')[:15]  # Show first 15 lines
                for line in sample_lines:
                    if line.strip():
                        logger.info(line)
                
                logger.info("=" * 50)
                logger.info("TEST COMPLETED SUCCESSFULLY!")
                logger.info("=" * 50)
                
            else:
                logger.error(f"Output file not found: {output_file_path}")
                
        except Exception as e:
            logger.error(f"Error in test_single_stc: {e}")
    
    def test_custom_text(self, custom_text: str, test_filename: str = "test_custom_stc.xml") -> None:
        """
        Test the NER pipeline on custom Chinese text.
        
        Args:
            custom_text: Custom Chinese text to test (with spaces between characters)
            test_filename: Name of the test file to create
        """
        try:
            logger.info("=" * 50)
            logger.info("STARTING CUSTOM TEXT TEST")
            logger.info("=" * 50)
            logger.info(f"Custom text: {custom_text}")
            
            # Create test XML structure with custom text
            test_xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<root>
    <PAGE>
        <SECT>
            <FILE>
                <STC>{custom_text}</STC>
            </FILE>
        </SECT>
    </PAGE>
</root>"""
            
            # Write test file
            test_file_path = self.input_dir / test_filename
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_xml_content)
            
            # Process the test file
            self.process_xml_file(test_file_path)
            
            # Read and display results
            output_file_path = self.output_dir / test_filename
            if output_file_path.exists():
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                stc_matches = re.findall(r'<STC>(.*?)</STC>', content, re.DOTALL)
                for stc_content in stc_matches:
                    stc_content = stc_content.strip()
                    logger.info(f"Original: {custom_text}")
                    logger.info(f"Result:   {stc_content}")
                    
                    # Show entities found
                    for entity_type in ['PER', 'LOC', 'ORG', 'MSC', 'TME']:
                        entities = re.findall(f'<{entity_type}>(.*?)</{entity_type}>', stc_content)
                        if entities:
                            logger.info(f"  {entity_type}: {entities}")
                
                logger.info("=" * 50)
                logger.info("CUSTOM TEXT TEST COMPLETED!")
                logger.info("=" * 50)
                
        except Exception as e:
            logger.error(f"Error in test_custom_text: {e}")

def main():
    """
    Main function to run the NER pipeline.
    """
    # Configure your Azure OpenAI credentials
    AZURE_ENDPOINT = "https://nvtph-mcf3aww2-eastus2.cognitiveservices.azure.com/"
    API_KEY = "3CeSRZXX37FMawuv7XkxeKbYjex79xMj5IwMRgzXUxoEfmJSXt5pJQQJ99BFACHYHv6XJ3w3AAAAACOGLBRA"
    API_VERSION = "2025-01-01-preview"
    
    # You can also use environment variables
    # AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    # API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not AZURE_ENDPOINT or not API_KEY:
        logger.error("Please configure Azure OpenAI credentials")
        return
    
    # Initialize pipeline
    pipeline = ChineseNERPipeline(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION
    )
    
    # Uncomment one of the following options:
    
    # Option 1: Test with default sample text
    # pipeline.test_single_stc()
    
    # Option 2: Test with custom text
    # custom_text = "了智 瑶 、 韩 康子 、 魏 桓 子 三 家 出 兵 围 住 晋 阳 , 又 引水 灌 城 ,城墙 头只 差 三 版 的 地 方 没有 被 淹没 , 锅 灶 都 被 泡 塌 , 鱼 圣 苞 生 , 人 民 仍 是没有 背叛 之 意 。"
    # pipeline.test_custom_text(custom_text)
    
    # Option 3: Process all files (comment out test functions above)
    pipeline.process_all_files()

if __name__ == "__main__":
    main()
