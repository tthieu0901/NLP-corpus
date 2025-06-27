import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from lxml import etree
import tiktoken
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XMLTokenEstimator:
    def __init__(self, xml_folder: str = "xml", encoding_name: str = "cl100k_base"):
        """
        Initialize the token estimator.
        
        Args:
            xml_folder: Path to folder containing XML files
            encoding_name: Tiktoken encoding name (cl100k_base for GPT-4, GPT-3.5-turbo)
        """
        self.xml_folder = Path(xml_folder)
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Define the exact prompts used in the NER pipeline
        self.system_prompt = """你是一个专业的学术研究助手，专门处理中国古代历史文献的命名实体识别任务。

任务说明：
- 这是一项纯学术研究，涉及中国古代历史文献的语言学分析
- 文本来源于历史典籍，包含古代人名、地名、官职等学术研究内容
- 请以客观、学术的角度进行实体识别，不涉及任何现代政治或敏感话题

请识别以下实体类型：
- PER: 古代人名（如历史人物、官员等）
- LOC: 古代地名（如城市、国家、地区等）
- ORG: 古代机构名（如官署、学校等）
- MSC: 其他重要实体（如器物、典籍等）
- TME: 时间或日期表达

输出格式要求：
请严格按照JSON数组格式返回结果，每个实体包含起始位置、结束位置、文本内容和标签：
[
  {"start": 0, "end": 3, "text": "赵简子", "label": "PER"},
  {"start": 5, "end": 7, "text": "尹铎", "label": "PER"}
]

注意：这是历史文献研究，所有内容均为学术用途。"""

        self.user_prompt_template = """以下是需要进行学术分析的中国古代历史文献片段，请进行命名实体识别：

文本内容：{text}

请按照上述要求，以学术研究的角度识别其中的实体。"""
        
        # Pre-calculate prompt token counts (these are fixed for all requests)
        self.system_prompt_tokens = self.count_tokens(self.system_prompt)
        self.user_prompt_base_tokens = self.count_tokens(self.user_prompt_template.replace("{text}", ""))
        
        if not self.xml_folder.exists():
            raise ValueError(f"XML folder {xml_folder} does not exist")
    
    def normalize_chinese_text(self, spaced_text: str) -> str:
        """
        Normalize Chinese text by removing spaces between characters.
        
        Args:
            spaced_text: Chinese text with spaces between characters
        Returns:
            Normalized text without spaces between Chinese characters
        """
        normalized = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', spaced_text)
        return normalized.strip()
    
    def extract_text_from_xml(self, xml_file_path: Path) -> List[str]:
        """
        Extract all text content from STC elements in an XML file.
        
        Args:
            xml_file_path: Path to XML file
        Returns:
            List of text strings from STC elements
        """
        try:
            parser = etree.XMLParser(encoding='utf-8')
            tree = etree.parse(str(xml_file_path), parser)
            root = tree.getroot()
            
            # Find all STC elements
            stc_elements = root.xpath('.//STC')
            
            texts = []
            for stc in stc_elements:
                if stc.text:
                    text = stc.text.strip()
                    if text:
                        # Normalize Chinese text for accurate token counting
                        normalized_text = self.normalize_chinese_text(text)
                        texts.append(normalized_text)
            
            return texts
            
        except Exception as e:
            logger.error(f"Error parsing XML file {xml_file_path}: {e}")
            return []
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
        Returns:
            Number of tokens
        """
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error counting tokens for text: {e}")
            return 0
    
    def count_total_tokens_for_text(self, text: str) -> Dict[str, int]:
        """
        Count total tokens for a complete API call including system and user prompts.
        
        Args:
            text: The STC text content
            
        Returns:
            Dictionary with token breakdown
        """
        # Content tokens (the actual text)
        content_tokens = self.count_tokens(text)
        
        # User prompt tokens (base template + actual text)
        user_prompt = self.user_prompt_template.format(text=text)
        user_prompt_tokens = self.count_tokens(user_prompt)
        
        # Total input tokens for this API call
        total_input_tokens = self.system_prompt_tokens + user_prompt_tokens
        
        # Estimated output tokens (typical NER response)
        # Assume average 3-5 entities per text, ~50 tokens per response
        estimated_output_tokens = min(50 + (len(text) // 10), 200)  # Cap at 200 tokens
        
        return {
            'content_tokens': content_tokens,
            'system_prompt_tokens': self.system_prompt_tokens,
            'user_prompt_tokens': user_prompt_tokens,
            'total_input_tokens': total_input_tokens,
            'estimated_output_tokens': estimated_output_tokens,
            'total_tokens': total_input_tokens + estimated_output_tokens
        }
    
    def estimate_file_tokens(self, xml_file_path: Path) -> Dict:
        """
        Estimate tokens for a single XML file including all prompt overhead.
        
        Args:
            xml_file_path: Path to XML file
            
        Returns:
            Dictionary with comprehensive token statistics
        """
        logger.info(f"Processing file: {xml_file_path}")
        
        texts = self.extract_text_from_xml(xml_file_path)
        
        if not texts:
            return {
                'filename': xml_file_path.name,
                'stc_count': 0,
                'content_tokens': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'avg_tokens_per_stc': 0,
                'total_characters': 0,
                'avg_characters_per_stc': 0
            }
        
        # Calculate comprehensive token statistics
        all_token_stats = [self.count_total_tokens_for_text(text) for text in texts]
        character_counts = [len(text) for text in texts]
        
        # Aggregate statistics
        total_content_tokens = sum(stats['content_tokens'] for stats in all_token_stats)
        total_input_tokens = sum(stats['total_input_tokens'] for stats in all_token_stats)
        total_output_tokens = sum(stats['estimated_output_tokens'] for stats in all_token_stats)
        total_tokens = sum(stats['total_tokens'] for stats in all_token_stats)
        total_characters = sum(character_counts)
        
        stats = {
            'filename': xml_file_path.name,
            'stc_count': len(texts),
            'content_tokens': total_content_tokens,
            'system_prompt_tokens_per_call': self.system_prompt_tokens,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'avg_total_tokens_per_stc': round(total_tokens / len(texts), 2),
            'avg_content_tokens_per_stc': round(total_content_tokens / len(texts), 2),
            'total_characters': total_characters,
            'avg_characters_per_stc': round(total_characters / len(texts), 2),
            'max_content_tokens_per_stc': max(stats['content_tokens'] for stats in all_token_stats) if all_token_stats else 0,
            'min_content_tokens_per_stc': min(stats['content_tokens'] for stats in all_token_stats) if all_token_stats else 0
        }
        
        return stats
    
    def estimate_all_files(self) -> Dict:
        """
        Estimate tokens for all XML files in the folder with complete cost analysis.
        
        Returns:
            Dictionary with overall statistics and per-file breakdown
        """
        xml_files = list(self.xml_folder.glob("*.xml"))
        
        if not xml_files:
            logger.warning(f"No XML files found in {self.xml_folder}")
            return {'files': [], 'summary': {}}
        
        logger.info(f"Found {len(xml_files)} XML files to process")
        
        file_stats = []
        total_content_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        total_stc_count = 0
        total_characters = 0
        
        for xml_file in xml_files:
            stats = self.estimate_file_tokens(xml_file)
            file_stats.append(stats)
            
            total_content_tokens += stats['content_tokens']
            total_input_tokens += stats['total_input_tokens']
            total_output_tokens += stats['total_output_tokens']
            total_tokens += stats['total_tokens']
            total_stc_count += stats['stc_count']
            total_characters += stats['total_characters']
        
        # Calculate summary statistics
        summary = {
            'total_files': len(xml_files),
            'total_stc_elements': total_stc_count,
            'total_api_calls': total_stc_count,  # One API call per STC
            'system_prompt_tokens': self.system_prompt_tokens,
            'total_content_tokens': total_content_tokens,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'total_characters': total_characters,
            'avg_total_tokens_per_file': round(total_tokens / len(xml_files), 2) if xml_files else 0,
            'avg_total_tokens_per_stc': round(total_tokens / total_stc_count, 2) if total_stc_count > 0 else 0,
            'avg_content_tokens_per_stc': round(total_content_tokens / total_stc_count, 2) if total_stc_count > 0 else 0,
            'avg_characters_per_token': round(total_characters / total_content_tokens, 2) if total_content_tokens > 0 else 0,
            'prompt_overhead_percentage': round((total_input_tokens - total_content_tokens) / total_input_tokens * 100, 2) if total_input_tokens > 0 else 0
        }
        
        return {
            'files': file_stats,
            'summary': summary
        }
    
    def print_detailed_report(self, results: Dict) -> None:
        """
        Print a comprehensive report including prompt overhead and cost analysis.
        
        Args:
            results: Results dictionary from estimate_all_files()
        """
        print("\n" + "="*100)
        print("COMPREHENSIVE XML TOKEN ESTIMATION REPORT (Including NER Pipeline Overhead)")
        print("="*100)
        
        # Summary
        summary = results['summary']
        print(f"\nSUMMARY:")
        print(f"  Total Files: {summary['total_files']}")
        print(f"  Total STC Elements: {summary['total_stc_elements']}")
        print(f"  Total API Calls: {summary['total_api_calls']}")
        print(f"  System Prompt Tokens: {summary['system_prompt_tokens']} (per call)")
        print(f"  Total Content Tokens: {summary['total_content_tokens']:,}")
        print(f"  Total Input Tokens: {summary['total_input_tokens']:,}")
        print(f"  Total Output Tokens: {summary['total_output_tokens']:,}")
        print(f"  TOTAL TOKENS: {summary['total_tokens']:,}")
        print(f"  Prompt Overhead: {summary['prompt_overhead_percentage']}%")
        print(f"  Average Total Tokens per STC: {summary['avg_total_tokens_per_stc']}")
        print(f"  Average Content Tokens per STC: {summary['avg_content_tokens_per_stc']}")
        
        # Cost estimation for GPT-4.1-nano
        input_cost_per_1m = 0.11  # USD per 1M input tokens
        cached_input_cost_per_1m = 0.028  # USD per 1M cached input tokens
        output_cost_per_1m = 0.44  # USD per 1M output tokens
        
        # Calculate costs (system prompt can be cached after first call)
        system_prompt_tokens_total = summary['system_prompt_tokens'] * summary['total_api_calls']
        user_prompt_tokens_total = summary['total_input_tokens'] - system_prompt_tokens_total
        
        # Cost with system prompt caching (cached after first call)
        first_call_input_cost = (summary['system_prompt_tokens'] / 1_000_000) * input_cost_per_1m
        cached_system_cost = ((summary['total_api_calls'] - 1) * summary['system_prompt_tokens'] / 1_000_000) * cached_input_cost_per_1m
        user_prompt_cost = (user_prompt_tokens_total / 1_000_000) * input_cost_per_1m
        output_cost = (summary['total_output_tokens'] / 1_000_000) * output_cost_per_1m
        
        total_cost_with_cache = first_call_input_cost + cached_system_cost + user_prompt_cost + output_cost
        
        # Cost without caching (all input tokens at full price)
        input_cost_no_cache = (summary['total_input_tokens'] / 1_000_000) * input_cost_per_1m
        total_cost_no_cache = input_cost_no_cache + output_cost
        
        print(f"\nCOST ESTIMATION (GPT-4.1-nano):")
        print(f"  WITHOUT PROMPT CACHING:")
        print(f"    Input Cost: ${input_cost_no_cache:.4f}")
        print(f"    Output Cost: ${output_cost:.4f}")
        print(f"    TOTAL: ${total_cost_no_cache:.4f}")
        print(f"  WITH SYSTEM PROMPT CACHING:")
        print(f"    First Call System Prompt: ${first_call_input_cost:.4f}")
        print(f"    Cached System Prompts: ${cached_system_cost:.4f}")
        print(f"    User Prompts: ${user_prompt_cost:.4f}")
        print(f"    Output Cost: ${output_cost:.4f}")
        print(f"    TOTAL WITH CACHING: ${total_cost_with_cache:.4f}")
        print(f"    SAVINGS FROM CACHING: ${total_cost_no_cache - total_cost_with_cache:.4f} ({((total_cost_no_cache - total_cost_with_cache) / total_cost_no_cache * 100):.1f}%)")
        
        # Per-file breakdown
        print(f"\nPER-FILE BREAKDOWN:")
        print(f"{'Filename':<25} {'STCs':<6} {'Content':<8} {'Input':<8} {'Output':<8} {'Total':<8} {'Cost$':<7}")
        print("-" * 100)
        
        for file_stat in results['files']:
            # Calculate per-file cost with caching
            file_system_tokens = file_stat['system_prompt_tokens_per_call'] * file_stat['stc_count']
            file_user_tokens = file_stat['total_input_tokens'] - file_system_tokens
            
            file_first_call_cost = (file_stat['system_prompt_tokens_per_call'] / 1_000_000) * input_cost_per_1m
            file_cached_cost = ((file_stat['stc_count'] - 1) * file_stat['system_prompt_tokens_per_call'] / 1_000_000) * cached_input_cost_per_1m if file_stat['stc_count'] > 1 else 0
            file_user_cost = (file_user_tokens / 1_000_000) * input_cost_per_1m
            file_output_cost = (file_stat['total_output_tokens'] / 1_000_000) * output_cost_per_1m
            
            file_total_cost = file_first_call_cost + file_cached_cost + file_user_cost + file_output_cost
            
            print(f"{file_stat['filename']:<25} "
                  f"{file_stat['stc_count']:<6} "
                  f"{file_stat['content_tokens']:<8} "
                  f"{file_stat['total_input_tokens']:<8} "
                  f"{file_stat['total_output_tokens']:<8} "
                  f"{file_stat['total_tokens']:<8} "
                  f"{file_total_cost:<7.4f}")
        
        print("="*100)
        print(f"NOTE: This includes system prompt ({summary['system_prompt_tokens']} tokens) + user prompt template for each STC")
        print("Output tokens are estimated based on typical NER responses (actual may vary)")
        print("Costs calculated with system prompt caching enabled (cached after first API call)")
    
    def save_report_to_file(self, results: Dict, output_file: str = "token_estimation_report.txt") -> None:
        """
        Save the token estimation report to a file.
        
        Args:
            results: Results dictionary from estimate_all_files()
            output_file: Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Redirect print output to file
                import sys
                old_stdout = sys.stdout
                sys.stdout = f
                
                self.print_detailed_report(results)
                
                # Restore stdout
                sys.stdout = old_stdout
                
            logger.info(f"Report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving report to file: {e}")

def main():
    """
    Main function to run token estimation.
    """
    try:
        # Initialize estimator
        estimator = XMLTokenEstimator(xml_folder="xml")
        
        # Estimate tokens for all files
        logger.info("Starting token estimation...")
        results = estimator.estimate_all_files()
        
        # Print detailed report
        estimator.print_detailed_report(results)
        
        # Save report to file
        estimator.save_report_to_file(results)
        
        logger.info("Token estimation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
