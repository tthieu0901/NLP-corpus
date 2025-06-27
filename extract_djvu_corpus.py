#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and build corpus from djvu.txt file
Creates organized text files with metadata from historical Chinese text
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('djvu_corpus_extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DjvuCorpusExtractor:
    """Extract and organize historical text corpus from djvu.txt file"""
    
    def __init__(self, input_file: str, output_dir: str = "ming_history_txt_files"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.metadata = {}
        self.chapters = []
          # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from the file content"""
        metadata = {}
        
        # Extract title from filename
        filename = self.input_file.name
        if "白话资治通鉴" in filename:
            metadata['title'] = "白话资治通鉴"
        
        # Extract volume information from filename
        volume_match = re.search(r'(\d+)', filename)
        if volume_match:
            metadata['volume'] = f"第{volume_match.group(1)}册"
        else:
            metadata['volume'] = "第1册"
            
        # Period covered (from filename)
        period_match = re.search(r'周威烈王二十三年.*?汉惠帝七', filename)
        if period_match:
            metadata['period'] = period_match.group(0)
        
        # Extract authors/editors from content
        authors = []
        editors = []
        
        # Look for main editors (主编)
        main_editor_pattern = r'主编[：:\s]*([^副\n]+)'
        main_editor_match = re.search(main_editor_pattern, content)
        if main_editor_match:
            editors_text = main_editor_match.group(1).strip()
            # Clean up and split
            editors_text = re.sub(r'["""\s]+', ' ', editors_text)
            main_editors = re.split(r'[\s,，]+', editors_text)
            editors.extend([ed.strip() for ed in main_editors if ed.strip()])
        
        # Look for deputy editors (副主编)
        deputy_editor_pattern = r'副主编[：:\s]*([^\n]+)'
        deputy_editor_match = re.search(deputy_editor_pattern, content)
        if deputy_editor_match:
            deputy_text = deputy_editor_match.group(1).strip()
            deputy_text = re.sub(r'["""\s]+', ' ', deputy_text)
            deputy_editors = re.split(r'[\s,，]+', deputy_text)
            editors.extend([ed.strip() for ed in deputy_editors if ed.strip()])
        
        # Clean up editors list
        editors = [ed for ed in editors if ed and len(ed) > 1 and not re.match(r'^[:\s"]+$', ed)]
        
        metadata['editors'] = ', '.join(editors[:6])  # Limit to first 6 editors
        
        # Extract publisher information
        publisher_info = []
        
        # Look for publisher
        publisher_patterns = [
            r'中华书局出版',
            r'新华书店.*?发行',
            r'北京第二新华印刷厂印刷'
        ]
        
        for pattern in publisher_patterns:
            if re.search(pattern, content):
                match = re.search(pattern, content)
                if match:
                    publisher_info.append(match.group(0))
        
        metadata['publisher'] = '; '.join(publisher_info) if publisher_info else '中华书局出版'
          # Extract publication date
        date_pattern = r'1993.*?年.*?月.*?第.*?版'
        date_match = re.search(date_pattern, content)
        if date_match:
            metadata['publication_date'] = date_match.group(0)
        else:
            metadata['publication_date'] = '1993年3月第1版'
        
        # Extract source description
        metadata['source'] = '白话资治通鉴djvu扫描版'
        
        return metadata
    
    def extract_volume_from_content(self, content_lines: List[str], start_idx: int) -> Optional[int]:
        """Extract the actual volume number from the content around the given index"""
        # Look at the current line and nearby lines for 第X卷 pattern
        search_range = range(max(0, start_idx - 2), min(len(content_lines), start_idx + 5))
        
        chinese_to_arabic = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6,
            '七': 7, '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12,
            '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, '十八': 18,
            '十九': 19, '二十': 20
        }
        
        for i in search_range:
            line = content_lines[i].strip()
            # Look for patterns like "第 X 卷" or "第X卷"
            volume_match = re.search(r'第\s*([一二三四五六七八九十]+)\s*卷', line)
            if volume_match:
                volume_chinese = volume_match.group(1)
                volume_num = chinese_to_arabic.get(volume_chinese)
                if volume_num:
                    return volume_num
        
        return None

    def find_content_start(self, lines: List[str]) -> int:
        """Find where the actual historical content starts"""
        for i, line in enumerate(lines):
            # Look for the start of the first volume
            if '资治通鉴第一卷' in line:
                return i
            # Look for Chapter 1 header pattern - this is the key fix
            if '第 一 卷' in line and '周 威 烈 王' in line:
                return max(0, i - 2)  # Start a bit before to include volume header
            # Alternative pattern - look for volume headers
            if re.search(r'第.*卷.*周纪.*威烈王', line):
                return i
        
        # If we can't find the exact start, look for content patterns
        for i, line in enumerate(lines):
            if re.search(r'周威烈王.*年.*公元前', line):
                return max(0, i - 5)  # Start a few lines before
        
        return len(lines) // 3  # Fallback: assume content starts after first third
    
    def extract_chapters(self, content: str) -> List[Dict]:
        """Extract individual pages from chapters in the content"""
        lines = content.split('\n')
        content_start = self.find_content_start(lines)
        content_lines = lines[content_start:]

        chapters = []        # Enhanced patterns to capture both chapter and page information
        # Pattern: page_number 第X卷 dynasty content (year)
        page_patterns = [
            # Pattern with page number at start: "218 第 十 卷 汉 高 帝 四 年 (前 203)"
            r'(\d{1,3})\s+第\s*([一二三四五六七八九十]+)\s*卷\s+([^(]+)\s*\([^)]+\)',
            # Alternative patterns
            r'(\d{1,3})\s+第\s*([一二三四五六七八九十]+)\s*卷\s+([^0-9]+)',
            r'第\s*([一二三四五六七八九十]+)\s*卷\s+([^(]+)\s*\([^)]+\)\s*(\d+)',
            r'第\s*([一二三四五六七八九十]+)\s*卷\s+([^0-9]+)\s+(\d+)',
            r'第\s*([一二三四五六七八九十]+)\s*卷\s*([^0-9]+)\s*(\d+)',
        ]
        
        # Convert Chinese numerals to Arabic
        chinese_to_arabic = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6,
            '七': 7, '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12
        }
        
        # Find page boundaries within chapters
        page_boundaries = []
        
        # First, look for the special first chapter pattern
        for i, line in enumerate(content_lines):
            line_clean = line.strip()
            if len(line_clean) < 5:
                continue
                
            # Special handling for Chapter 1
            if '资治通鉴第一卷' in line_clean:
                # Look for the next few lines for the actual chapter header
                for j in range(i, min(i + 5, len(content_lines))):
                    next_line = content_lines[j].strip()
                    if '周纪一' in next_line and '威烈王' in next_line:
                        # Extract year information
                        year_match = re.search(r'前\s*(\d+)', next_line)
                        page_match = re.search(r'(\d+)', next_line)
                        
                        page_num = "1"  # Default for first chapter
                        if page_match:
                            page_num = page_match.group(1)
                        
                        page_boundaries.append({
                            'start_idx': i,
                            'chapter_num': 1,
                            'page_num': page_num,
                            'title': '周纪一_威烈王二十三年',
                            'dynasty': '周纪',
                            'full_title': next_line
                        })
                        break
                break
        
        # Then look for regular patterns
        for i, line in enumerate(content_lines):
            line_clean = line.strip()            # Skip very short lines that are unlikely to be headers
            if len(line_clean) < 10:
                continue
                
            for pattern in page_patterns:
                match = re.search(pattern, line_clean)
                if match:
                    # Handle different pattern groups
                    if pattern.startswith(r'(\d{1,3})'):  # Pattern with page number first
                        page_num = match.group(1)
                        volume_num = match.group(2)
                        content_desc = match.group(3).strip()
                    else:  # Traditional patterns with page number last
                        volume_num = match.group(1)
                        content_desc = match.group(2).strip()
                        page_num = match.group(3)
                    
                    # Convert Chinese volume number to Arabic for chapter numbering
                    chapter_num = chinese_to_arabic.get(volume_num, len(page_boundaries) + 1)
                    
                    # Extract the actual volume number from the content around this line
                    # Look at the actual content to find 第X卷 pattern
                    actual_volume_num = self.extract_volume_from_content(content_lines, i)
                    if actual_volume_num:
                        chapter_num = actual_volume_num
                    
                    # Skip if we already have this chapter/page
                    if any(pb['chapter_num'] == chapter_num and pb['page_num'] == page_num for pb in page_boundaries):
                        continue
                    
                    # Clean up content description for title
                    title = content_desc.strip()
                    title = re.sub(r'\s+', '_', title)
                    title = re.sub(r'["""\[\]()（）]', '', title)
                    
                    # Extract dynasty info (周纪, 秦纪, 汉纪)
                    dynasty = ""
                    if '周' in title or chapter_num <= 5:
                        dynasty = "周纪"
                    elif '秦' in title or 6 <= chapter_num <= 8:
                        dynasty = "秦纪" 
                    elif '汉' in title or chapter_num >= 9:
                        dynasty = "汉纪"
                    else:
                        dynasty = "未知朝代"
                    
                    page_boundaries.append({
                        'start_idx': i,
                        'chapter_num': chapter_num,
                        'page_num': page_num,
                        'title': title,
                        'dynasty': dynasty,
                        'full_title': line_clean
                    })
                    break
        # If we found fewer than expected, look for simpler patterns
        if len(page_boundaries) < 10:
            logger.warning(f"Only found {len(page_boundaries)} page boundaries, looking for simpler patterns")
            
            # Look for lines that might contain volume and page info
            simple_patterns = [
                r'第\s*([一二三四五六七八九十]+)\s*卷.*?(\d{1,3})',
                r'(周|秦|汉).*?(\d{1,3})',
            ]
            
            for i, line in enumerate(content_lines):
                line_clean = line.strip()
                if len(line_clean) < 10 or len(line_clean) > 200:  # Skip very long or short lines
                    continue
                    
                for pattern in simple_patterns:
                    match = re.search(pattern, line_clean)
                    if match:
                        if '第' in match.group(0):
                            volume_num = match.group(1)
                            page_num = match.group(2)
                            chapter_num = chinese_to_arabic.get(volume_num, len(page_boundaries) + 1)
                        else:
                            page_num = match.group(2)
                            chapter_num = len(page_boundaries) + 1
                        
                        # Avoid duplicates
                        if not any(pb['start_idx'] == i for pb in page_boundaries):
                            dynasty = "周纪" if '周' in line_clean else "秦纪" if '秦' in line_clean else "汉纪"
                            title = line_clean[:50].replace(' ', '_')
                            
                            page_boundaries.append({
                                'start_idx': i,
                                'chapter_num': chapter_num,
                                'page_num': page_num,
                                'title': title,
                                'dynasty': dynasty,
                                'full_title': line_clean
                            })
                        break        # Sort by page number to maintain book order, but keep original volume-based chapter numbers
        page_boundaries.sort(key=lambda x: int(x['page_num']))
        
        # Extract content for each page/section
        for i, boundary in enumerate(page_boundaries):
            start_idx = boundary['start_idx']
            end_idx = page_boundaries[i + 1]['start_idx'] if i + 1 < len(page_boundaries) else len(content_lines)
            
            section_lines = content_lines[start_idx:end_idx]
            section_text = '\n'.join(section_lines).strip()
            
            if len(section_text) < 100:  # Skip very short sections
                continue
            
            # Clean up title for filename
            cleaned_title = boundary['title'].replace(' ', '_')
            cleaned_title = re.sub(r'[^\w\u4e00-\u9fff_]', '_', cleaned_title)
            cleaned_title = cleaned_title[:50]  # Limit length
            
            # Determine correct dynasty based on volume number
            dynasty = self.get_dynasty_by_volume(boundary['chapter_num'], boundary['title'])
            
            chapter_info = {
                'index': boundary['chapter_num'],
                'page_num': boundary['page_num'],
                'title': cleaned_title,
                'dynasty': dynasty,
                'content': section_text,
                'length': len(section_text),
                'full_title': boundary['full_title']
            }
            
            chapters.append(chapter_info)
        
        logger.info(f"Extracted {len(chapters)} pages across chapters")
        return chapters
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and other artifacts
        text = re.sub(r'\(\d+\)', '', text)
        text = re.sub(r'第.*?页', '', text)
        
        # Clean up punctuation
        text = re.sub(r'\s*([，。；：！？])\s*', r'\1', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[,，]{2,}', '，', text)
        text = re.sub(r'[.。]{2,}', '。', text)
        
        return text.strip()
    
    def create_filename(self, metadata: Dict, chapter_info: Dict) -> str:
        """Create filename in format: <page_number>_<chapter_number>_<title>.txt"""
        # Get page number
        page_num = chapter_info.get('page_num', '001')
        
        # Get chapter number with zero padding
        chapter_num = f"{chapter_info['index']:02d}"
        
        # Clean up chapter title for filename
        title = chapter_info['title']
        
        # Remove any invalid filename characters
        title = re.sub(r'[<>:"/\\|?*]', '_', title)
        title = re.sub(r'\s+', '_', title)
        title = re.sub(r'[（）()]', '', title)  # Remove parentheses
        title = re.sub(r'第.*?卷', '', title)  # Remove volume prefix
        title = title.strip('_')
          # Ensure title is not too long
        if len(title) > 30:
            title = title[:30]
        
        # Remove trailing underscores
        title = title.strip('_')
        
        filename = f"{page_num}_{chapter_num}_{title}.txt"
        
        return filename
    
    def save_chapter(self, chapter_info: Dict, metadata: Dict) -> str:
        """Save individual chapter to file"""
        filename = self.create_filename(metadata, chapter_info)
        filepath = self.output_dir / filename
        
        # Prepare content with metadata header
        content_parts = []        # Add metadata header
        content_parts.append("=" * 50)
        content_parts.append("文档元数据 (Document Metadata)")
        content_parts.append("=" * 50)
        content_parts.append(f"标题: {metadata.get('title', 'N/A')}")
        content_parts.append(f"卷册: {self.get_volume_by_chapter(chapter_info['index'])}")
        content_parts.append(f"章节: 第{chapter_info['index']:02d}章")
        content_parts.append(f"页码: {chapter_info['page_num']}")
        content_parts.append(f"时期: {metadata.get('period', 'N/A')}")
        content_parts.append(f"章节标题: {chapter_info['title']}")
        content_parts.append(f"编者: {metadata.get('editors', 'N/A')}")
        content_parts.append(f"出版社: {metadata.get('publisher', 'N/A')}")
        content_parts.append(f"出版日期: {metadata.get('publication_date', 'N/A')}")
        content_parts.append(f"来源: {metadata.get('source', 'N/A')}")
        content_parts.append("")
        content_parts.append("=" * 50)
        content_parts.append("正文内容 (Main Content)")
        content_parts.append("=" * 50)
        content_parts.append("")
        
        # Clean and add main content
        cleaned_content = self.clean_text(chapter_info['content'])
        content_parts.append(cleaned_content)
        
        # Write to file
        full_content = '\n'.join(content_parts)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            logger.info(f"Saved chapter: {filename} ({len(cleaned_content)} characters)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving chapter {filename}: {e}")
            return ""
    
    def create_summary_file(self, metadata: Dict, chapters: List[Dict]) -> str:
        """Create a summary file with corpus information"""
        summary_file = self.output_dir / "corpus_summary.txt"
        
        summary_parts = []
        summary_parts.append("白话资治通鉴语料库总结")
        summary_parts.append("=" * 40)
        summary_parts.append("")
        
        # Corpus metadata
        summary_parts.append("语料库元数据:")
        for key, value in metadata.items():
            summary_parts.append(f"  {key}: {value}")
        summary_parts.append("")
        
        # Chapter summary
        summary_parts.append(f"章节总数: {len(chapters)}")
        summary_parts.append(f"总字符数: {sum(ch['length'] for ch in chapters):,}")
        summary_parts.append("")
        
        # Chapter list
        summary_parts.append("章节列表:")
        for i, chapter in enumerate(chapters, 1):
            summary_parts.append(f"  {i:2d}. {chapter['title'][:30]} ({chapter['length']:,} 字符)")
        
        summary_content = '\n'.join(summary_parts)
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            logger.info(f"Created summary file: {summary_file}")
            return str(summary_file)
            
        except Exception as e:
            logger.error(f"Error creating summary file: {e}")
            return ""
    
    def extract_corpus(self) -> bool:
        """Main method to extract and organize the corpus"""
        try:
            logger.info(f"Starting corpus extraction from: {self.input_file}")
            
            # Read input file
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Read file: {len(content):,} characters")
            
            # Extract metadata
            self.metadata = self.extract_metadata(content)
            logger.info(f"Extracted metadata: {self.metadata}")
            
            # Extract chapters
            self.chapters = self.extract_chapters(content)
            logger.info(f"Extracted {len(self.chapters)} chapters")
            
            # Save chapters
            saved_files = []
            for chapter in self.chapters:
                filepath = self.save_chapter(chapter, self.metadata)
                if filepath:
                    saved_files.append(filepath)
            
            # Create summary
            summary_file = self.create_summary_file(self.metadata, self.chapters)
            if summary_file:
                saved_files.append(summary_file)
            
            logger.info(f"Successfully extracted corpus: {len(saved_files)} files created")
            return True
            
        except Exception as e:
            logger.error(f"Error during corpus extraction: {e}")
            return False

    def get_dynasty_by_volume(self, volume_num: int, title: str) -> str:
        """Determine dynasty based on volume number and content analysis"""
        # Clean up title for analysis
        title_clean = title.lower()
        
        # First, check explicit dynasty markers in the title
        if '周' in title or '威烈王' in title or '显王' in title or '慎靓王' in title:
            return "周纪"
        elif '秦' in title or '始皇' in title or '二世' in title:
            return "秦纪"
        elif '汉' in title or '高帝' in title or '惠帝' in title:
            return "汉纪"
        
        # Based on the typical structure of 资治通鉴:
        # Volumes 1-5: Zhou periods (周纪)
        # Volumes 6-8: Qin period (秦纪) 
        # Volumes 9+: Han period (汉纪)
        
        if volume_num <= 5:            return "周纪"
        elif volume_num <= 8:
            return "秦纪" 
        else:
            return "汉纪"
    
    def get_volume_by_chapter(self, chapter_num: int) -> str:
        """Determine the correct volume number based on the actual chapter/volume number"""
        # Since we're now using the original volume numbers from the content,
        # chapter_num already represents the correct volume number
        
        chinese_nums = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', 
                       '十一', '十二', '十三', '十四', '十五', '十六', '十七', '十八', '十九', '二十']
        
        if chapter_num <= len(chinese_nums):
            return f"第{chinese_nums[chapter_num-1]}卷"
        else:
            return f"第{chapter_num}卷"

def main():
    """Main execution function"""
    # File path
    input_file = r"c:\Users\nguyenphong\Downloads\study master\NLP\06.Minh Sử_MinhNguyen\06.Minh Sử\白话资治通鉴01—周威烈王二十三年.至.汉惠帝七_djvu.txt"
    
    # Create extractor
    extractor = DjvuCorpusExtractor(input_file)
    
    # Extract corpus
    success = extractor.extract_corpus()
    
    if success:
        print(f"\n✅ Corpus extraction completed successfully!")
        print(f"📁 Output directory: {extractor.output_dir.absolute()}")
        print(f"📊 Chapters extracted: {len(extractor.chapters)}")
        print(f"📄 Total files created: {len(list(extractor.output_dir.glob('*.txt')))}")
    else:
        print("\n❌ Corpus extraction failed. Check the log for details.")

if __name__ == "__main__":
    main()