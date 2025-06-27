#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction script for preliminary content (frontmatter, prefaces, guidelines, etc.)
from djvu.txt file before the main historical content begins.
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
        logging.FileHandler('preliminary_extraction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_metadata_header(title: str, sec_id: int, name: str, page_id: int) -> str:
    """Create standardized metadata header for each file"""
    return f"""==================================================
文档元数据 (Document Metadata)
==================================================
标题: 白话资治通鉴
类别: {title}
章节: {sec_id:02d}
节名: {name}
段落: {page_id:03d}
编者: 沈志华, 张宏儒
出版社: 中华书局出版
出版日期: 1993年3月第1版
来源: 白话资治通鉴djvu扫描版

==================================================
正文内容 (Main Content)
==================================================

"""


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Fix common OCR issues
    text = re.sub(r'([。，；：！？])\s+', r'\1', text)  # Remove space after punctuation
    text = re.sub(r'\s+([。，；：！？])', r'\1', text)  # Remove space before punctuation
    return text


def detect_section_boundaries(lines: List[str]) -> List[Tuple[int, str, str]]:
    """
    Detect major section boundaries in the preliminary content
    Returns list of (line_number, title, name) tuples
    """
    boundaries = []
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        # Cover and publication info (beginning)
        if i < 80 and ('白话' in line_clean or '中华书局' in line_clean or '主编' in line_clean):
            if not any(b[1] == 'frontmatter' for b in boundaries):
                boundaries.append((i, 'frontmatter', 'cover_and_publication'))
        
        # Editorial committee section
        elif '编委' in line_clean or '译稿人' in line_clean or '审校人' in line_clean:
            if not any(b[2] == 'editorial_committee' for b in boundaries):
                boundaries.append((i, 'frontmatter', 'editorial_committee'))
        
        # Zhou Yiliang preface
        elif '今译系列与' in line_clean and '古籍今译' in line_clean:
            boundaries.append((i, 'preface', 'zhou_yiliang_preface'))
        
        # Editorial guidelines
        elif '编译说明' in line_clean:
            boundaries.append((i, 'guidelines', 'translation_guidelines'))
        
        # Imperial preface
        elif '御制' in line_clean and '资治通鉴' in line_clean and '序' in line_clean:
            boundaries.append((i, 'imperial', 'song_emperor_preface'))
        
        # Table of contents
        elif '目录' in line_clean and len(line_clean) < 10:
            boundaries.append((i, 'contents', 'table_of_contents'))
        
        # Main content begins (stop processing)
        elif '资治通鉴第一卷' in line_clean or ('周纪一' in line_clean and '威烈王' in line_clean):
            boundaries.append((i, 'END', 'main_content_begins'))
            break
    
    return boundaries


def split_into_paragraphs(text: str, max_chars: int = 800) -> List[str]:
    """
    Split text into logical paragraphs with reasonable length
    """
    # First split by natural paragraph breaks
    paragraphs = []
    current_para = ""
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            if current_para:
                paragraphs.append(current_para.strip())
                current_para = ""
            continue
        
        # Check if adding this line would exceed max_chars
        if len(current_para) + len(line) + 1 > max_chars and current_para:
            paragraphs.append(current_para.strip())
            current_para = line
        else:
            if current_para:
                current_para += " " + line
            else:
                current_para = line
    
    # Add the last paragraph
    if current_para:
        paragraphs.append(current_para.strip())
    
    # Filter out very short paragraphs (likely OCR artifacts)
    meaningful_paragraphs = []
    for para in paragraphs:
        if len(para) > 50:  # Minimum meaningful content
            meaningful_paragraphs.append(clean_text(para))
    
    return meaningful_paragraphs


def process_section(lines: List[str], start_idx: int, end_idx: int, 
                   title: str, name: str, output_dir: Path) -> int:
    """
    Process a section of content and save as individual paragraph files
    Returns the number of files created
    """
    section_text = '\n'.join(lines[start_idx:end_idx])
    paragraphs = split_into_paragraphs(section_text)
    
    files_created = 0
    sec_id = 1  # Could be made dynamic based on section type
    
    for page_id, paragraph in enumerate(paragraphs, 1):
        if len(paragraph.strip()) < 30:  # Skip very short paragraphs
            continue
            
        filename = f"{title}_{sec_id:02d}_{name}_{page_id:03d}.txt"
        filepath = output_dir / filename
        
        metadata = create_metadata_header(title, sec_id, name, page_id)
        full_content = metadata + paragraph
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(full_content)
            files_created += 1
            logger.info(f"Created: {filename}")
        except Exception as e:
            logger.error(f"Error creating {filename}: {e}")
    
    return files_created


def extract_preliminary_content(input_file: str, output_dir: str):
    """
    Main function to extract and organize preliminary content
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Reading input file: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return
    
    # Detect section boundaries
    logger.info("Detecting section boundaries...")
    boundaries = detect_section_boundaries(lines)
    
    logger.info(f"Found {len(boundaries)} sections:")
    for line_num, title, name in boundaries:
        logger.info(f"  Line {line_num}: {title} - {name}")
    
    # Process each section
    total_files = 0
    for i, (start_line, title, name) in enumerate(boundaries):
        if title == 'END':
            break
            
        # Determine end line for this section
        if i + 1 < len(boundaries):
            end_line = boundaries[i + 1][0]
        else:
            end_line = len(lines)
        
        logger.info(f"Processing section: {title} - {name} (lines {start_line}-{end_line})")
        
        files_created = process_section(
            lines, start_line, end_line, title, name, output_path
        )
        total_files += files_created
        
        logger.info(f"  Created {files_created} files for this section")
    
    logger.info(f"Extraction complete! Total files created: {total_files}")
    logger.info(f"Files saved to: {output_path}")


def main():
    """Main execution function"""
    input_file = "白话资治通鉴01—周威烈王二十三年.至.汉惠帝七_djvu.txt"
    output_dir = "ming_history_txt_others"
    
    # Get the current script directory
    script_dir = Path(__file__).parent
    input_path = script_dir / input_file
    output_path = script_dir / output_dir
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info("=" * 60)
    logger.info("Starting preliminary content extraction")
    logger.info("=" * 60)
    
    extract_preliminary_content(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
