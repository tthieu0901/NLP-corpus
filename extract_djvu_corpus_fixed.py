#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed extraction script for djvu.txt corpus
Based on actual page boundary patterns in the file
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('djvu_corpus_extraction_fixed.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def chinese_to_number(chinese_num: str) -> int:
    """Convert Chinese numerals to Arabic numbers"""
    chinese_to_arabic = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6,
        '七': 7, '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12,
        '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17,
        '十八': 18, '十九': 19, '二十': 20
    }
    return chinese_to_arabic.get(chinese_num, 1)


def extract_title_from_line(line: str) -> str:
    """Extract and clean title from a line"""
    # Remove year patterns like (前 403)
    title = re.sub(r'\([^)]+\)', '', line)
    # Remove volume patterns like 第 一 卷
    title = re.sub(r'第\s*[一二三四五六七八九十]+\s*卷\s*', '', title)
    # Clean up spaces and special characters
    title = re.sub(r'\s+', '_', title.strip())
    title = re.sub(r'["""''「」（）()[\]{}]', '', title)
    # Remove any remaining numeric patterns at the start
    title = re.sub(r'^\d+\s*', '', title)
    return title


def get_dynasty_from_content(content: str, chapter_num: int) -> str:
    """Determine dynasty based on content and chapter number"""
    content_lower = content.lower()
    
    # Check for explicit dynasty markers in content
    if any(word in content for word in ['周威烈王', '周安王', '周烈王', '周显王', '周慎靓王', '周赧王']):
        return "周纪"
    elif any(word in content for word in ['秦始皇', '秦二世', '秦朝', '嬴政']):
        return "秦纪"
    elif any(word in content for word in ['汉高帝', '汉惠帝', '汉文帝', '刘邦', '刘盈']):
        return "汉纪"
    
    # Fallback based on typical chapter ranges
    if chapter_num <= 5:
        return "周纪"
    elif 6 <= chapter_num <= 8:
        return "秦纪"
    else:
        return "汉纪"


def get_volume_by_chapter(chapter_num: int) -> str:
    """Get volume name based on chapter number"""
    volume_mapping = {
        1: "第一卷", 2: "第二卷", 3: "第三卷", 4: "第四卷", 5: "第五卷",
        6: "第六卷", 7: "第七卷", 8: "第八卷", 9: "第九卷", 10: "第十卷",
        11: "第十一卷", 12: "第十二卷"
    }
    return volume_mapping.get(chapter_num, f"第{chapter_num}卷")


def extract_pages_from_file(file_path: str) -> List[Dict]:
    """Extract pages based on actual page boundary patterns"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    pages = []
    
    current_page_content = []
    current_page_info = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Look for page boundary patterns:
        # Pattern 1: "4 第 一 卷 周 威 烈 王 二 十 三 年 (前 403)" (page number at start)
        page_match = re.match(r'^(\d+)\s+第\s*([一二三四五六七八九十]+)\s*卷\s*(.+)', line)
        
        # Pattern 2: "第 十 卷 汉 高 帝 三 年 (前 204) 211" (page number at end)
        if not page_match:
            page_match_end = re.search(r'第\s*([一二三四五六七八九十]+)\s*卷\s*([^0-9]*(?:\([^)]+\))?)\s*(\d+)\s*$', line)
            if page_match_end:
                # Reorganize groups to match the first pattern format
                page_match = type('Match', (), {
                    'group': lambda self, n: [None, page_match_end.group(3), page_match_end.group(1), page_match_end.group(2).strip()][n]
                })()
        
        if page_match:
            # Save previous page if it exists
            if current_page_info and current_page_content:
                content_text = '\n'.join(current_page_content).strip()
                if len(content_text) > 50:  # Only save pages with substantial content
                    pages.append({
                        'page_num': current_page_info['page_num'],
                        'chapter_num': current_page_info['chapter_num'],
                        'volume_chinese': current_page_info['volume_chinese'],
                        'title': current_page_info['title'],
                        'dynasty': get_dynasty_from_content(content_text, current_page_info['chapter_num']),
                        'content': content_text
                    })
            
            # Start new page
            page_num = int(page_match.group(1))
            volume_chinese = page_match.group(2)
            title_part = page_match.group(3).strip()
            
            current_page_info = {
                'page_num': page_num,
                'chapter_num': chinese_to_number(volume_chinese),
                'volume_chinese': volume_chinese,
                'title': extract_title_from_line(title_part)
            }
            current_page_content = []
            
        else:
            # Add content to current page (skip very short lines)
            if current_page_info and len(line) > 2:
                # Skip obvious page headers and footers
                if not re.match(r'^\d+\s*$', line) and '白话资治通鉴' not in line:
                    current_page_content.append(line)
    
    # Add the last page
    if current_page_info and current_page_content:
        content_text = '\n'.join(current_page_content).strip()
        if len(content_text) > 50:
            pages.append({
                'page_num': current_page_info['page_num'],
                'chapter_num': current_page_info['chapter_num'],
                'volume_chinese': current_page_info['volume_chinese'],
                'title': current_page_info['title'],
                'dynasty': get_dynasty_from_content(content_text, current_page_info['chapter_num']),
                'content': content_text
            })
    
    return pages


def save_page_to_file(page_data: Dict, output_dir: Path, base_metadata: Dict):
    """Save a single page to a file with metadata"""
    # Create filename: page_chapter_title.txt
    filename = f"{page_data['page_num']:03d}_{page_data['chapter_num']:02d}_{page_data['title']}.txt"
    filename = filename.replace('/', '_').replace('\\', '_')
    
    file_path = output_dir / filename
    
    # Create file content with metadata
    content = f"""==================================================
文档元数据 (Document Metadata)
==================================================
标题: {base_metadata.get('title', '白话资治通鉴')}
卷册: {get_volume_by_chapter(page_data['chapter_num'])}
章节: 第{page_data['chapter_num']:02d}章
页码: {page_data['page_num']}
时期: {base_metadata.get('period', '周威烈王二十三年.至.汉惠帝七')}
章节标题: {page_data['title']}
编者: {base_metadata.get('editors', '')}
出版社: {base_metadata.get('publisher', '中华书局出版')}
出版日期: {base_metadata.get('publication_date', '1993年3月第1版')}
来源: {base_metadata.get('source', '白话资治通鉴djvu扫描版')}

==================================================
正文内容 (Main Content)
==================================================

{page_data['content']}
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Saved page {page_data['page_num']} of chapter {page_data['chapter_num']} to {filename}")


def main():
    """Main extraction function"""
    input_file = "白话资治通鉴01—周威烈王二十三年.至.汉惠帝七_djvu.txt"
    output_dir = Path("ming_history_txt_files")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Basic metadata
    base_metadata = {
        'title': '白话资治通鉴',
        'period': '周威烈王二十三年.至.汉惠帝七',
        'editors': '沈志华, 张宏儒',
        'publisher': '中华书局出版',
        'publication_date': '1993年3月第1版',
        'source': '白话资治通鉴djvu扫描版'
    }
    
    # Extract pages
    logger.info(f"Starting extraction from {input_file}")
    pages = extract_pages_from_file(input_file)
    
    logger.info(f"Found {len(pages)} pages")
    
    # Sort pages by page number to ensure correct ordering
    pages.sort(key=lambda x: x['page_num'])
    
    # Save each page
    for page_data in pages:
        save_page_to_file(page_data, output_dir, base_metadata)
    
    # Log summary
    page_nums = [p['page_num'] for p in pages]
    chapter_nums = [p['chapter_num'] for p in pages]
    
    logger.info(f"Extraction complete!")
    logger.info(f"Page range: {min(page_nums)} - {max(page_nums)}")
    logger.info(f"Chapter range: {min(chapter_nums)} - {max(chapter_nums)}")
    logger.info(f"Total files created: {len(pages)}")
    
    # Verify ordering
    page_num_sorted = sorted(page_nums)
    chapter_num_sorted = sorted(chapter_nums)
    
    if page_nums == page_num_sorted and chapter_nums == chapter_num_sorted:
        logger.info("✓ All pages and chapters are in correct order")
    else:
        logger.warning("⚠ Ordering issues detected:")
        if page_nums != page_num_sorted:
            logger.warning(f"Page numbers not in order: first few = {page_nums[:10]}")
        if chapter_nums != chapter_num_sorted:
            logger.warning(f"Chapter numbers not in order: {chapter_nums}")


if __name__ == "__main__":
    main()
