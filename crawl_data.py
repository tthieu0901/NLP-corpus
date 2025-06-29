#!/usr/bin/env python3
"""
Cleaned-up Ming History Crawler
Crawls chapters from https://www.xuges.com/ls/mingshi/index.htm, splits each chapter into paragraphs, and saves each paragraph as a separate file with metadata.
"""

import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ming_history_crawl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MingHistoryCrawler:
    def __init__(self, base_url: str = "https://www.xuges.com/ls/mingshi/index.htm", 
                 output_dir: str = "ming_history_chapters", test_mode: bool = False):
        """
        Initialize the Ming History crawler.
        """
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.test_mode = test_mode
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.output_dir.mkdir(exist_ok=True)
        self.chapter_types = {
            '本纪': 'benji',
            '志': 'zhi', 
            '表': 'biao',
            '列传': 'liezhuan'
        }

    @staticmethod
    def clean_filename(filename: str) -> str:
        """Clean filename by removing unsafe characters and limiting length."""
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        filename = re.sub(r'\s+', '_', filename.strip())
        filename = re.sub(r'\.+', '.', filename)
        filename = re.sub(r'_+', '_', filename)
        if len(filename) > 200:
            name, ext = filename[:190], ''
            if '.' in filename:
                name, ext = filename[:190].rsplit('.', 1)
                ext = '.' + ext
            filename = name + ext
        return filename

    def get_page_content(self, url: str, encoding: str = 'gbk') -> Optional[BeautifulSoup]:
        """Fetch and parse page content."""
        try:
            response = self.session.get(url, timeout=30)
            response.encoding = encoding
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def extract_chapter_links(self) -> List[Dict]:
        """Extract all chapter links from the main index page."""
        logger.info(f"Fetching main index page: {self.base_url}")
        soup = self.get_page_content(self.base_url)
        if not soup:
            logger.error("Failed to fetch main index page")
            return []
        chapters = []
        current_chapter_type = None
        volume_counters = {}
        try:
            links = soup.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                text = link.get_text(strip=True)
                if not href or not text or href.startswith('#'):
                    continue
                chapter_type = self.detect_chapter_type(link, text)
                if chapter_type:
                    current_chapter_type = chapter_type
                    if chapter_type not in volume_counters:
                        volume_counters[chapter_type] = 0
                if self.is_chapter_link(href, text):
                    volume_counters[current_chapter_type] = volume_counters.get(current_chapter_type, 0) + 1
                    chapter_info = {
                        'title': text,
                        'url': urljoin(self.base_url, href),
                        'chapter_type': current_chapter_type or '未分类',
                        'volume': volume_counters.get(current_chapter_type, 1),
                        'original_href': href
                    }
                    chapters.append(chapter_info)
                    logger.info(f"Found chapter: {chapter_info['chapter_type']} - Vol.{chapter_info['volume']} - {text}")
            logger.info(f"Total chapters found: {len(chapters)}")
            if self.test_mode:
                chapters = chapters[:5]
                logger.info(f"Test mode: Limited to {len(chapters)} chapters")
            return chapters
        except Exception as e:
            logger.error(f"Error extracting chapter links: {e}")
            return []

    def detect_chapter_type(self, link_element, link_text: str) -> Optional[str]:
        """Detect chapter type from link context or text."""
        for chapter_type in self.chapter_types.keys():
            if chapter_type in link_text:
                return chapter_type
        parent = link_element.parent
        if parent:
            parent_text = parent.get_text()
            for chapter_type in self.chapter_types.keys():
                if chapter_type in parent_text:
                    return chapter_type
        prev_elements = link_element.find_previous_siblings()
        for elem in prev_elements[:3]:
            elem_text = elem.get_text() if hasattr(elem, 'get_text') else str(elem)
            for chapter_type in self.chapter_types.keys():
                if chapter_type in elem_text:
                    return chapter_type
        return None

    @staticmethod
    def is_chapter_link(href: str, text: str) -> bool:
        """Determine if a link is a chapter link."""
        if href.endswith('.htm') or href.endswith('.html'):
            if 'index' not in href.lower() and len(text) > 2:
                return True
        return False

    def extract_chapter_content(self, chapter_info: Dict) -> Optional[Dict]:
        """Extract content from a chapter page."""
        url = chapter_info['url']
        logger.info(f"Extracting content from: {url}")
        soup = self.get_page_content(url)
        if not soup:
            return None
        try:
            title_element = soup.find('td', class_='t50')
            true_title = title_element.get_text(strip=True) if title_element else chapter_info['title']
            content = ""
            tb_table = soup.find('table', class_='tb')
            if tb_table:
                rows = tb_table.find_all('tr')
                if len(rows) >= 4:
                    content_row = rows[3]
                    content = str(content_row)
                else:
                    content = str(tb_table)
            if not content:
                main_content = soup.find('div', class_='content') or soup.find('div', id='content')
                if main_content:
                    content = str(main_content)
                else:
                    body = soup.find('body')
                    if body:
                        content = str(body)
            return {
                'title': true_title,
                'chapter_type': chapter_info['chapter_type'],
                'volume': chapter_info['volume'],
                'url': url,
                'content': content,
                'original_title': chapter_info['title']
            }
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None

    @staticmethod
    def split_into_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs using <br /><br /> and similar HTML break patterns, then remove HTML tags from each paragraph."""
        import re
        # Split on <br><br>, <br /><br />, <br> <br>, etc.
        paragraphs = re.split(r'<br\s*/?>\s*<br\s*/?>', text)
        # Remove all HTML tags from each paragraph
        cleaned = [re.sub(r'<[^>]+>', '', p).strip() for p in paragraphs if p.strip()]
        return [p for p in cleaned if len(p) > 10]

    def save_chapter(self, chapter_data: Dict) -> bool:
        """Save chapter data to multiple paragraph files."""
        try:
            paragraphs = self.split_into_paragraphs(chapter_data['content'])
            if not paragraphs:
                logger.warning(f"No paragraphs found for chapter: {chapter_data['title']}")
                return False
            for idx, para in enumerate(paragraphs, 1):
                filename = f"{chapter_data['chapter_type']}_{chapter_data['volume']}_{chapter_data['title']}_{idx}.txt"
                filename = self.clean_filename(filename)
                filepath = self.output_dir / filename
                file_content = (
                    f"标题: {chapter_data['title']}\n"
                    f"章节类型: {chapter_data['chapter_type']}\n"
                    f"卷数: {chapter_data['volume']}\n"
                    f"URL: {chapter_data['url']}\n"
                    f"原始标题: {chapter_data['original_title']}\n"
                    f"段落序号: {idx}\n"
                    f"\n{'='*50}\n\n{para}\n"
                )
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                logger.info(f"Saved paragraph {idx} of chapter to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving chapter {chapter_data['title']}: {e}")
            return False

    def crawl_chapter(self, chapter_info: Dict) -> bool:
        """Crawl a single chapter (extract content and save)."""
        try:
            chapter_data = self.extract_chapter_content(chapter_info)
            if not chapter_data:
                return False
            return self.save_chapter(chapter_data)
        except Exception as e:
            logger.error(f"Error crawling chapter {chapter_info['title']}: {e}")
            return False

    def crawl_all_chapters(self, max_workers: int = 5) -> None:
        """Crawl all chapters using multithreading."""
        chapters = self.extract_chapter_links()
        if not chapters:
            logger.error("No chapters found to crawl")
            return
        logger.info(f"Starting to crawl {len(chapters)} chapters with {max_workers} workers")
        successful_downloads = 0
        failed_downloads = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chapter = {
                executor.submit(self.crawl_chapter, chapter): chapter 
                for chapter in chapters
            }
            for future in as_completed(future_to_chapter):
                chapter = future_to_chapter[future]
                try:
                    success = future.result()
                    if success:
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                except Exception as e:
                    logger.error(f"Exception crawling {chapter['title']}: {e}")
                    failed_downloads += 1
                time.sleep(0.1)
        logger.info(f"Crawling completed. Successful: {successful_downloads}, Failed: {failed_downloads}")
        self.generate_summary(successful_downloads, failed_downloads, chapters)

    def generate_summary(self, successful: int, failed: int, chapters: List[Dict]) -> None:
        """Generate a summary of the crawling process."""
        summary_file = self.output_dir / "crawl_summary.txt"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"明史爬取总结\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"爬取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总章节数: {len(chapters)}\n")
                f.write(f"成功下载: {successful}\n")
                f.write(f"失败数量: {failed}\n")
                f.write(f"成功率: {successful/len(chapters)*100:.1f}%\n\n")
                f.write("按类型统计:\n")
                type_counts = {}
                for chapter in chapters:
                    chapter_type = chapter['chapter_type']
                    type_counts[chapter_type] = type_counts.get(chapter_type, 0) + 1
                for chapter_type, count in type_counts.items():
                    f.write(f"  {chapter_type}: {count} 章\n")
                f.write(f"\n输出目录: {self.output_dir.absolute()}\n")
            logger.info(f"Summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")

def main():
    """Main function to run the crawler."""
    BASE_URL = "https://www.xuges.com/ls/mingshi/index.htm"
    OUTPUT_DIR = "ming_history_chapters_new"
    TEST_MODE = False
    MAX_WORKERS = 3
    logger.info("Starting Ming History Crawler")
    logger.info(f"Test Mode: {TEST_MODE}")
    try:
        crawler = MingHistoryCrawler(
            base_url=BASE_URL,
            output_dir=OUTPUT_DIR,
            test_mode=TEST_MODE
        )
        crawler.crawl_all_chapters(max_workers=MAX_WORKERS)
        logger.info("Ming History crawling completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    main() 