#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese Historical Text Processor for 白话资治通鉴
Process and split historical text into organized chapters and files
"""

import re
import os
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

class ChineseHistoricalTextProcessor:
    """
    Process Chinese historical texts and organize them into structured files
    """
    
    def __init__(self, input_file: str, output_dir: str = "organized_text"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.setup_logging()
        self.setup_directories()
        
        # Regex patterns for different types of headers/titles
        self.patterns = {
            # Main volume headers like "资 治 通 鉴 第 一 卷"
            'main_volume': re.compile(r'^资\s*治\s*通\s*鉴\s*第\s*[一二三四五六七八九十百千万]+\s*卷'),
            
            # Dynasty chronicles like "周 纪 一", "汉 纪 一"
            'dynasty_chronicle': re.compile(r'^[周秦汉魏晋宋齐梁陈隋唐五代宋辽金元明清]\s*纪\s*[一二三四五六七八九十百千万]+'),
            
            # Year headers like "周威烈王二十三年"
            'year_header': re.compile(r'^[周秦汉魏晋宋齐梁陈隋唐五代宋辽金元明清][^\s]*[王帝][^\s]*[年代纪元]+'),
            
            # Reign period markers
            'reign_period': re.compile(r'^[一二三四五六七八九十百千万]+年'),
            
            # Chapter markers (数字 + 章/篇/节)
            'chapter': re.compile(r'^第?[一二三四五六七八九十百千万]+[章篇节]'),
            
            # Special events or important dates
            'special_event': re.compile(r'^[春夏秋冬]?[正二三四五六七八九十十一十二]?月')
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('text_processing.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/chapters",
            f"{self.output_dir}/corpus",
            f"{self.output_dir}/metadata"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def is_title_line(self, line: str) -> Tuple[bool, str]:
        """
        Check if a line is a title/header and return the type
        Returns: (is_title, title_type)
        """
        line = line.strip()
        if not line:
            return False, ""
        
        for pattern_name, pattern in self.patterns.items():
            if pattern.match(line):
                return True, pattern_name
        
        # Additional heuristics for titles
        # Lines that are short and contain only Chinese characters (potential titles)
        if (len(line) < 50 and 
            re.match(r'^[^\w\s]*[\u4e00-\u9fff\s\u3000-\u303f\uff00-\uffef]+[^\w\s]*$', line) and
            not re.search(r'[。，！？；：]', line)):
            return True, "potential_title"
        
        return False, ""
    
    def create_filename(self, title: str, index: int, title_type: str) -> str:
        """Create a safe filename from title"""
        # Remove problematic characters for filename
        safe_title = re.sub(r'[<>:"/\\|?*\s]', '_', title)
        safe_title = re.sub(r'_+', '_', safe_title)  # Replace multiple underscores
        safe_title = safe_title.strip('_')
        
        # Limit filename length
        if len(safe_title) > 100:
            safe_title = safe_title[:100]
        
        return f"{index:04d}_{title_type}_{safe_title}.txt"
    
    def extract_metadata(self, title: str, content: str, title_type: str) -> Dict:
        """Extract metadata from title and content"""
        metadata = {
            'title': title,
            'title_type': title_type,
            'content_length': len(content),
            'word_count': len(content.replace(' ', '')),  # Chinese character count
            'line_count': len(content.split('\n')),
        }
        
        # Extract dynasty information
        dynasty_match = re.search(r'([周秦汉魏晋宋齐梁陈隋唐五代宋辽金元明清])', title)
        if dynasty_match:
            metadata['dynasty'] = dynasty_match.group(1)
        
        # Extract year information
        year_match = re.search(r'([一二三四五六七八九十百千万]+年)', title)
        if year_match:
            metadata['year_text'] = year_match.group(1)
        
        return metadata
    
    def process_text(self) -> List[Dict]:
        """
        Main processing function to split text into chapters
        Returns list of chapter information
        """
        self.logger.info(f"Starting to process file: {self.input_file}")
        
        chapters = []
        current_content = []
        current_title = ""
        current_title_type = ""
        chapter_index = 0
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            self.logger.info(f"Total lines to process: {len(lines)}")
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    if current_content:  # Add to content if we're in a chapter
                        current_content.append("")
                    continue
                
                is_title, title_type = self.is_title_line(line)
                
                if is_title:
                    # Save previous chapter if exists
                    if current_title and current_content:
                        content_text = '\n'.join(current_content).strip()
                        if content_text:  # Only save if there's actual content
                            chapter_info = self.save_chapter(
                                current_title, content_text, current_title_type, chapter_index
                            )
                            chapters.append(chapter_info)
                            chapter_index += 1
                    
                    # Start new chapter
                    current_title = line
                    current_title_type = title_type
                    current_content = []
                    self.logger.info(f"Found title at line {line_num}: {line}")
                
                else:
                    # Add to current content
                    current_content.append(line)
            
            # Save the last chapter
            if current_title and current_content:
                content_text = '\n'.join(current_content).strip()
                if content_text:
                    chapter_info = self.save_chapter(
                        current_title, content_text, current_title_type, chapter_index
                    )
                    chapters.append(chapter_info)
        
        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            raise
        
        self.logger.info(f"Processing complete. Total chapters extracted: {len(chapters)}")
        return chapters
    
    def save_chapter(self, title: str, content: str, title_type: str, index: int) -> Dict:
        """Save individual chapter to file and return metadata"""
        filename = self.create_filename(title, index, title_type)
        filepath = os.path.join(self.output_dir, "chapters", filename)
        
        # Clean content
        content = self.clean_text(content)
        
        # Save chapter file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(content)
        
        # Extract metadata
        metadata = self.extract_metadata(title, content, title_type)
        metadata['filename'] = filename
        metadata['filepath'] = filepath
        metadata['index'] = index
        
        self.logger.info(f"Saved chapter: {filename}")
        return metadata
    
    def create_corpus_files(self, chapters: List[Dict]):
        """Create corpus files for different purposes"""
        self.logger.info("Creating corpus files...")
        
        # Create combined corpus
        corpus_file = os.path.join(self.output_dir, "corpus", "complete_corpus.txt")
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for chapter in chapters:
                with open(chapter['filepath'], 'r', encoding='utf-8') as chapter_file:
                    f.write(chapter_file.read())
                    f.write('\n\n' + '='*50 + '\n\n')
        
        # Create dynasty-specific corpus
        dynasties = {}
        for chapter in chapters:
            dynasty = chapter.get('dynasty', 'unknown')
            if dynasty not in dynasties:
                dynasties[dynasty] = []
            dynasties[dynasty].append(chapter)
        
        for dynasty, dynasty_chapters in dynasties.items():
            dynasty_file = os.path.join(self.output_dir, "corpus", f"corpus_{dynasty}.txt")
            with open(dynasty_file, 'w', encoding='utf-8') as f:
                for chapter in dynasty_chapters:
                    with open(chapter['filepath'], 'r', encoding='utf-8') as chapter_file:
                        f.write(chapter_file.read())
                        f.write('\n\n' + '-'*30 + '\n\n')
    
    def save_metadata(self, chapters: List[Dict]):
        """Save metadata in various formats"""
        metadata_dir = os.path.join(self.output_dir, "metadata")
        
        # Save as JSON
        with open(os.path.join(metadata_dir, "chapters_metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(chapters, f, ensure_ascii=False, indent=2)
        
        # Save as CSV-like format
        with open(os.path.join(metadata_dir, "chapters_summary.txt"), 'w', encoding='utf-8') as f:
            f.write("Index\tTitle\tType\tDynasty\tContent_Length\tWord_Count\tFilename\n")
            for chapter in chapters:
                f.write(f"{chapter['index']}\t{chapter['title']}\t{chapter['title_type']}\t"
                       f"{chapter.get('dynasty', 'N/A')}\t{chapter['content_length']}\t"
                       f"{chapter['word_count']}\t{chapter['filename']}\n")
        
        # Create statistics
        stats = {
            'total_chapters': len(chapters),
            'title_types': {},
            'dynasties': {},
            'total_characters': sum(ch['word_count'] for ch in chapters)
        }
        
        for chapter in chapters:
            title_type = chapter['title_type']
            dynasty = chapter.get('dynasty', 'unknown')
            
            stats['title_types'][title_type] = stats['title_types'].get(title_type, 0) + 1
            stats['dynasties'][dynasty] = stats['dynasties'].get(dynasty, 0) + 1
        
        with open(os.path.join(metadata_dir, "processing_statistics.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Metadata saved. Statistics: {stats}")
    
    def run(self):
        """Run the complete processing pipeline"""
        self.logger.info("Starting Chinese Historical Text Processing Pipeline")
        
        try:
            # Process text and extract chapters
            chapters = self.process_text()
            
            if not chapters:
                self.logger.warning("No chapters were extracted!")
                return
            
            # Create corpus files
            self.create_corpus_files(chapters)
            
            # Save metadata
            self.save_metadata(chapters)
            
            self.logger.info("Processing pipeline completed successfully!")
            self.print_summary(chapters)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def print_summary(self, chapters: List[Dict]):
        """Print processing summary"""
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total chapters extracted: {len(chapters)}")
        print(f"Output directory: {self.output_dir}")
        
        # Count by type
        type_counts = {}
        for chapter in chapters:
            title_type = chapter['title_type']
            type_counts[title_type] = type_counts.get(title_type, 0) + 1
        
        print("\nChapters by type:")
        for title_type, count in sorted(type_counts.items()):
            print(f"  {title_type}: {count}")
        
        print(f"\nFiles created:")
        print(f"  Chapters: {len(chapters)} files in {self.output_dir}/chapters/")
        print(f"  Corpus: Multiple files in {self.output_dir}/corpus/")
        print(f"  Metadata: Multiple files in {self.output_dir}/metadata/")


def main():
    """Main function to run the processor"""
    input_file = r"白话资治通鉴01—周威烈王二十三年.至.汉惠帝七_djvu.txt"
    output_dir = "organized_historical_text"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    print("Chinese Historical Text Processor")
    print("="*40)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create processor and run
    processor = ChineseHistoricalTextProcessor(input_file, output_dir)
    processor.run()


if __name__ == "__main__":
    main()