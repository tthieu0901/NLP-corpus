#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data inspection script for Ming History corpus
"""

import os
import re
from pathlib import Path
from collections import Counter

def analyze_data_folder(folder_path="ming_history_chapters"):
    """Analyze the structure and content of the data folder"""
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder {folder_path} not found!")
        return
    
    print("="*60)
    print(f"MING HISTORY DATA ANALYSIS")
    print("="*60)
    
    # Get all text files
    txt_files = [f for f in folder.glob("*.txt") if f.name != "crawl_summary.txt"]
    print(f"Total text files: {len(txt_files)}")
    
    # Analyze file types
    file_types = Counter()
    total_chars = 0
    total_files_processed = 0
    sample_texts = []
    
    for file_path in txt_files[:20]:  # Analyze first 20 files for speed
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata
            lines = content.split('\n')
            chapter_type = ""
            title = ""
            
            for line in lines[:10]:  # Check first 10 lines for metadata
                if line.startswith("章节类型:"):
                    chapter_type = line.split(":", 1)[1].strip()
                elif line.startswith("标题:"):
                    title = line.split(":", 1)[1].strip()
            
            file_types[chapter_type] += 1
            
            # Extract main text content
            parts = content.split('=' * 50)
            if len(parts) >= 2:
                text_content = parts[1].strip()
                total_chars += len(text_content)
                total_files_processed += 1
                
                if len(sample_texts) < 3 and text_content:
                    sample_texts.append({
                        'title': title,
                        'type': chapter_type,
                        'content': text_content[:200] + "..." if len(text_content) > 200 else text_content
                    })
        
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    print(f"\nFiles analyzed: {total_files_processed}")
    print(f"Average characters per file: {total_chars // max(total_files_processed, 1):,}")
    print(f"Total characters (sample): {total_chars:,}")
    
    print(f"\nChapter types found:")
    for chapter_type, count in file_types.most_common():
        print(f"  {chapter_type}: {count} files")
    
    print(f"\nSample content:")
    print("-" * 40)
    for i, sample in enumerate(sample_texts, 1):
        print(f"\n{i}. {sample['title']} ({sample['type']})")
        print(f"   {sample['content']}")
    
    # Estimate training data size
    estimated_total_chars = (total_chars // total_files_processed) * len(txt_files)
    print(f"\nEstimated total training data: {estimated_total_chars:,} characters")
    print(f"Estimated tokens (approx): {estimated_total_chars // 2:,}")  # Rough estimate for Chinese

def inspect_specific_file(file_path):
    """Inspect a specific file in detail"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\nDetailed analysis of: {file_path}")
        print("-" * 50)
        
        # Extract metadata
        lines = content.split('\n')
        print("Metadata:")
        for line in lines[:10]:
            if ':' in line and any(keyword in line for keyword in ['标题', '章节类型', '卷数', 'URL']):
                print(f"  {line}")
        
        # Extract text content
        parts = content.split('=' * 50)
        if len(parts) >= 2:
            text_content = parts[1].strip()
            print(f"\nText content length: {len(text_content)} characters")
            print(f"First 300 characters:")
            print(f"  {text_content[:300]}...")
            
            # Count common classical Chinese characters
            common_chars = Counter(text_content)
            print(f"\nMost common characters:")
            for char, count in common_chars.most_common(10):
                if char not in [' ', '\n', '\t']:
                    print(f"  {char}: {count}")
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main inspection function"""
    
    # Analyze the entire folder
    analyze_data_folder()
    
    # Inspect a few specific files
    folder = Path("ming_history_chapters")
    if folder.exists():
        txt_files = [f for f in folder.glob("*.txt") if f.name != "crawl_summary.txt"]
        
        if txt_files:
            print("\n" + "="*60)
            print("DETAILED FILE INSPECTION")
            print("="*60)
            
            # Inspect first 2 files
            for file_path in txt_files[:2]:
                inspect_specific_file(file_path)

if __name__ == "__main__":
    main()
