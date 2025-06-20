#!/usr/bin/env python3
"""
Example usage of the Ming History Crawler

This script demonstrates how to use the MingHistoryCrawler class
with different configurations.
"""

from crawl_data import MingHistoryCrawler
import logging

def example_test_run():
    """Example: Run crawler in test mode (first 5 chapters only)"""
    print("=== Test Mode Example ===")
    
    crawler = MingHistoryCrawler(
        base_url="https://www.xuges.com/ls/mingshi/index.htm",
        output_dir="test_output",
        test_mode=True  # Only crawl first 5 links
    )
    
    crawler.crawl_all_chapters(max_workers=2)

def example_full_run():
    """Example: Run crawler for all chapters"""
    print("=== Full Crawl Example ===")
    
    crawler = MingHistoryCrawler(
        base_url="https://www.xuges.com/ls/mingshi/index.htm",
        output_dir="ming_history_complete",
        test_mode=False  # Crawl all chapters
    )
    
    crawler.crawl_all_chapters(max_workers=5)

def example_custom_run():
    """Example: Custom configuration"""
    print("=== Custom Configuration Example ===")
    
    # Configure logging level
    logging.getLogger().setLevel(logging.DEBUG)
    
    crawler = MingHistoryCrawler(
        base_url="https://www.xuges.com/ls/mingshi/index.htm",
        output_dir="custom_output",
        test_mode=False
    )
    
    # Get chapter links first (without downloading)
    chapters = crawler.extract_chapter_links()
    print(f"Found {len(chapters)} chapters")
    
    # Print some statistics
    chapter_types = {}
    for chapter in chapters:
        chapter_type = chapter['chapter_type']
        chapter_types[chapter_type] = chapter_types.get(chapter_type, 0) + 1
    
    print("\nChapter types found:")
    for chapter_type, count in chapter_types.items():
        print(f"  {chapter_type}: {count} chapters")
    
    # Crawl specific chapters (e.g., only 本纪)
    benji_chapters = [ch for ch in chapters if ch['chapter_type'] == '本纪']
    print(f"\nCrawling {len(benji_chapters)} 本纪 chapters...")
    
    for chapter in benji_chapters:
        success = crawler.crawl_chapter(chapter)
        if success:
            print(f"✓ Successfully crawled: {chapter['title']}")
        else:
            print(f"✗ Failed to crawl: {chapter['title']}")

if __name__ == "__main__":
    # Uncomment the example you want to run
    
    # Run test mode (recommended for first try)
    example_test_run()
    
    # Run full crawl (uncomment when ready)
    # example_full_run()
    
    # Run custom configuration (advanced usage)
    # example_custom_run()
