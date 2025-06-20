# Ming History Crawler (明史爬虫)

A Python script to crawl chapters from the Ming History (明史) website at https://www.xuges.com/ls/mingshi/index.htm.

## Features

- ✅ **Chapter Type Detection**: Automatically detects chapter types (本纪, 列传, 志, 表, etc.)
- ✅ **Volume Numbering**: Incremental volume numbering per chapter section
- ✅ **Content Extraction**: Extracts true titles and full text content
- ✅ **File Organization**: Saves files with format `章节类型_卷数_标题.txt`
- ✅ **Encoding Support**: Handles GBK encoding properly
- ✅ **Safe Filenames**: Cleans unsafe characters from filenames
- ✅ **Test Mode**: Option to crawl only first 5 links for testing
- ✅ **Multithreading**: Uses ThreadPoolExecutor for performance
- ✅ **Comprehensive Logging**: Detailed logs and error handling
- ✅ **Progress Summary**: Generates summary report after completion

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Test Mode)

Run the script in test mode to crawl only the first 5 chapters:

```bash
python crawl_data.py
```

### Full Crawl

To crawl all chapters, modify the `TEST_MODE` variable in `crawl_data.py`:

```python
TEST_MODE = False  # Change to False for full crawl
```

Then run:
```bash
python crawl_data.py
```

### Advanced Usage

Use the example script for more control:

```bash
python example_usage.py
```

### Configuration Options

In `crawl_data.py`, you can modify:

- `BASE_URL`: The main index page URL
- `OUTPUT_DIR`: Directory to save chapters
- `TEST_MODE`: Enable/disable test mode (first 5 chapters only)
- `MAX_WORKERS`: Number of concurrent threads (default: 3)

## Output Structure

The script creates the following files:

```
ming_history_chapters/
├── crawl_summary.txt                    # Summary of crawling process
├── 本纪_1_太祖朱元璋纪(1).txt              # Chapter files
├── 本纪_2_太祖朱元璋纪(2).txt
├── 列传_1_徐达列传.txt
└── ...
```

Each chapter file contains:
- 标题 (Title)
- 章节类型 (Chapter Type)
- 卷数 (Volume Number)
- URL (Source URL)
- 原始标题 (Original Title)
- Full content

## Class Documentation

### MingHistoryCrawler

Main crawler class with the following methods:

- `__init__(base_url, output_dir, test_mode)`: Initialize crawler
- `extract_chapter_links()`: Extract all chapter links from index page
- `extract_chapter_content(chapter_info)`: Extract content from a chapter page
- `crawl_all_chapters(max_workers)`: Crawl all chapters with multithreading
- `clean_filename(filename)`: Clean unsafe characters from filenames

## Example Output

Successfully crawled file example:

```
标题: 太祖朱元璋纪(1)
章节类型: 本纪
卷数: 1
URL: https://www.xuges.com/ls/mingshi/0001.htm
原始标题: 1.太祖朱元璋纪

==================================================

太祖开天行道肇纪立极大圣至神仁文义武俊德成功高皇帝，讳元璋...
[Full chapter content continues...]
```

## Configuration

### Threading Configuration

Adjust the number of concurrent workers based on your system:

```python
MAX_WORKERS = 3  # Conservative setting
MAX_WORKERS = 5  # Faster but more resource intensive
```

### Test Mode

For testing and development:

```python
TEST_MODE = True   # Only crawl first 5 chapters
TEST_MODE = False  # Crawl all chapters (full run)
```

## Error Handling

The script includes comprehensive error handling:

- Network timeouts and connection errors
- Encoding issues (handles GBK properly)
- File system errors
- Missing content elements
- Invalid URLs

All errors are logged to `ming_history_crawl.log`.

## Logging

The script creates detailed logs in `ming_history_crawl.log` including:

- Progress updates
- Successful downloads
- Error messages
- Performance statistics

## Performance

- Uses ThreadPoolExecutor for concurrent downloads
- Includes small delays to be respectful to the server
- Handles encoding efficiently
- Memory-efficient content extraction

## Requirements

- Python 3.7+
- requests>=2.28.0
- beautifulsoup4>=4.11.0
- lxml>=4.9.0

## Notes

- The script respects the server by including delays between requests
- Uses proper User-Agent headers
- Handles network failures gracefully
- Creates backup of content with metadata
- Supports resume capability through file checking

## Troubleshooting

1. **Import Errors**: Make sure all dependencies are installed via `pip install -r requirements.txt`
2. **Network Errors**: Check internet connection and try reducing `MAX_WORKERS`
3. **Encoding Issues**: The script handles GBK encoding automatically
4. **File Permission Errors**: Ensure write permissions in the output directory

## License

This script is for educational and research purposes. Please respect the website's terms of service and robots.txt.
