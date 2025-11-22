"""
HUMAN-LIKE WEB READER
=====================
Enables the Cosmic Synapse models to "read" the web like a human.
Fetches, cleans, and streams text from URLs for continuous learning.
"""

import urllib.request
import urllib.error
import re
from html.parser import HTMLParser
from typing import List, Generator, Tuple

class TextExtractor(HTMLParser):
    """
    Simple HTML parser to extract human-readable text.
    Ignores scripts, styles, and metadata.
    """
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.ignore_tags = {'script', 'style', 'head', 'title', 'meta', '[document]'}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.ignore_tags:
            # Clean whitespace
            text = data.strip()
            if len(text) > 0:
                self.text_parts.append(text)

    def get_text(self) -> str:
        return '\n'.join(self.text_parts)

class WebCurriculum:
    """
    Manages a curriculum of URLs for the model to read.
    """
    def __init__(self, urls: List[str] = None):
        self.urls = urls or []
        self.history = []

    def add_url(self, url: str):
        self.urls.append(url)

    def read_stream(self) -> Generator[Tuple[str, str], None, None]:
        """
        Yields (source_url, content) tuples.
        """
        for url in self.urls:
            print(f"[WEB READER] Reading: {url}")
            try:
                # Mimic a browser user agent
                req = urllib.request.Request(
                    url, 
                    data=None, 
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                )
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    html_content = response.read().decode('utf-8', errors='ignore')
                
                # Extract text
                parser = TextExtractor()
                parser.feed(html_content)
                text = parser.get_text()
                
                # Post-processing cleanup
                # Remove excessive newlines
                text = re.sub(r'\n\s*\n', '\n\n', text)
                
                if len(text) < 100:
                    print(f"[WEB READER] Skipped {url} (content too short)")
                    continue
                    
                self.history.append(url)
                yield url, text
                
            except Exception as e:
                print(f"[WEB READER] Failed to read {url}: {e}")
                continue

# Example usage
if __name__ == "__main__":
    reader = WebCurriculum([
        "https://www.gutenberg.org/cache/epub/1513/pg1513.txt", # Romeo and Juliet
        "https://example.com"
    ])
    
    for src, content in reader.read_stream():
        print(f"\n--- Source: {src} ---")
        print(content[:500] + "...")
