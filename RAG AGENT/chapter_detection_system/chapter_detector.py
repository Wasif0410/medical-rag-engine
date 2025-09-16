"""
Chapter Detection System - Production Version
============================================

Simple, efficient chapter detection for PDF documents.
Uses multiple detection methods with automatic fallbacks.

Usage:
    from chapter_detector import detect_chapters
    
    chapters = detect_chapters("document.pdf")
    for chapter in chapters:
        print(f"{chapter.title}: pages {chapter.start_page}-{chapter.end_page}")
"""

import fitz  # PyMuPDF
import re
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class Chapter:
    """Chapter information"""
    title: str
    start_page: int
    end_page: int
    confidence: float = 0.0
    method: str = ""

class ChapterDetector:
    """Main chapter detection class"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.doc = None
        
        # Chapter patterns
        self.patterns = [
            r'^chapter\s+\d+\b',
            r'^appendix\s+([a-z]|[ivxlcdm]+)\b',
            r'^\d+\.\s+\S',
            r'^section\s+\d+\b'
        ]
        
        # ToC keywords
        self.toc_keywords = [
            'contents', 'table of contents', 'toc', 'index',
            'inhalt', 'sommaire', 'indice', 'índice'
        ]

    def __enter__(self):
        self.doc = fitz.open(str(self.pdf_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.doc:
            self.doc.close()

    def detect(self) -> List[Chapter]:
        """Main detection method with fallbacks"""
        
        # Method 1: Extract bookmarks (highest accuracy)
        chapters = self._extract_bookmarks()
        if chapters:
            logger.info(f"Found {len(chapters)} chapters from bookmarks")
            return self._finalize_chapters(chapters)
        
        # Method 2: Parse Table of Contents
        chapters = self._parse_toc()
        if chapters:
            logger.info(f"Found {len(chapters)} chapters from ToC")
            return self._finalize_chapters(chapters)
        
        # Method 3: Heading detection
        chapters = self._detect_headings()
        if chapters:
            logger.info(f"Found {len(chapters)} chapters from headings")
            return self._finalize_chapters(chapters)
        
        logger.warning("No chapters detected")
        return []

    def _extract_bookmarks(self) -> List[Chapter]:
        """Extract chapters from PDF bookmarks"""
        try:
            toc = self.doc.get_toc(simple=False)
            if not toc:
                return []
            
            chapters = []
            for entry in toc:
                level, title, page_num = entry[:3]
                
                # Include level 1 entries (main chapters) or entries that match chapter patterns
                # Also include level 2 entries that are numbered subsections
                include_entry = (
                    level == 1 or  # Main chapters
                    self._matches_pattern(title) or  # Pattern matches (like "Chapter X", "Appendix X") 
                    (level == 2 and re.match(r'^\d+\.\d+\s+', title.strip()))  # Numbered subsections like "9.1", "2.7"
                )
                
                if include_entry:
                    chapters.append(Chapter(
                        title=title.strip(),
                        start_page=max(1, page_num),  # Keep 1-based (PyMuPDF bookmarks are already 1-based)
                        end_page=0,
                        confidence=0.9,
                        method="bookmarks"
                    ))
            
            return chapters
            
        except Exception as e:
            logger.error(f"Bookmark extraction failed: {e}")
            return []

    def _parse_toc(self) -> List[Chapter]:
        """Parse printed Table of Contents"""
        try:
            # Find ToC pages
            toc_pages = []
            for page_num in range(min(20, len(self.doc))):
                page = self.doc[page_num]
                text = page.get_text().lower()
                
                # Check for ToC keywords and many page numbers
                has_keywords = any(keyword in text for keyword in self.toc_keywords)
                page_number_lines = len(re.findall(r'\s+\d+\s*$', text, re.MULTILINE))
                
                if has_keywords and page_number_lines >= 5:
                    toc_pages.append(page_num)
            
            # Parse ToC entries
            chapters = []
            for page_num in toc_pages:
                page = self.doc[page_num]
                text = page.get_text()
                
                # Pattern: Title ... Page Number
                pattern = r'^(?P<title>.+?)(\.{2,}|\s{3,})\s*(?P<page>\d+)\s*$'
                
                for line in text.split('\n'):
                    match = re.match(pattern, line.strip())
                    if match and len(match.group('title')) >= 3:
                        try:
                            start_page = int(match.group('page')) - 1  # Convert to 0-based
                            if 0 <= start_page < len(self.doc):
                                chapters.append(Chapter(
                                    title=match.group('title').strip(),
                                    start_page=start_page,
                                    end_page=0,
                                    confidence=0.8,
                                    method="toc"
                                ))
                        except ValueError:
                            continue
            
            return chapters
            
        except Exception as e:
            logger.error(f"ToC parsing failed: {e}")
            return []

    def _detect_headings(self) -> List[Chapter]:
        """Detect chapters using font size analysis"""
        try:
            # Analyze font sizes
            font_sizes = []
            max_pages = min(50, len(self.doc))
            
            for page_num in range(max_pages):
                page = self.doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                size = span.get("size", 12.0)
                                if 8.0 <= size <= 72.0:
                                    font_sizes.append(size)
            
            if not font_sizes:
                return []
            
            # Calculate thresholds
            body_font = statistics.median(font_sizes)
            threshold = max(
                statistics.quantiles(font_sizes, n=4)[2],  # 75th percentile
                body_font + 2.0,
                14.0
            )
            
            # Find headings
            chapters = []
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                page_height = page.rect.height
                top_limit = page_height * 0.25  # Top 25% of page
                
                blocks = page.get_text("dict")["blocks"]
                best_candidate = None
                best_font_size = 0
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                        
                    for line in block["lines"]:
                        line_text = ""
                        max_font = 0
                        y_pos = None
                        
                        for span in line["spans"]:
                            line_text += span.get("text", "")
                            font_size = span.get("size", 12.0)
                            max_font = max(max_font, font_size)
                            if y_pos is None:
                                bbox = span.get("bbox", [0, 0, 0, 0])
                                y_pos = bbox[1]
                        
                        line_text = line_text.strip()
                        
                        # Check if it's a heading
                        is_heading = (
                            max_font >= threshold or 
                            self._matches_pattern(line_text)
                        )
                        
                        if (is_heading and y_pos and y_pos <= top_limit and 
                            len(line_text) >= 3 and max_font > best_font_size):
                            best_candidate = Chapter(
                                title=line_text,
                                start_page=page_num,
                                end_page=0,
                                confidence=0.7,
                                method="headings"
                            )
                            best_font_size = max_font
                
                if best_candidate:
                    chapters.append(best_candidate)
            
            # Filter to top-level chapters
            filtered = []
            for chapter in chapters:
                title = chapter.title.lower()
                # Keep chapters with keywords or single numbers
                if (any(keyword in title for keyword in ['chapter', 'appendix', 'part']) or
                    re.match(r'^\d+\.?\s+', title) and not re.match(r'^\d+\.\d+', title)):
                    filtered.append(chapter)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Heading detection failed: {e}")
            return []

    def _matches_pattern(self, text: str) -> bool:
        """Check if text matches chapter patterns"""
        text_lower = text.lower().strip()
        return any(re.match(pattern, text_lower) for pattern in self.patterns)

    def _finalize_chapters(self, chapters: List[Chapter]) -> List[Chapter]:
        """Compute end pages and validate"""
        if not chapters:
            return []
        
        # Sort by start page
        chapters.sort(key=lambda c: c.start_page)
        
        # Remove duplicates and invalid entries
        validated = []
        for chapter in chapters:
            if (chapter.start_page >= 0 and 
                chapter.start_page < len(self.doc) and
                (not validated or validated[-1].start_page != chapter.start_page)):
                validated.append(chapter)
        
        # Compute end pages
        for i, chapter in enumerate(validated):
            if i < len(validated) - 1:
                chapter.end_page = validated[i + 1].start_page - 1
            else:
                chapter.end_page = len(self.doc) - 1
            
            chapter.end_page = max(chapter.end_page, chapter.start_page)
        
        return validated

def detect_chapters(pdf_path: str) -> List[Chapter]:
    """
    Simple function to detect chapters in a PDF
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of Chapter objects with title, start_page, end_page
    """
    with ChapterDetector(pdf_path) as detector:
        return detector.detect()

def save_chapters(chapters: List[Chapter], output_path: str, format: str = 'json'):
    """
    Save chapters to file
    
    Args:
        chapters: List of Chapter objects
        output_path: Output file path
        format: 'json' or 'csv'
    """
    import json
    import csv
    
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        data = [
            {
                'title': ch.title,
                'start_page': ch.start_page,  # Already 1-based
                'end_page': ch.end_page,
                'pages': ch.end_page - ch.start_page + 1,
                'confidence': ch.confidence,
                'method': ch.method
            }
            for ch in chapters
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    elif format == 'csv':
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['title', 'start_page', 'end_page', 'pages', 'confidence', 'method'])
            
            for ch in chapters:
                writer.writerow([
                    ch.title,
                    ch.start_page,  # Already 1-based
                    ch.end_page,
                    ch.end_page - ch.start_page + 1,
                    ch.confidence,
                    ch.method
                ])

# Example usage
if __name__ == "__main__":
    # Simple example
    chapters = detect_chapters("document.pdf")
    
    for i, chapter in enumerate(chapters, 1):
        print(f"{i}. {chapter.title}")
        print(f"   Pages: {chapter.start_page}-{chapter.end_page}")  # Already 1-based
        print(f"   Method: {chapter.method}, Confidence: {chapter.confidence:.2f}")
        print()
    
    # Save results to organized output folder
    if chapters:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        save_chapters(chapters, output_dir / "chapters.json", "json")
        save_chapters(chapters, output_dir / "chapters.csv", "csv")
        print(f"💾 Saved {len(chapters)} chapters to output/ folder")
