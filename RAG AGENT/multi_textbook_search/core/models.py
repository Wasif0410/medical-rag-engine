"""
Production Models for Multi-Textbook Processing System
=====================================================
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from langchain.schema import Document

@dataclass
class TextbookInfo:
    """Metadata for a textbook"""
    id: str
    title: str
    edition: Optional[str] = None
    year: Optional[str] = None
    authors: Optional[List[str]] = None
    file_path: str = ""
    chapters_detected: int = 0
    total_chunks: int = 0
    
    @property
    def display_name(self) -> str:
        """Display name for the textbook"""
        name = self.title
        if self.edition:
            name += f" ({self.edition})"
        if self.year:
            name += f" {self.year}"
        return name

@dataclass
class ProcessingResult:
    """Result of textbook processing"""
    textbook_info: TextbookInfo
    documents: List[Document]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class TextbookCollection:
    """Collection of processed textbooks"""
    
    def __init__(self):
        self.textbooks: Dict[str, TextbookInfo] = {}
        self.all_documents: List[Document] = []
    
    def add_textbook(self, result: ProcessingResult):
        """Add a processed textbook to the collection"""
        if result.success:
            self.textbooks[result.textbook_info.id] = result.textbook_info
            self.all_documents.extend(result.documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'total_textbooks': len(self.textbooks),
            'total_documents': len(self.all_documents),
            'textbooks': [info.display_name for info in self.textbooks.values()]
        }