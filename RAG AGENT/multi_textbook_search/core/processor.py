"""
Production Textbook Processor
============================

Handles processing of multiple textbooks with chapter detection integration.
"""

import os
import pickle
import time
from typing import List, Dict, Any, Optional
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
import csv

from .models import TextbookInfo, ProcessingResult, TextbookCollection

class ProductionTextbookProcessor:
    """Production-ready textbook processor for multiple textbooks"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_chapter_data(self, chapter_csv_path: str) -> List[Dict]:
        """Load ALL chapter data from CSV file - completely general"""
        print(f"📖 Loading chapter data from: {chapter_csv_path}")
        
        chapters = []
        
        try:
            with open(chapter_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    chapter = {
                        'title': row['title'],
                        'start_page': int(row['start_page']),
                        'end_page': int(row['end_page']),
                        'pages': int(row['pages']),
                        'confidence': float(row['confidence']),
                        'method': row['method']
                    }
                    chapters.append(chapter)
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return []
        
        print(f"✅ Loaded {len(chapters)} entries from CSV")
        return chapters
    
    def process_textbook(self, 
                        textbook_info: TextbookInfo,
                        pdf_path: str, 
                        chapter_csv_path: str) -> ProcessingResult:
        """Process a single textbook with chapter detection"""
        
        print(f"\n📚 Processing: {textbook_info.display_name}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load chapter data
            chapters = self.load_chapter_data(chapter_csv_path)
            if not chapters:
                return ProcessingResult(
                    textbook_info=textbook_info,
                    documents=[],
                    processing_time=0,
                    success=False,
                    error_message="No chapter data found"
                )
            
            # Extract text with chapter awareness
            documents = self._extract_text_from_pdf_with_chapters(
                pdf_path, chapters, textbook_info
            )
            
            processing_time = time.time() - start_time
            
            # Update textbook info
            textbook_info.chapters_detected = len(chapters)
            textbook_info.total_chunks = len(documents)
            
            print(f"✅ Processing completed in {processing_time:.2f}s")
            print(f"📊 Created {len(documents)} chunks from {len(chapters)} chapters")
            
            return ProcessingResult(
                textbook_info=textbook_info,
                documents=documents,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Error processing textbook: {e}")
            
            return ProcessingResult(
                textbook_info=textbook_info,
                documents=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _extract_text_from_pdf_with_chapters(self, 
                                           pdf_path: str, 
                                           chapters: List[Dict], 
                                           textbook_info: TextbookInfo) -> List[Document]:
        """Extract text from PDF using precise chapter boundaries"""
        print(f"📖 Processing PDF: {os.path.basename(pdf_path)}")
        
        documents = []
        
        # Use PyPDF2 for text extraction
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"📄 Total pages in PDF: {total_pages}")
            
            for chapter in chapters:
                chapter_title = chapter['title']
                start_page = chapter['start_page']
                end_page = chapter['end_page']
                
                print(f"   📚 Processing {chapter_title} (pages {start_page}-{end_page})")
                
                # Extract text for this chapter
                chapter_text = ""
                for page_num in range(start_page - 1, min(end_page, total_pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        # DEBUG: Print raw extracted text for the first page of each chapter
                        if page_num == start_page - 1:
                            print(f"\n--- RAW EXTRACTED TEXT (Page {page_num + 1}) ---\n{page_text}\n--- END RAW TEXT ---\n")
                        if page_text and page_text.strip():
                            chapter_text += f"\n\nPage {page_num + 1}:\n{page_text}"
                    except Exception as e:
                        print(f"      ⚠️ Error extracting page {page_num + 1}: {e}")
                        continue
                
                if chapter_text.strip():
                    # Create chunks for this chapter
                    chapter_chunks = self._create_chapter_chunks(
                        chapter_text, 
                        chapter_title, 
                        start_page, 
                        end_page,
                        textbook_info
                    )
                    documents.extend(chapter_chunks)
                    print(f"      ✅ Created {len(chapter_chunks)} chunks")
                else:
                    print(f"      ⚠️ No text extracted for {chapter_title}")
        
        return documents
    
    def _create_chapter_chunks(self, 
                              chapter_text: str, 
                              chapter_title: str, 
                              start_page: int, 
                              end_page: int,
                              textbook_info: TextbookInfo) -> List[Document]:
        """Create chunks for a specific chapter with accurate metadata"""
        chunks = self.text_splitter.split_text(chapter_text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            # Extract page number from chunk if possible
            page_match = re.search(r'Page (\d+):', chunk)
            if page_match:
                page_num = int(page_match.group(1))
            else:
                # Estimate page based on chunk position
                pages_in_chapter = end_page - start_page + 1
                estimated_page = start_page + int((i / len(chunks)) * pages_in_chapter)
                page_num = min(estimated_page, end_page)
            
            # Clean the chunk text (remove page markers)
            clean_chunk = re.sub(r'Page \d+:\n', '', chunk).strip()
            
            if clean_chunk and len(clean_chunk) > 50:  # Only keep substantial chunks
                doc = Document(
                    page_content=clean_chunk,
                    metadata={
                        "source": textbook_info.file_path,
                        "textbook_id": textbook_info.id,
                        "textbook": textbook_info.display_name,
                        "chapter": chapter_title,
                        "chapter_start_page": start_page,
                        "chapter_end_page": end_page,
                        "page": page_num,
                        "chunk_id": f"{textbook_info.id}_{chapter_title.split(' - ')[0] if ' - ' in chapter_title else chapter_title}_{i+1}",
                        "chunk_index": i
                    }
                )
                documents.append(doc)
        
        return documents

class ProductionPipeline:
    """Complete production pipeline for multiple textbooks"""
    
    def __init__(self, data_dir: str = "data"):
        self.processor = ProductionTextbookProcessor()
        self.data_dir = data_dir
        self.collection = TextbookCollection()
    
    def process_multiple_textbooks(self, textbook_configs: List[Dict[str, str]]) -> TextbookCollection:
        """Process multiple textbooks in batch"""
        
        print("🏭 PRODUCTION TEXTBOOK PROCESSING PIPELINE")
        print("=" * 60)
        
        for i, config in enumerate(textbook_configs, 1):
            print(f"\n📚 TEXTBOOK {i}/{len(textbook_configs)}")
            
            # Create textbook info
            textbook_info = TextbookInfo(
                id=config.get('id', f"textbook_{i}"),
                title=config['title'],
                edition=config.get('edition'),
                year=config.get('year'),
                file_path=config['pdf_path']
            )
            
            # Process textbook
            result = self.processor.process_textbook(
                textbook_info=textbook_info,
                pdf_path=config['pdf_path'],
                chapter_csv_path=config['chapter_csv_path']
            )
            
            # Add to collection
            self.collection.add_textbook(result)
            
            if result.success:
                print(f"✅ Successfully processed {textbook_info.display_name}")
            else:
                print(f"❌ Failed to process {textbook_info.display_name}: {result.error_message}")
        
        # Save collection
        collection_path = os.path.join(self.data_dir, "multi_textbook_processed_documents.pkl")
        self._save_collection(collection_path)
        
        # Show final statistics
        self._show_final_stats()
        
        return self.collection
    
    def _save_collection(self, output_path: str):
        """Save the complete textbook collection"""
        print(f"\n💾 Saving collection to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.collection, f)
        
        print(f"✅ Collection saved with {len(self.collection.all_documents)} total documents")
    
    def _show_final_stats(self):
        """Show final processing statistics"""
        stats = self.collection.get_stats()
        
        print(f"\n📊 FINAL PROCESSING STATISTICS")
        print("=" * 60)
        print(f"📚 Total textbooks processed: {stats['total_textbooks']}")
        print(f"📄 Total documents created: {stats['total_documents']}")
        
        print(f"\n📖 Processed textbooks:")
        for textbook in stats['textbooks']:
            print(f"   ✅ {textbook}")
