"""
Optimized Single Textbook Processing Pipeline
=============================================

Fast, scalable approach with separate files per textbook.
"""

import os
import sys
import pickle
import numpy as np
from typing import Dict, List
from core.processor import ProductionTextbookProcessor
from core.search_engine import ProductionSearchEngine
from core.models import TextbookInfo, TextbookCollection

class OptimizedTextbookManager:
    """Manages textbooks with separate files for optimal performance"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.textbooks_dir = os.path.join(data_dir, "textbooks")
        self.registry_path = os.path.join(data_dir, "textbook_registry.pkl")
        self.vectors_path = os.path.join(data_dir, "optimized_search_vectors.pkl")
        
        # Create directories
        os.makedirs(self.textbooks_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
    
    def load_registry(self) -> Dict:
        """Load textbook registry (metadata only)"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'rb') as f:
                return pickle.load(f)
        return {"textbooks": {}, "total_chunks": 0, "version": 1}
    
    def save_registry(self, registry: Dict):
        """Save textbook registry"""
        with open(self.registry_path, 'wb') as f:
            pickle.dump(registry, f)
    
    def save_textbook_chunks(self, textbook_id: str, documents: List, textbook_info: TextbookInfo):
        """Save chunks for a single textbook"""
        textbook_file = os.path.join(self.textbooks_dir, f"{textbook_id}_chunks.pkl")
        
        textbook_data = {
            "textbook_info": textbook_info,
            "documents": documents,
            "chunk_count": len(documents)
        }
        
        with open(textbook_file, 'wb') as f:
            pickle.dump(textbook_data, f)
        
        print(f"💾 Saved {len(documents)} chunks to {textbook_id}_chunks.pkl")
    
    def load_textbook_chunks(self, textbook_id: str):
        """Load chunks for a specific textbook"""
        textbook_file = os.path.join(self.textbooks_dir, f"{textbook_id}_chunks.pkl")
        
        if os.path.exists(textbook_file):
            with open(textbook_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_all_documents(self) -> List:
        """Load all documents from all textbook files (only when needed)"""
        registry = self.load_registry()
        all_documents = []
        
        for textbook_id in registry["textbooks"].keys():
            textbook_data = self.load_textbook_chunks(textbook_id)
            if textbook_data:
                all_documents.extend(textbook_data["documents"])
        
        return all_documents
    
    def get_collection_for_search(self) -> TextbookCollection:
        """Create a TextbookCollection for search (loads all docs)"""
        collection = TextbookCollection()
        registry = self.load_registry()
        
        for textbook_id in registry["textbooks"].keys():
            textbook_data = self.load_textbook_chunks(textbook_id)
            if textbook_data:
                # Create a fake ProcessingResult for the collection
                from core.models import ProcessingResult
                result = ProcessingResult(
                    textbook_info=textbook_data["textbook_info"],
                    documents=textbook_data["documents"],
                    processing_time=0,
                    success=True
                )
                collection.add_textbook(result)
        
        return collection

def process_single_textbook_optimized(textbook_config):
    """Optimized single textbook processing"""
    
    print("🚀 OPTIMIZED SINGLE TEXTBOOK PROCESSING")
    print("=" * 60)
    
    manager = OptimizedTextbookManager()
    
    # Create textbook info
    textbook_info = TextbookInfo(
        id=textbook_config['id'],
        title=textbook_config['title'],
        edition=textbook_config.get('edition'),
        year=textbook_config.get('year'),
        file_path=textbook_config['pdf_path']
    )
    
    print(f"🔄 Processing: {textbook_info.display_name}")
    
    # Check if already exists
    registry = manager.load_registry()
    if textbook_info.id in registry["textbooks"]:
        print(f"⚠️ Textbook '{textbook_info.id}' already exists. Replacing...")
    
    # Process the textbook
    processor = ProductionTextbookProcessor()
    result = processor.process_textbook(
        textbook_info=textbook_info,
        pdf_path=textbook_config['pdf_path'],
        chapter_csv_path=textbook_config['chapter_csv_path']
    )
    
    if not result.success:
        print(f"❌ Processing failed: {result.error_message}")
        return False
    
    print(f"✅ Successfully processed {len(result.documents)} document chunks")
    
    # Save textbook chunks to separate file
    manager.save_textbook_chunks(textbook_info.id, result.documents, textbook_info)
    
    # Update registry
    registry["textbooks"][textbook_info.id] = {
        "title": textbook_info.display_name,
        "chunk_count": len(result.documents),
        "file_path": f"{textbook_info.id}_chunks.pkl"
    }
    
    # Calculate total chunks
    registry["total_chunks"] = sum(tb["chunk_count"] for tb in registry["textbooks"].values())
    manager.save_registry(registry)
    
    print(f"📝 Updated registry: {len(registry['textbooks'])} textbooks, {registry['total_chunks']} total chunks")
    
    # Rebuild search index (this loads all documents temporarily)
    print("🔍 Rebuilding search index...")
    collection = manager.get_collection_for_search()
    
    search_engine = ProductionSearchEngine(cache_dir="data")
    search_engine.build_index_from_collection(collection, force_rebuild=True)
    
    # Show final stats
    stats = search_engine.get_statistics()
    print(f"\n🎯 SYSTEM UPDATED!")
    print("=" * 60)
    print(f"📚 Total textbooks: {stats['total_textbooks']}")
    print(f"📄 Total documents: {stats['total_documents']}")
    print(f"✅ Search index: Updated")
    
    print(f"\n📖 Available textbooks:")
    for textbook, count in stats['textbook_breakdown'].items():
        print(f"   • {textbook}: {count} chunks")
    
    print(f"\n📁 Storage structure:")
    print(f"   📚 Individual textbook files: data/textbooks/")
    print(f"   📝 Registry: textbook_registry.pkl")
    print(f"   🧠 Search vectors: faiss_semantic_search_vectors_384d.pkl")
    
    return True

def main():
    """Main function - configure your textbook here"""
    
    # CONFIGURE YOUR TEXTBOOK HERE
    textbook_config = {
        'id': 'nursing_skills_2e',
        'title': "Nursing Skills",
        'edition': '2nd Edition',
        'year': '2024',
        'pdf_path': '../data_files/Nursing-Skills-2e-1720739371._print.pdf',
        'chapter_csv_path': '../chapter_detection_system/output/Nursing-Skills-2e-1720739_chapters.csv'
    }
    
    success = process_single_textbook_optimized(textbook_config)
    
    if success:
        print(f"\n🚀 Ready for searches! Use production_search.py")
        print(f"\n💡 Performance benefits:")
        print(f"   ⚡ Separate files per textbook")
        print(f"   🧠 Only new textbook processed")
        print(f"   📈 Scales efficiently to hundreds of textbooks")
    else:
        print(f"\n❌ Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
