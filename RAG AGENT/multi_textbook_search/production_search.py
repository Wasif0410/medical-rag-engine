"""
Quick Search - Production Ready
==============================

Simple search interface for the production system.
"""

import sys
import pickle
from core.search_engine import ProductionSearchEngine
from core.models import TextbookCollection

def quick_search(query: str, top_k: int = 3, textbook_filter: str = None):
    """Quick search using production system"""
    print(f"🔍 QUICK SEARCH: '{query}'")
    if textbook_filter:
        print(f"📚 Filtered to: {textbook_filter}")
    print("=" * 60)
    
    try:
        # Load collection or documents
        try:
            # Try optimized structure first
            from optimized_pipeline import OptimizedTextbookManager
            manager = OptimizedTextbookManager()
            collection = manager.get_collection_for_search()
            documents = collection.all_documents
            print(f"📚 Loaded optimized collection with {len(documents)} documents from {len(collection.textbooks)} textbooks")
        except (FileNotFoundError, ImportError):
            try:
                # Fall back to legacy unified file
                with open('data/multi_textbook_processed_documents.pkl', 'rb') as f:
                    collection = pickle.load(f)
                documents = collection.all_documents
                print(f"📚 Loaded legacy collection with {len(documents)} documents from {len(collection.textbooks)} textbooks")
            except FileNotFoundError:
                # Fall back to old single textbook data
                with open('data/rosen_enhanced_chapter_aware_complete.pkl', 'rb') as f:
                    documents = pickle.load(f)
                collection = None
                print(f"📚 Loaded {len(documents)} documents (single textbook mode)")
        
        # Create search engine
        search_engine = ProductionSearchEngine(cache_dir="data")
        
        # Build/load index
        if collection and hasattr(collection, 'all_documents'):
            search_engine.build_index_from_collection(collection)
        else:
            search_engine.build_index_from_documents(documents, "quick_search_cache.pkl")
        
        # Search
        results = search_engine.search(query, top_k=top_k, textbook_filter=textbook_filter)
        
        # Display results
        print(f"\n🎯 Found {len(results)} results:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            doc = result['document']
            score = result['score']
            
            print(f"\n📖 Result {i}:")
            print(f"   📚 {doc.metadata.get('textbook', 'Unknown Textbook')}")
            print(f"   📑 {doc.metadata.get('chapter', 'Unknown Chapter')}")
            print(f"   📄 Page {doc.metadata.get('page', 'Unknown')}")
            print(f"   💯 Similarity: {score:.3f}")
            
            # Show content preview with improved formatting
            content = doc.page_content.strip()
            # Remove excessive whitespace and line breaks
            import re
            content = re.sub(r'\s+', ' ', content)
            # Optionally, split into sentences and show first 2-3
            sentences = re.split(r'(?<=[.!?])\s+', content)
            preview = ' '.join(sentences[:3])
            if len(sentences) > 3:
                preview += ' ...'
            print(f"   📝 Content: {preview}")
        
        print(f"\n⏱️ Search completed!")
        return results
        
    except FileNotFoundError as e:
        print(f"❌ Data not found: {e}")
        print("💡 Run production_pipeline.py first to process textbooks")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def interactive_search():
    """Interactive search mode"""
    print("🔍 INTERACTIVE QUICK SEARCH")
    print("=" * 40)
    print("💡 Type 'help' for commands")
    print("=" * 40)
    
    while True:
        query = input("\n🔍 Enter search query (or 'quit'): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif query.lower() == 'help':
            print("\n📖 Available commands:")
            print("   • Regular text - search query")
            print("   • 'quit' or 'q' - exit")
            print("   • 'help' - show this help")
            continue
        elif not query:
            continue
        
        quick_search(query)
        print("\n" + "-" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line query
        query = " ".join(sys.argv[1:])
        quick_search(query)
    else:
        # Interactive mode
        interactive_search()
