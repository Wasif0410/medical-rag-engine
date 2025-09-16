"""
Production Search Engine
=======================

Fast semantic search with caching for multiple textbooks.
"""

import pickle
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
from langchain.schema import Document

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from .models import TextbookCollection

class ProductionSearchEngine:
    """Production-ready search engine with caching"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "data"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        self.embedding_dim = 384
        
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"🧠 Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model loaded, embedding dimension: {self.embedding_dim}")
        else:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        if not HAS_FAISS:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")
    
    def build_index_from_collection(self, collection: TextbookCollection, force_rebuild: bool = False):
        """Build search index from textbook collection"""
        cache_path = os.path.join(self.cache_dir, "faiss_semantic_search_vectors_384d.pkl")
        
        # Try to load cached index first
        if not force_rebuild and self._load_cached_index(cache_path):
            print("🚀 Using cached search index!")
            return True
        
        print("🔄 Building new search index...")
        return self._build_fresh_index(collection.all_documents, cache_path)
    
    def build_index_from_documents(self, documents: List[Document], cache_name: str = "search_index.pkl", force_rebuild: bool = False):
        """Build search index from document list"""
        cache_path = os.path.join(self.cache_dir, cache_name)
        
        # Try to load cached index first
        if not force_rebuild and self._load_cached_index(cache_path):
            print("🚀 Using cached search index!")
            return True
        
        print("🔄 Building new search index...")
        return self._build_fresh_index(documents, cache_path)
    
    def _build_fresh_index(self, documents: List[Document], cache_path: str) -> bool:
        """Build a fresh search index"""
        print(f"📦 Building index for {len(documents)} documents...")
        
        self.documents = documents
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        print("🔄 Generating embeddings...")
        start_time = time.time()
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        embed_time = time.time() - start_time
        print(f"✅ Generated {len(self.embeddings)} embeddings in {embed_time:.2f}s")
        
        # Build FAISS index
        print("🏗️ Building FAISS index...")
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
        
        # Save to cache
        self._save_index_cache(cache_path)
        
        return True
    
    def _load_cached_index(self, cache_path: str) -> bool:
        """Load cached search index"""
        try:
            print(f"📂 Loading cached index from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.documents = cache_data['documents']
            self.embeddings = cache_data['embeddings']
            self.faiss_index = cache_data['faiss_index']
            self.embedding_dim = cache_data['embedding_dim']
            
            print(f"✅ Loaded cached index with {len(self.documents)} documents!")
            return True
            
        except FileNotFoundError:
            print(f"⚠️ No cached index found at {cache_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading cached index: {e}")
            return False
    
    def _save_index_cache(self, cache_path: str):
        """Save search index to cache"""
        cache_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'faiss_index': self.faiss_index,
            'embedding_dim': self.embedding_dim
        }
        
        print(f"💾 Saving search index to {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print("✅ Index cached for future use!")
    
    def search(self, query: str, top_k: int = 5, textbook_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.model or not self.faiss_index:
            print("❌ Search index not built")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search with more results if filtering
        search_k = top_k * 3 if textbook_filter else top_k
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Apply textbook filter if specified
                if textbook_filter and doc.metadata.get('textbook_id') != textbook_filter:
                    continue
                
                result = {
                    'document': doc,
                    'score': float(score),
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'citation': self._format_citation(doc)
                }
                results.append(result)
                
                # Stop when we have enough results
                if len(results) >= top_k:
                    break
        
        return results
    
    def _format_citation(self, doc: Document) -> str:
        """Format citation for a document"""
        metadata = doc.metadata
        textbook = metadata.get('textbook', 'Unknown Textbook')
        chapter = metadata.get('chapter', 'Unknown Chapter')
        page = metadata.get('page', 'Unknown')
        
        return f"{textbook}, {chapter}, page {page}"
    
    def get_textbook_list(self) -> List[str]:
        """Get list of available textbooks"""
        textbooks = set()
        for doc in self.documents:
            textbook_id = doc.metadata.get('textbook_id')
            if textbook_id:
                textbooks.add(textbook_id)
        return sorted(list(textbooks))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        textbook_counts = {}
        for doc in self.documents:
            textbook = doc.metadata.get('textbook', 'Unknown')
            textbook_counts[textbook] = textbook_counts.get(textbook, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'total_textbooks': len(textbook_counts),
            'textbook_breakdown': textbook_counts,
            'index_built': self.faiss_index is not None,
            'embedding_dimension': self.embedding_dim
        }
