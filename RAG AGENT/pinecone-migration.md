# Pinecone Migration Guide

## Overview

This document outlines the migration strategy from the current FAISS-based local vector search to Pinecone cloud-based vector database for the RAG Agent textbook search system.

## Current Architecture Analysis

### What Stays the Same ✅

#### 1. **Textbook Extraction & Processing**

- `ProductionTextbookProcessor` class remains **completely unchanged**
- Chapter detection and CSV integration work exactly the same
- PDF text extraction with PyPDF2 stays identical
- Chunking strategy (800 chars, 100 overlap) remains the same
- Document metadata structure stays intact

#### 2. **Data Models**

- `TextbookInfo`, `ProcessingResult`, and `TextbookCollection` classes unchanged
- Document structure and metadata remain the same
- Chapter-based chunking logic preserved

#### 3. **Core Processing Pipeline**

- Chapter detection system integration
- Text extraction from PDFs
- Document chunking and metadata assignment
- Textbook registry management

### What Changes 🔄

#### 1. **Search Engine Architecture**

**Current System (FAISS):**

```python
class ProductionSearchEngine:
    - Local FAISS index
    - Pickle files for caching
    - Local sentence-transformers embeddings
    - File-based storage in data/ directory
```

**New System (Pinecone):**

```python
class PineconeSearchEngine:
    - Cloud-based Pinecone index
    - API-based vector storage
    - Same sentence-transformers embeddings
    - Managed infrastructure
```

#### 2. **Dependencies**

**Remove:**

```
faiss-cpu
```

**Add:**

```
pinecone-client
```

#### 3. **Configuration**

- Add Pinecone API key management
- Environment configuration (production/development)
- Index naming conventions

## Migration Implementation

### 1. New Pinecone Search Engine

```python
"""
Pinecone Search Engine
=====================

Cloud-based semantic search with Pinecone for multiple textbooks.
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import pinecone
from sentence_transformers import SentenceTransformer

from .models import TextbookCollection

class PineconeSearchEngine:
    """Production-ready search engine with Pinecone"""

    def __init__(self,
                 api_key: str,
                 environment: str,
                 index_name: str,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.model_name = model_name
        self.embedding_dim = 384

        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)

        # Load sentence transformer model
        print(f"🧠 Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded, embedding dimension: {self.embedding_dim}")

        # Connect to or create index
        self._setup_index()

    def _setup_index(self):
        """Setup Pinecone index"""
        if self.index_name not in pinecone.list_indexes():
            print(f"🏗️ Creating Pinecone index: {self.index_name}")
            pinecone.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric='cosine'
            )
            print("✅ Index created!")
        else:
            print(f"📂 Using existing index: {self.index_name}")

        self.index = pinecone.Index(self.index_name)

    def build_index_from_collection(self, collection: TextbookCollection, batch_size: int = 100):
        """Build search index from textbook collection"""
        print(f"📦 Building Pinecone index for {len(collection.all_documents)} documents...")

        # Process documents in batches
        documents = collection.all_documents
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]

            print(f"🔄 Processing batch {batch_idx + 1}/{total_batches}")

            # Generate embeddings for batch
            texts = [doc.page_content for doc in batch_docs]
            embeddings = self.model.encode(texts, show_progress_bar=False)

            # Prepare vectors for Pinecone
            vectors = []
            for i, (doc, embedding) in enumerate(zip(batch_docs, embeddings)):
                vector_id = f"doc_{start_idx + i}"
                vectors.append({
                    'id': vector_id,
                    'values': embedding.tolist(),
                    'metadata': {
                        'textbook_id': doc.metadata.get('textbook_id', ''),
                        'textbook_title': doc.metadata.get('textbook_title', ''),
                        'chapter': doc.metadata.get('chapter', ''),
                        'page_range': doc.metadata.get('page_range', ''),
                        'chunk_index': doc.metadata.get('chunk_index', 0),
                        'content': doc.page_content[:1000]  # Truncate for metadata limits
                    }
                })

            # Upload batch to Pinecone
            self.index.upsert(vectors)

        print(f"✅ Successfully uploaded {len(documents)} documents to Pinecone!")
        return True

    def search(self, query: str, top_k: int = 5, textbook_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        print(f"🔍 Searching for: '{query}'")

        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()

        # Prepare search filters
        filter_dict = {}
        if textbook_filter:
            filter_dict['textbook_id'] = textbook_filter

        # Search Pinecone
        search_args = {
            'vector': query_embedding,
            'top_k': top_k,
            'include_metadata': True
        }
        if filter_dict:
            search_args['filter'] = filter_dict

        results = self.index.query(**search_args)

        # Format results
        formatted_results = []
        for match in results['matches']:
            result = {
                'score': match['score'],
                'content': match['metadata']['content'],
                'metadata': {
                    'textbook_id': match['metadata']['textbook_id'],
                    'textbook_title': match['metadata']['textbook_title'],
                    'chapter': match['metadata']['chapter'],
                    'page_range': match['metadata']['page_range'],
                    'chunk_index': match['metadata']['chunk_index']
                }
            }
            formatted_results.append(result)

        print(f"✅ Found {len(formatted_results)} results")
        return formatted_results

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats['total_vector_count'],
            'dimension': stats['dimension'],
            'index_fullness': stats['index_fullness']
        }

    def delete_textbook(self, textbook_id: str):
        """Delete all vectors for a specific textbook"""
        print(f"🗑️ Deleting textbook: {textbook_id}")
        self.index.delete(filter={'textbook_id': textbook_id})
        print("✅ Textbook deleted from index")
```

### 2. Configuration Management

```python
"""
Pinecone Configuration
=====================
"""

import os
from typing import Optional

class PineconeConfig:
    """Pinecone configuration management"""

    def __init__(self):
        self.api_key = self._get_api_key()
        self.environment = self._get_environment()
        self.index_name = self._get_index_name()

    def _get_api_key(self) -> str:
        """Get Pinecone API key from environment"""
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        return api_key

    def _get_environment(self) -> str:
        """Get Pinecone environment"""
        return os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')

    def _get_index_name(self) -> str:
        """Get Pinecone index name"""
        return os.getenv('PINECONE_INDEX_NAME', 'rag-agent-textbooks')

    @property
    def is_configured(self) -> bool:
        """Check if Pinecone is properly configured"""
        try:
            return bool(self.api_key and self.environment and self.index_name)
        except ValueError:
            return False
```

### 3. Environment Variables

Create a `.env` file in the root directory:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=rag-agent-textbooks
```

### 4. Updated Requirements

```txt
# Core dependencies (unchanged)
PyPDF2>=3.0.1
langchain>=0.0.200
sentence-transformers>=2.2.2

# Replace FAISS with Pinecone
pinecone-client>=2.2.4

# Additional dependencies
python-dotenv>=1.0.0
```

## Migration Steps

### Phase 1: Setup

1. **Install Pinecone**: `pip install pinecone-client python-dotenv`
2. **Create Pinecone account** and get API key
3. **Set environment variables** in `.env` file
4. **Test connection** with simple script

### Phase 2: Implementation

1. **Create new `pinecone_search_engine.py`** with the implementation above
2. **Update imports** in main application files
3. **Create configuration management** system
4. **Add error handling** for API failures

### Phase 3: Migration

1. **Run existing system** to generate document collection
2. **Initialize Pinecone index** with new search engine
3. **Upload all documents** to Pinecone
4. **Test search functionality** with existing queries
5. **Validate results** match FAISS output

### Phase 4: Cleanup

1. **Remove FAISS dependencies** from requirements.txt
2. **Delete local cache files** (\*.pkl files in data/)
3. **Update documentation** and README files
4. **Archive old search engine** code

## Benefits of Migration

### 1. **Scalability**

- No local storage limits
- Handles millions of documents
- Automatic scaling based on usage

### 2. **Performance**

- Managed infrastructure with optimized hardware
- Global CDN for fast access
- Built-in caching and optimization

### 3. **Collaboration**

- Multiple users can access same index
- Shared vector database across team
- No local setup required for new users

### 4. **Reliability**

- Built-in backups and redundancy
- 99.9% uptime SLA
- Automatic failover

### 5. **Advanced Features**

- Hybrid search (dense + sparse)
- Advanced filtering capabilities
- Namespace support for multi-tenancy
- Real-time updates

## Cost Considerations

### Pinecone Pricing (Starter Plan)

- **Free tier**: 1M vectors, 1 pod
- **Paid plans**: Start at $70/month for 5M vectors
- **Pay-as-you-scale**: Additional pods and storage

### Current Local Costs

- Storage: Local disk space
- Compute: Local CPU for search
- Maintenance: Manual backup and management

## Testing Strategy

### 1. **Parallel Testing**

- Keep FAISS system running during migration
- Compare search results between systems
- Validate performance and accuracy

### 2. **Gradual Rollout**

- Start with development environment
- Test with subset of textbooks
- Full production migration after validation

### 3. **Rollback Plan**

- Keep FAISS backup for quick rollback
- Document restoration procedure
- Test rollback process before migration

## Conclusion

The migration to Pinecone is **highly compatible** with the existing system architecture. The well-designed separation of concerns means that:

- **0% change** to textbook processing and chunking
- **100% change** only to the search engine layer
- **Minimal changes** to application code using the search engine

This migration will provide significant benefits in scalability, performance, and reliability while maintaining all existing functionality.
