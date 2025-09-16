"""
Delete a textbook from the vector index and registry.
Usage:
    python delete_textbook.py "Textbook Name"
"""
import pickle
import sys
from pathlib import Path

DATA_DIR = Path("data")
INDEX_FILE = DATA_DIR / "faiss_semantic_search_vectors_384d.pkl"
REGISTRY_FILE = DATA_DIR / "textbook_registry.pkl"

if len(sys.argv) < 2:
    print("Usage: python delete_textbook.py \"Textbook Name\"")
    sys.exit(1)

textbook_name = sys.argv[1].strip().lower()

# Load registry
with open(REGISTRY_FILE, "rb") as f:
    registry = pickle.load(f)

if not (isinstance(registry, dict) and 'textbooks' in registry):
    print("Registry format not supported for deletion.")
    sys.exit(1)

textbooks_dict = registry['textbooks']
if textbook_name not in textbooks_dict:
    print(f"No entries found for textbook: {textbook_name}")
    sys.exit(0)

# Remove textbook from registry
del textbooks_dict[textbook_name]
registry['textbooks'] = textbooks_dict

# Load FAISS index
with open(INDEX_FILE, "rb") as f:
    faiss_data = pickle.load(f)

embeddings = faiss_data['embeddings']
documents = faiss_data['documents']

# If no textbooks remain, clear all vectors and documents
if not textbooks_dict:
    keep_embeddings = []
    keep_documents = []
else:
    keep_embeddings = []
    keep_documents = []
    current = 0
    all_textbooks = list(textbooks_dict.keys())
    for name in all_textbooks:
        num_chunks = len(textbooks_dict.get(name, []))
        for _ in range(num_chunks):
            if current < len(embeddings):
                keep_embeddings.append(embeddings[current])
            if current < len(documents):
                keep_documents.append(documents[current])
            current += 1

# Save updated registry
with open(REGISTRY_FILE, "wb") as f:
    pickle.dump(registry, f)

# Save updated index
faiss_data['embeddings'] = keep_embeddings
faiss_data['documents'] = keep_documents
# Force clear documents if no textbooks remain
if not textbooks_dict:
    faiss_data['documents'] = []
with open(INDEX_FILE, "wb") as f:
    pickle.dump(faiss_data, f)

print(f"✅ Deleted all entries for textbook: {textbook_name} (embeddings and documents)")