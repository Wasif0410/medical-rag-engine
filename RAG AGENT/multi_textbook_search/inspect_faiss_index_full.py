import pickle
from pathlib import Path

INDEX_FILE = Path("data/faiss_semantic_search_vectors_384d.pkl")

with open(INDEX_FILE, "rb") as f:
    faiss_data = pickle.load(f)

print("FAISS index type:", type(faiss_data))
if isinstance(faiss_data, dict):
    for k, v in faiss_data.items():
        print(f"Key: {k}, Type: {type(v)}, Length: {len(v) if hasattr(v, '__len__') else 'N/A'}")
else:
    print("Not a dict. Type:", type(faiss_data))