"""
View all textbooks currently in the registry.
Usage:
    python view_textbooks.py
"""
import pickle
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")
REGISTRY_FILE = DATA_DIR / "textbook_registry.pkl"

with open(REGISTRY_FILE, "rb") as f:
    try:
        registry = pickle.load(f)
    except Exception as e:
        print(f"Error loading registry: {e}")
        registry = None

if registry is None:
    print("Registry could not be loaded.")
    exit(1)

if isinstance(registry, dict) and 'textbooks' in registry:
    textbooks_dict = registry['textbooks']
    print("Trained textbooks:")
    for name, entries in textbooks_dict.items():
        count = len(entries) if hasattr(entries, '__len__') else 1
        print(f"- {name} ({count} chunks)")
    print(f"\nTotal textbooks: {len(textbooks_dict)}")
else:
    print("No trained textbooks found in registry.")