from pathlib import Path

import numpy as np
import pymupdf
import pytest

from medical_rag.models import Chunk, Textbook
from medical_rag.storage import CorpusStore


class FakeEmbedder:
    """Deterministic test dependency; never exposed as a production backend."""

    identity = "test:model-v1"
    dimension = 3

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array(
            [[text.lower().count("heart"), text.lower().count("burn"), 0.1] for text in texts],
            dtype=np.float32,
        )


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    with pymupdf.open() as doc:
        for text in [
            "Heart anatomy and cardiac structure.",
            "Heart rhythms and assessment.",
            "Burn wound care and assessment.",
        ]:
            page = doc.new_page()
            page.insert_text((72, 72), text)
        doc.set_toc([[1, "Cardiology", 1], [1, "Burns", 3]])
        doc.save(path)
    return path


@pytest.fixture
def store(tmp_path: Path) -> CorpusStore:
    corpus = CorpusStore(tmp_path / "corpus.sqlite3")
    corpus.replace_textbook(
        Textbook("book-a", "Clinical Fundamentals", "a.pdf", "abc"),
        [
            Chunk("a1", "book-a", "Clinical Fundamentals", "Cardiology", 1, "heart heart"),
            Chunk("a2", "book-a", "Clinical Fundamentals", "Burns", 2, "burn burn"),
        ],
    )
    return corpus
