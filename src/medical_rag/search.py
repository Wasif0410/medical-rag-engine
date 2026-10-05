"""Snapshot-consistent cosine search with exact textbook filtering."""

import hashlib
import json
import logging
import threading
from dataclasses import asdict

import numpy as np

from .embeddings import Embedder
from .models import Chunk, SearchResult, validate_textbook_id
from .storage import CorpusStore

logger = logging.getLogger(__name__)


def normalize_vectors(values: np.ndarray, rows: int, dimension: int) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.shape != (rows, dimension) or not np.isfinite(vectors).all():
        raise ValueError("Embedding shape or values are invalid")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Embedding model produced zero or invalid vectors")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


class SearchEngine:
    def __init__(self, store: CorpusStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self._lock = threading.RLock()
        self._revision = -1
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._index = None

    def _refresh(self, force: bool = False) -> None:
        if not force and self._revision == self.store.revision():
            return
        revision, chunks = self.store.snapshot()
        if not chunks:
            self._chunks, self._vectors, self._index, self._revision = [], None, None, revision
            return
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "schema": 1,
                    "model": self.embedder.identity,
                    "dimension": self.embedder.dimension,
                    "chunks": [asdict(chunk) for chunk in chunks],
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        vectors = None if force else self.store.load_vectors(fingerprint)
        if vectors is not None:
            try:
                vectors = normalize_vectors(vectors, len(chunks), self.embedder.dimension)
            except ValueError:
                vectors = None
        if vectors is None:
            logger.info("Encoding corpus with %d chunks", len(chunks))
            vectors = normalize_vectors(
                self.embedder.encode([chunk.text for chunk in chunks]),
                len(chunks),
                self.embedder.dimension,
            )
            self.store.save_vectors(fingerprint, vectors, revision)
        try:
            import faiss
        except ImportError:
            index = None  # Exact NumPy search is useful for small corpora and offline tests.
        else:
            index = faiss.IndexFlatIP(self.embedder.dimension)
            index.add(vectors)
        # Publish only a fully built snapshot; failed builds leave no partial index.
        self._chunks, self._vectors, self._index, self._revision = chunks, vectors, index, revision

    def build_index(self, force: bool = False) -> int:
        with self._lock:
            self._refresh(force)
            return len(self._chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        textbook_id: str | None = None,
    ) -> list[SearchResult]:
        if not query.strip() or len(query) > 2000:
            raise ValueError("Query must contain 1–2000 characters with nonempty text")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        if textbook_id is not None:
            validate_textbook_id(textbook_id)
        with self._lock:
            self._refresh()
            if not self._chunks:
                return []
            query_vector = normalize_vectors(
                self.embedder.encode([query.strip()]),
                1,
                self.embedder.dimension,
            )
            if textbook_id is not None or self._index is None:
                # Filter before ranking: overfetching cannot guarantee filtered recall.
                candidates = np.array(
                    [
                        index
                        for index, chunk in enumerate(self._chunks)
                        if textbook_id is None or chunk.textbook_id == textbook_id
                    ],
                    dtype=np.int64,
                )
                if not len(candidates):
                    return []
                scores = self._vectors[candidates] @ query_vector[0]
                order = np.argsort(-scores, kind="stable")[:top_k]
                return [SearchResult(self._chunks[candidates[i]], float(scores[i])) for i in order]
            scores, indices = self._index.search(query_vector, min(top_k, len(self._chunks)))
            return [
                SearchResult(self._chunks[int(index)], float(score))
                for score, index in zip(scores[0], indices[0], strict=True)
                if 0 <= index < len(self._chunks)
            ]
