"""Lazy model adapter; importing the package never downloads a model."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model: str, revision: str | None = None, *, local_only: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the semantic extra: uv sync --extra semantic") from exc
        self.model_name = model
        self.revision = revision
        self._model = SentenceTransformer(
            model,
            revision=revision,
            local_files_only=local_only,
            device="cpu",
            trust_remote_code=False,
            model_kwargs={"use_safetensors": True},
        )
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        dimension = (
            get_dimension()
            if get_dimension is not None
            else self._model.get_sentence_embedding_dimension()
        )
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("Embedding model has no supported output dimension")
        self._dimension = dimension

    @property
    def identity(self) -> str:
        return f"sentence-transformers:{self.model_name}@{self.revision or 'default'}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
