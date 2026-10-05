"""Validated environment configuration, shared by CLI and API."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Pin the default model artifacts so persisted embeddings do not silently change.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true, false, 1, or 0")
    return value in {"true", "1"}


@dataclass(frozen=True)
class Settings:
    database: Path = Path("data/corpus.sqlite3")
    model: str = DEFAULT_MODEL
    model_revision: str | None = DEFAULT_REVISION
    local_model_only: bool = False
    api_key: str = field(default="", repr=False)
    allow_unauthenticated: bool = False
    max_pdf_bytes: int = 500 * 1024 * 1024
    max_pdf_pages: int = 10000

    def __post_init__(self) -> None:
        if self.max_pdf_bytes < 1 or self.max_pdf_pages < 1:
            raise ValueError("PDF limits must be positive")
        if not self.model.strip():
            raise ValueError("Embedding model cannot be empty")

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.getenv("MEDICAL_RAG_MODEL", DEFAULT_MODEL)
        revision = os.getenv("MEDICAL_RAG_MODEL_REVISION")
        if revision is None and model == DEFAULT_MODEL:
            revision = DEFAULT_REVISION
        return cls(
            database=Path(os.getenv("MEDICAL_RAG_DATABASE", "data/corpus.sqlite3")),
            model=model,
            model_revision=revision or None,
            local_model_only=env_bool("MEDICAL_RAG_LOCAL_MODEL_ONLY"),
            api_key=os.getenv("MEDICAL_RAG_API_KEY", ""),
            allow_unauthenticated=env_bool("MEDICAL_RAG_ALLOW_UNAUTHENTICATED"),
            max_pdf_bytes=int(os.getenv("MEDICAL_RAG_MAX_PDF_BYTES", str(500 * 1024 * 1024))),
            max_pdf_pages=int(os.getenv("MEDICAL_RAG_MAX_PDF_PAGES", "10000")),
        )
