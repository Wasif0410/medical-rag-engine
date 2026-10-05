"""Domain types shared by ingestion, persistence, and retrieval."""

import re
from dataclasses import dataclass


def validate_textbook_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}", value):
        raise ValueError("Textbook ID must be 1–100 letters, digits, underscores, or hyphens")
    return value


@dataclass(frozen=True)
class Chapter:
    """Inclusive, one-based physical PDF page boundaries."""

    title: str
    start_page: int
    end_page: int
    method: str = "manual"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.title.strip() or not 1 <= self.start_page <= self.end_page:
            raise ValueError("Chapter requires a title and positive, ordered page boundaries")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Chapter confidence must be between zero and one")


@dataclass(frozen=True)
class Textbook:
    id: str
    title: str
    source_file: str
    sha256: str
    edition: str = ""
    year: str = ""

    def __post_init__(self) -> None:
        validate_textbook_id(self.id)
        if not self.title.strip():
            raise ValueError("Textbook title cannot be empty")

    @property
    def display_name(self) -> str:
        return " ".join(part for part in (self.title, self.edition, self.year) if part)


@dataclass(frozen=True)
class Chunk:
    id: str
    textbook_id: str
    textbook_title: str
    chapter: str
    page: int
    text: str

    def __post_init__(self) -> None:
        validate_textbook_id(self.textbook_id)
        if self.page < 1 or not self.text.strip():
            raise ValueError("Chunk requires a positive page and nonempty text")

    @property
    def citation(self) -> str:
        return f"{self.textbook_title}, {self.chapter}, PDF page {self.page}"


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.id,
            "textbook_id": self.chunk.textbook_id,
            "textbook_title": self.chunk.textbook_title,
            "chapter": self.chunk.chapter,
            "page": self.chunk.page,
            "content": self.chunk.text,
            "score": self.score,
            "citation": self.chunk.citation,
        }
