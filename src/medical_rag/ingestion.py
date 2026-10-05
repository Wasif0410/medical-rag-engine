"""PDF extraction with exact page provenance; persistence is a separate concern."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .chapters import detect_chapters, load_chapters, validate_chapters
from .chunking import split_text
from .models import Chapter, Chunk, Textbook

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    textbook: Textbook
    chunks: list[Chunk]
    chapters: list[Chapter]
    empty_pages: list[int]


def open_pdf(path: Path, max_bytes: int = 500 * 1024 * 1024) -> pymupdf.Document:
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("Source must be an existing PDF file")
    if path.stat().st_size > max_bytes:
        raise ValueError("PDF exceeds the configured size limit")
    doc = pymupdf.open(path)
    if not doc.is_pdf or doc.needs_pass or len(doc) == 0:
        doc.close()
        raise ValueError("Source must be a nonempty, unencrypted PDF")
    return doc


def ingest_pdf(
    path: Path,
    textbook_id: str,
    title: str,
    *,
    chapter_csv: Path | None = None,
    edition: str = "",
    year: str = "",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    max_bytes: int = 500 * 1024 * 1024,
    max_pages: int = 10000,
) -> IngestionResult:
    # Validate chunk settings before any expensive extraction.
    split_text("", chunk_size, chunk_overlap)
    with open_pdf(path, max_bytes) as doc:
        if len(doc) > max_pages:
            raise ValueError("PDF exceeds the configured page limit")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        textbook = Textbook(textbook_id, title, path.name, digest, edition, year)
        chapters = validate_chapters(
            load_chapters(chapter_csv) if chapter_csv else detect_chapters(doc), len(doc)
        )
        chunks = []
        empty_pages = []
        for chapter in chapters:
            for number in range(chapter.start_page, chapter.end_page + 1):
                text = doc[number - 1].get_text(sort=True).strip()
                if not text:
                    empty_pages.append(number)
                    continue
                for index, content in enumerate(split_text(text, chunk_size, chunk_overlap)):
                    chunk_id = hashlib.sha256(
                        f"{textbook_id}\0{number}\0{index}\0{content}".encode()
                    ).hexdigest()
                    chunks.append(
                        Chunk(
                            chunk_id,
                            textbook_id,
                            textbook.display_name,
                            chapter.title,
                            number,
                            content,
                        )
                    )
    if not chunks:
        raise ValueError("No text extracted; scanned PDFs require OCR before ingestion")
    if empty_pages:
        logger.warning("Skipped %d pages without extractable text", len(empty_pages))
    return IngestionResult(textbook, chunks, chapters, empty_pages)
