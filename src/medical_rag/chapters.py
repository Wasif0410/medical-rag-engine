"""Conservative chapter detection and strict CSV interchange."""

import csv
import re
from pathlib import Path

import pymupdf

from .models import Chapter


def validate_chapters(chapters: list[Chapter], page_count: int) -> list[Chapter]:
    ordered = sorted(chapters, key=lambda chapter: chapter.start_page)
    if not ordered:
        raise ValueError("Chapter map is empty")
    previous_end = 0
    for chapter in ordered:
        if chapter.end_page > page_count:
            raise ValueError(f"Chapter '{chapter.title}' extends beyond the PDF")
        if chapter.start_page <= previous_end:
            raise ValueError("Chapter ranges overlap")
        previous_end = chapter.end_page
    return ordered


def detect_chapters(doc: pymupdf.Document) -> list[Chapter]:
    """Prefer top-level bookmarks, then explicit chapter headings, then whole document.

    Printed tables of contents are deliberately not guessed: printed page labels
    often differ from physical PDF pages. Supply a reviewed CSV for those books.
    """
    starts: dict[int, tuple[str, str, float]] = {}
    for level, title, page in doc.get_toc():
        if level == 1 and title.strip() and 1 <= page <= len(doc):
            starts.setdefault(page, (title.strip(), "bookmarks", 0.9))
    if not starts:
        pattern = re.compile(r"^(chapter\s+\d+\b|appendix\s+[A-Z]\b)", re.IGNORECASE)
        for index, page in enumerate(doc):
            # Heading candidates must occur in the first six nonempty lines.
            lines = [line.strip() for line in page.get_text().splitlines() if line.strip()]
            for line in lines[:6]:
                if pattern.match(line):
                    starts[index + 1] = (line[:200], "headings", 0.6)
                    break
    if not starts:
        return [Chapter("Full document", 1, len(doc), "fallback", 0.0)]
    if min(starts) > 1:
        starts[1] = ("Front matter", "fallback", 0.0)
    pages = sorted(starts)
    return [
        Chapter(
            starts[page][0],
            page,
            pages[index + 1] - 1 if index + 1 < len(pages) else len(doc),
            starts[page][1],
            starts[page][2],
        )
        for index, page in enumerate(pages)
    ]


def load_chapters(path: Path) -> list[Chapter]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"title", "start_page", "end_page"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Chapter CSV needs title, start_page, and end_page columns")
        return [
            Chapter(
                row["title"],
                int(row["start_page"]),
                int(row["end_page"]),
                row.get("method") or "manual",
                float(row.get("confidence") or 1.0),
            )
            for row in reader
        ]


def save_chapters(chapters: list[Chapter], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["title", "start_page", "end_page", "pages", "confidence", "method"])
        for chapter in chapters:
            writer.writerow(
                [
                    chapter.title,
                    chapter.start_page,
                    chapter.end_page,
                    chapter.end_page - chapter.start_page + 1,
                    chapter.confidence,
                    chapter.method,
                ]
            )
