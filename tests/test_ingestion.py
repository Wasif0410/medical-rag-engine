from pathlib import Path

import pymupdf
import pytest

from medical_rag.chapters import detect_chapters, load_chapters, save_chapters
from medical_rag.chunking import split_text
from medical_rag.ingestion import ingest_pdf
from medical_rag.models import Chapter


def test_bookmarks_include_first_and_last_physical_page(pdf):
    with pymupdf.open(pdf) as doc:
        chapters = detect_chapters(doc)
    assert [(chapter.start_page, chapter.end_page) for chapter in chapters] == [(1, 2), (3, 3)]


def test_ingestion_has_exact_page_citations_and_keeps_short_passages(pdf):
    result = ingest_pdf(pdf, "sample", "Clinical Fundamentals")
    assert [chunk.page for chunk in result.chunks] == [1, 2, 3]
    assert result.chunks[-1].chapter == "Burns"
    assert result.chunks[-1].citation.endswith("PDF page 3")
    assert result.textbook.source_file == "sample.pdf"
    assert len(result.textbook.sha256) == 64


def test_whole_document_fallback_and_front_matter(pdf):
    with pymupdf.open(pdf) as doc:
        doc.set_toc([])
        assert detect_chapters(doc) == [Chapter("Full document", 1, 3, "fallback", 0.0)]
        doc.set_toc([[1, "Burns", 3]])
        chapters = detect_chapters(doc)
        assert chapters[0].title == "Front matter"
        assert chapters[0].end_page == 2


def test_heading_detection_uses_one_based_pages(tmp_path):
    path = tmp_path / "headings.pdf"
    with pymupdf.open() as doc:
        for text in ["Chapter 1 Introduction", "Body material", "Chapter 2 Results"]:
            doc.new_page().insert_text((72, 72), text)
        doc.save(path)
    result = ingest_pdf(path, "headings", "Heading Test")
    assert [(c.start_page, c.end_page) for c in result.chapters] == [(1, 2), (3, 3)]


def test_csv_round_trip_and_reviewed_map(pdf, tmp_path):
    path = tmp_path / "chapters.csv"
    chapters = [Chapter("Reviewed", 1, 3)]
    save_chapters(chapters, path)
    assert load_chapters(path) == chapters
    result = ingest_pdf(pdf, "sample", "Sample", chapter_csv=path)
    assert all(chunk.chapter == "Reviewed" for chunk in result.chunks)


@pytest.mark.parametrize(
    "rows",
    [
        "title,start_page,end_page\nBad,0,2\n",
        "title,start_page,end_page\nBad,1,4\n",
        "title,start_page,end_page\nA,1,2\nB,2,3\n",
        "wrong,columns\n1,2\n",
        "title,start_page,end_page\n",
    ],
)
def test_invalid_csv_fails_without_partial_ingestion(pdf, tmp_path, rows):
    path = tmp_path / "bad.csv"
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError):
        ingest_pdf(pdf, "sample", "Sample", chapter_csv=path)


def test_scan_without_text_is_rejected(tmp_path):
    path = tmp_path / "scan.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(path)
    with pytest.raises(ValueError, match="OCR"):
        ingest_pdf(path, "scan", "Scan")


def test_limits_and_invalid_id(pdf):
    for kwargs in [{"max_bytes": 1}, {"max_pages": 1}, {"chunk_overlap": 800}]:
        with pytest.raises(ValueError):
            ingest_pdf(pdf, "sample", "Sample", **kwargs)
    with pytest.raises(ValueError):
        ingest_pdf(pdf, "../escape", "Sample")
    with pytest.raises(ValueError):
        ingest_pdf(Path("missing.pdf"), "sample", "Sample")


def test_chunking_bounds_progress_and_source_coverage():
    text = "0123456789" * 50
    chunks = split_text(text, 100, 10)
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    assert chunks[0] == text[:100]
    assert chunks[1].startswith(text[90:100])
    assert chunks[-1].endswith(text[-10:])
    assert split_text("Short sentence.") == ["Short sentence."]
