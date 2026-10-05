"""Scriptable JSON CLI; no source edits or working-directory-dependent imports."""

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .chapters import detect_chapters, save_chapters
from .config import Settings
from .ingestion import ingest_pdf, open_pdf
from .service import create_engine
from .storage import CorpusStore


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Medical textbook ingestion and cited retrieval")
    root.add_argument("--database", type=Path, help="Override MEDICAL_RAG_DATABASE")
    root.add_argument("--verbose", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect", help="Write a reviewable chapter CSV")
    detect.add_argument("pdf", type=Path)
    detect.add_argument("--output", type=Path, required=True)
    ingest = commands.add_parser("ingest", help="Atomically add or replace one textbook")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--id", required=True)
    ingest.add_argument("--title", required=True)
    ingest.add_argument("--edition", default="")
    ingest.add_argument("--year", default="")
    ingest.add_argument("--chapters", type=Path)
    ingest.add_argument("--chunk-size", type=int, default=800)
    ingest.add_argument("--chunk-overlap", type=int, default=100)
    index = commands.add_parser("index", help="Precompute corpus embeddings")
    index.add_argument("--force", action="store_true")
    search = commands.add_parser("search", help="Retrieve cited passages as JSON")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--textbook")
    commands.add_parser("list", help="List textbooks and chunk counts")
    delete = commands.add_parser("delete", help="Delete a textbook and invalidate vectors")
    delete.add_argument("textbook_id")
    delete.add_argument("--yes", action="store_true", required=True)
    return root


def run(args: argparse.Namespace, settings: Settings) -> object:
    if args.command == "detect":
        with open_pdf(args.pdf, settings.max_pdf_bytes) as doc:
            if len(doc) > settings.max_pdf_pages:
                raise ValueError("PDF exceeds the configured page limit")
            chapters = detect_chapters(doc)
        save_chapters(chapters, args.output)
        return {"output": str(args.output), "chapters": [asdict(chapter) for chapter in chapters]}
    if args.command == "ingest":
        result = ingest_pdf(
            args.pdf,
            args.id,
            args.title,
            chapter_csv=args.chapters,
            edition=args.edition,
            year=args.year,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_bytes=settings.max_pdf_bytes,
            max_pages=settings.max_pdf_pages,
        )
        CorpusStore(settings.database).replace_textbook(result.textbook, result.chunks)
        return {
            "textbook_id": result.textbook.id,
            "chunks": len(result.chunks),
            "chapters": len(result.chapters),
            "empty_pages": result.empty_pages,
        }
    if args.command == "list":
        return CorpusStore(settings.database).list_textbooks()
    if args.command == "delete":
        if not CorpusStore(settings.database).delete_textbook(args.textbook_id):
            raise ValueError("Textbook ID does not exist")
        return {"deleted": args.textbook_id}
    engine = create_engine(settings)
    if args.command == "index":
        count = engine.build_index(args.force)
        if not count:
            raise ValueError("Corpus is empty; ingest a PDF first")
        return {"indexed_chunks": count}
    return [result.to_dict() for result in engine.search(args.query, args.top_k, args.textbook)]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = Settings.from_env()
        if args.database:
            settings = replace(settings, database=args.database)
        print(json.dumps(run(args, settings), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
