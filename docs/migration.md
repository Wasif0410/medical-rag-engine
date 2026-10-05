# Migrating from the prototype

Version 0.2 replaces the `RAG AGENT/` script layout with an installable `src/medical_rag` package. The old scripts, virtual environment, bytecode, detected chapter outputs, registries, and embedding pickle files are removed from the tracked tree. Historical commits remain intact.

## Rebuild from source PDFs

1. Keep your source PDFs outside Git and back up existing local artifacts.
2. Install dependencies with `uv sync --locked --all-extras`.
3. Run `medical-rag detect` and review the chapter CSV against physical PDF pages.
4. Run `medical-rag ingest` once per textbook with a stable ID and title.
5. Run `medical-rag index`, then check several known passages and citations.

```sh
uv run --all-extras medical-rag detect /path/to/book.pdf --output data/book-chapters.csv
uv run --all-extras medical-rag ingest /path/to/book.pdf --id book-2e --title "Book Title" --edition "2nd edition" --chapters data/book-chapters.csv
uv run --all-extras medical-rag index
uv run --all-extras medical-rag search "known source passage" --textbook book-2e
```

The old detector mixed zero-based and one-based boundaries, including incorrect final-page handling. Regenerate chapter maps rather than assuming previous CSVs are correct. New CSVs must use inclusive, one-based physical PDF page numbers.

## Interface mapping

| Previous entry point | Replacement |
| --- | --- |
| `chapter_detection_system/run.py` | `medical-rag detect PDF --output CSV` |
| Hardcoded config in `optimized_pipeline.py` | `medical-rag ingest PDF --id ID --title TITLE` |
| `production_search.py` | `medical-rag search QUERY --top-k K --textbook ID` |
| `view_textbooks.py` | `medical-rag list` |
| `delete_textbook.py` | `medical-rag delete ID --yes` |
| `data/*.pkl` | `data/corpus.sqlite3` and validated numeric vectors |

The new format does not deserialize legacy pickle files. Those files can execute Python code during loading and contain class references tied to the old layout. Reingestion preserves traceability without executing old data. PDFs themselves are not bundled in the repository.

## Repository rename

The canonical repository is `Wasif0410/medical-rag-engine`. Update existing remotes:

```sh
git remote set-url origin https://github.com/Wasif0410/medical-rag-engine.git
```

Removing files from the current tree does not remove their blobs from Git history. Existing clones still contain historical artifacts; shrinking history would require a separately coordinated history rewrite. No history rewrite is part of this migration.
