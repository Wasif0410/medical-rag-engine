# Contributing

Use Python 3.11–3.13 and uv. Create a branch from `main` and keep commits focused on one change.

```sh
uv sync --locked --all-extras
uv run --all-extras ruff check src tests examples
uv run --all-extras ruff format --check src tests examples
uv run --all-extras pytest -m "not integration"
uv build
```

For model or search changes, also run the integration test with `MEDICAL_RAG_RUN_INTEGRATION=true`. It downloads the pinned model on its first run. Unit tests inject a deterministic embedder and generate PDFs locally, so they do not need network access or private documents.

Keep domain and ingestion code independent of API and CLI presentation. Preserve one-based physical PDF page citations, transactional corpus updates, exact textbook filtering, and numeric-only cache formats. Add regression tests for behavior changes, and update the README and operations guide when commands or deployment requirements change.

Update `uv.lock` with dependency changes. Use the CPU source configuration for PyTorch, and use immutable model revisions for reproducibility. Never commit virtual environments, keys, patient records, source textbooks, generated PDFs, databases, or caches.

Pull requests should explain the problem, resulting behavior, and relevant validation. CI must pass before merging. The repository has not yet selected a project license; do not add a license or import differently licensed source without the maintainer's decision.
