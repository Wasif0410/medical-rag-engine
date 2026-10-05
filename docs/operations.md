# Operations

## Supported deployment

The provided configuration targets one CPU host and a controlled collection of text-based PDFs. Use one API worker per container, precompute embeddings with `medical-rag index`, and measure memory and query latency with the intended corpus before setting resource limits. Compose starts with two CPUs and a 2 GiB memory limit; those defaults are not a capacity guarantee.

The image builds the pinned model into `/opt/models`. Serving sets local-only model loading and does not need a provider API key. The corpus is stored in `/data/corpus.sqlite3` on a persistent volume. The API still needs a writable data volume for SQLite WAL and embedding cache updates, even with a read-only root filesystem.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEDICAL_RAG_DATABASE` | `data/corpus.sqlite3` | SQLite path; `/data/corpus.sqlite3` in Docker |
| `MEDICAL_RAG_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model name or immutable local path |
| `MEDICAL_RAG_MODEL_REVISION` | Pinned commit for default model | Use an immutable revision for remote models |
| `MEDICAL_RAG_LOCAL_MODEL_ONLY` | `false` | Refuse model downloads; `true` in Docker |
| `MEDICAL_RAG_API_KEY` | Empty | Required key of at least 32 characters |
| `MEDICAL_RAG_ALLOW_UNAUTHENTICATED` | `false` | Local development escape hatch; never enable on a remote service |
| `MEDICAL_RAG_MAX_PDF_BYTES` | `524288000` | Ingestion file limit: 500 MiB |
| `MEDICAL_RAG_MAX_PDF_PAGES` | `10000` | Ingestion page limit |

CLI `--database` overrides the environment and goes before the command. Native commands do not automatically load `.env`; use shell environment variables or `uv run --env-file .env`. Compose automatically reads `.env` but only passes the API key explicitly; add other variables to `compose.yaml` when changing container configuration. Changing the model requires rebuilding the image or mounting a complete immutable model directory.

## Remote serving

1. Keep the default localhost port binding; use a reverse proxy for remote access.
2. Terminate HTTPS at the proxy. Set request body limits to 16 KiB, rate limits, and bounded upstream timeouts.
3. Store the API key in the deployment's secret manager. Rotate it by replacing the secret and restarting the API; a process accepts one configured key.
4. Restrict access to the corpus and source PDFs. A shared key does not provide user identities, tenant separation, or per-textbook authorization.
5. Disable access logging that records request bodies, query parameters, or credentials. Search uses POST; application errors log request IDs and exception types, not source text.
6. Run ingestion separately from the serving process. Treat untrusted PDFs as parser inputs requiring isolation; file and page limits do not constitute a sandbox.

The application checks request sizes even for chunked bodies. There is no application-level distributed rate limiter or inference deadline. Uvicorn's concurrency limit bounds admitted requests; an inference lock serializes model work. Enforce stronger controls at the gateway, and avoid updating a large corpus during peak traffic because refresh can be expensive.

## Health and failures

- `/healthz` returns liveness without exposing corpus metadata.
- `/readyz` returns 200 only when a nonempty corpus can be indexed, otherwise 503. Docker uses readiness as its health check.
- Invalid or missing API credentials return 401; invalid search bodies return 422; oversized bodies return 413.
- Unexpected retrieval failures return a generic 500 with a request ID. Investigate locally using that ID and the logged exception type without enabling source-content logging.
- Startup fails when authentication is enabled with a missing or short key, or the configured model cannot be loaded.

An empty corpus allows startup for diagnostics but remains unready. A corrupt numeric cache is regenerated; a corrupt SQLite database requires restore from backup. An unavailable offline model requires a correct image build or local model directory.

## Updates, backups, and restores

Ingest a reviewed source PDF with the same ID to replace a textbook. Database replacement is atomic and cache invalidation occurs in the same transaction. Run `medical-rag index` after a batch of changes to warm the cache. A running API detects changes on its next retrieval; one in-flight request may finish using the previous consistent snapshot.

Use SQLite's backup API rather than copying a live database without its WAL. This native example creates a consistent backup:

```sh
uv run python -c "import sqlite3; src=sqlite3.connect('data/corpus.sqlite3'); dst=sqlite3.connect('data/backup.sqlite3'); src.backup(dst); dst.close(); src.close()"
```

Keep backups and original PDFs outside Git, with access controls and storage encryption appropriate to the source material. To restore, stop the API and ingestion processes, replace the database using a verified backup, remove obsolete WAL/SHM sidecars while all connections are closed, and restart. Verify known citations and readiness before routing traffic. Alternatively rebuild the corpus from the reviewed PDFs and chapter maps.

## Release validation

CI runs offline tests on Linux and Windows for Python 3.11–3.13, a real-model integration test, a pinned runtime dependency audit, and a non-root container smoke test. The smoke test generates a synthetic PDF, ingests it, builds the index, starts the service with an offline model and read-only root, and verifies a cited API result.

Run corpus-specific relevance and citation checks before deployment; the synthetic tests verify engineering behavior, not clinical accuracy. Resolve the project's license, [PyMuPDF licensing](https://pymupdf.io/licensing), and textbook processing rights before redistribution.
