FROM ghcr.io/astral-sh/uv:0.12.5 AS uv
FROM python:3.12-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 HF_HOME=/opt/models
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra api --extra semantic --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra api --extra semantic --no-editable
# Download the pinned model at build time, so serving does not need network access.
RUN .venv/bin/python -c "from medical_rag.config import Settings; from medical_rag.embeddings import SentenceTransformerEmbedder; s=Settings(); SentenceTransformerEmbedder(s.model, s.model_revision)"

FROM python:3.12-slim AS runtime
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/opt/models \
    MEDICAL_RAG_DATABASE=/data/corpus.sqlite3 \
    MEDICAL_RAG_LOCAL_MODEL_ONLY=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/models /opt/models
COPY examples ./examples
RUN mkdir /data && chown app:app /data
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)" || exit 1
CMD ["uvicorn", "medical_rag.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "32", "--timeout-keep-alive", "5", "--no-access-log"]
