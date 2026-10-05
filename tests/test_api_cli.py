import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from medical_rag.api import create_app
from medical_rag.cli import main
from medical_rag.config import Settings
from medical_rag.search import SearchEngine

API_KEY = "test-key-32-characters-or-longer-12345"


def test_api_auth_validation_and_citations(store, embedder):
    app = create_app(Settings(api_key=API_KEY), SearchEngine(store, embedder))
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.post("/v1/search", json={"query": "heart"}).status_code == 401
        assert (
            client.post(
                "/v1/search", json={"query": "heart"}, headers={"X-API-Key": "wrong"}
            ).status_code
            == 401
        )
        headers = {"X-API-Key": API_KEY}
        for payload in [
            {"query": " "},
            {"query": "heart", "top_k": 0},
            {"query": "heart", "top_k": True},
            {"query": "x" * 2001},
            {"query": "heart", "textbook_id": "../escape"},
        ]:
            assert client.post("/v1/search", json=payload, headers=headers).status_code == 422
        response = client.post("/v1/search", json={"query": "heart"}, headers=headers)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
        assert response.json()["results"][0]["citation"].endswith("PDF page 1")


def test_api_requires_credentials_by_default(store, embedder):
    with pytest.raises(RuntimeError, match="API_KEY"):
        with TestClient(create_app(Settings(), SearchEngine(store, embedder))):
            pass
    with TestClient(
        create_app(Settings(allow_unauthenticated=True), SearchEngine(store, embedder))
    ) as client:
        assert client.post("/v1/search", json={"query": "heart"}).status_code == 200


def test_large_request_and_malformed_json_are_rejected(store, embedder):
    with TestClient(create_app(Settings(api_key=API_KEY), SearchEngine(store, embedder))) as client:
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
        assert client.post("/v1/search", content=b"x" * 16385, headers=headers).status_code == 413
        assert client.post("/v1/search", content=b"{broken", headers=headers).status_code == 422


def test_empty_readiness_and_sanitized_error(store, embedder, monkeypatch, caplog):
    engine = SearchEngine(store, embedder)
    with TestClient(
        create_app(Settings(api_key=API_KEY), engine), raise_server_exceptions=False
    ) as client:

        def fail(*args, **kwargs):
            raise RuntimeError("sensitive source passage")

        monkeypatch.setattr(engine, "search", fail)
        response = client.post(
            "/v1/search", json={"query": "private query"}, headers={"X-API-Key": API_KEY}
        )
        assert response.status_code == 500
        assert "sensitive" not in response.text
        assert "sensitive" not in caplog.text
        assert "private query" not in caplog.text
        store.delete_textbook("book-a")
        assert client.get("/readyz").status_code == 503


def test_cli_ingestion_is_idempotent_and_does_not_load_model(pdf, tmp_path, capsys):
    database = str(tmp_path / "cli.sqlite3")
    command = ["--database", database, "ingest", str(pdf), "--id", "sample", "--title", "Sample"]
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["chunks"] == 3
    assert main(command) == 0
    capsys.readouterr()
    assert main(["--database", database, "list"]) == 0
    books = json.loads(capsys.readouterr().out)
    assert len(books) == 1
    assert books[0]["chunk_count"] == 3
    assert main(["--database", database, "delete", "sample", "--yes"]) == 0
    assert json.loads(capsys.readouterr().out) == {"deleted": "sample"}


def test_cli_failed_ingestion_exits_nonzero(tmp_path, capsys):
    assert (
        main(
            [
                "--database",
                str(tmp_path / "empty.sqlite3"),
                "ingest",
                "missing.pdf",
                "--id",
                "sample",
                "--title",
                "Sample",
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert not output.out
    assert "Error:" in output.err


def test_invalid_environment_and_secret_repr(monkeypatch):
    assert API_KEY not in repr(Settings(api_key=API_KEY))
    monkeypatch.setenv("MEDICAL_RAG_LOCAL_MODEL_ONLY", "maybe")
    with pytest.raises(ValueError):
        Settings.from_env()
    with pytest.raises(ValueError):
        replace(Settings(), max_pdf_pages=0)
