"""Opt-in verification of real model inference and FAISS, beyond test doubles."""

import os
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from medical_rag.api import create_app
from medical_rag.config import Settings
from medical_rag.ingestion import ingest_pdf
from medical_rag.service import create_engine
from medical_rag.storage import CorpusStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("MEDICAL_RAG_RUN_INTEGRATION") != "true",
        reason="Set MEDICAL_RAG_RUN_INTEGRATION=true to download and test the model",
    ),
]


def test_real_pdf_to_semantic_search_and_authenticated_api(pdf, tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("sentence_transformers")
    settings = replace(
        Settings(),
        database=tmp_path / "semantic.sqlite3",
        api_key="integration-key-at-least-32-characters",
    )
    result = ingest_pdf(pdf, "sample", "Clinical Fundamentals")
    CorpusStore(settings.database).replace_textbook(result.textbook, result.chunks)
    engine = create_engine(settings)
    assert engine.build_index() == 3
    assert engine._index is not None
    assert engine.embedder.dimension == 384
    matches = engine.search("burn wound", 1)
    assert matches[0].chunk.page == 3
    assert matches[0].chunk.chapter == "Burns"
    # The second process can run offline using downloaded model and persisted vectors.
    offline = create_engine(replace(settings, local_model_only=True))
    assert offline.build_index() == 3
    with TestClient(create_app(settings, offline)) as client:
        response = client.post(
            "/v1/search",
            json={"query": "cardiac anatomy", "top_k": 1},
            headers={"X-API-Key": settings.api_key},
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["page"] == 1
