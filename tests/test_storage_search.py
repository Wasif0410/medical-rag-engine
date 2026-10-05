import sqlite3

import numpy as np
import pytest

from medical_rag.models import Chunk, Textbook
from medical_rag.search import SearchEngine, normalize_vectors
from medical_rag.storage import CorpusStore


def test_top_k_larger_than_corpus_never_duplicates_last_result(store, embedder):
    results = SearchEngine(store, embedder).search("heart", 100)
    assert len(results) == 2
    assert len({result.chunk.id for result in results}) == 2
    assert results[0].chunk.id == "a1"
    assert 0.99 < results[0].score <= 1.001


def test_filter_ranks_only_matching_book_even_beyond_overfetch_window(store, embedder):
    store.replace_textbook(
        Textbook("book-b", "Second Book", "b.pdf", "def"),
        [Chunk("b1", "book-b", "Second Book", "Burns", 8, "burn")],
    )
    store.replace_textbook(
        Textbook("book-c", "Distractors", "c.pdf", "ghi"),
        [Chunk(f"c{i}", "book-c", "Distractors", "Cardiology", i + 1, "heart") for i in range(30)],
    )
    engine = SearchEngine(store, embedder)
    results = engine.search("heart", 1, "book-b")
    assert len(results) == 1
    assert results[0].chunk.id == "b1"
    assert engine.search("heart", 5, "missing") == []


def test_cache_reuse_and_same_id_content_replacement(store, embedder):
    engine = SearchEngine(store, embedder)
    engine.build_index()
    assert len(embedder.calls) == 1
    SearchEngine(store, embedder).build_index()
    assert len(embedder.calls) == 1
    store.replace_textbook(
        Textbook("book-a", "Updated Edition", "a.pdf", "new"),
        [Chunk("new", "book-a", "Updated Edition", "Burns", 5, "burn")],
    )
    results = engine.search("burn")
    assert len(results) == 1
    assert results[0].chunk.id == "new"
    assert results[0].chunk.citation == "Updated Edition, Burns, PDF page 5"


def test_model_identity_invalidates_cache(store, embedder):
    SearchEngine(store, embedder).build_index()
    embedder.identity = "test:model-v2"
    SearchEngine(store, embedder).build_index()
    assert len(embedder.calls) == 2


def test_delete_refreshes_existing_engine_and_handles_empty_corpus(store, embedder):
    engine = SearchEngine(store, embedder)
    engine.build_index()
    assert store.delete_textbook("book-a")
    assert not store.delete_textbook("book-a")
    assert engine.search("heart") == []
    assert store.list_textbooks() == []


def test_failed_replace_rolls_back_old_book_and_revision(store):
    revision = store.revision()
    duplicate = Chunk("duplicate", "book-a", "Sample", "Chapter", 1, "text")
    with pytest.raises(sqlite3.IntegrityError):
        store.replace_textbook(
            Textbook("book-a", "Replacement", "x.pdf", "xyz"), [duplicate, duplicate]
        )
    assert store.revision() == revision
    assert store.list_textbooks()[0]["title"] == "Clinical Fundamentals"
    assert len(store.snapshot()[1]) == 2


def test_cache_write_from_stale_snapshot_is_discarded(store):
    revision = store.revision()
    store.delete_textbook("book-a")
    store.save_vectors("stale", np.ones((2, 3), dtype=np.float32), revision)
    assert store.load_vectors("stale") is None


def test_corrupted_numeric_cache_is_rebuilt(store, embedder):
    SearchEngine(store, embedder).build_index()
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE embedding_cache SET vectors=?", (b"corrupted data",))
    SearchEngine(store, embedder).build_index()
    assert len(embedder.calls) == 2


@pytest.mark.parametrize(
    "query,k",
    [
        ("", 1),
        ("  ", 1),
        ("a" * 2001, 1),
        ("heart", 0),
        ("heart", -1),
        ("heart", 101),
        ("heart", True),
    ],
)
def test_invalid_search_arguments(store, embedder, query, k):
    with pytest.raises(ValueError):
        SearchEngine(store, embedder).search(query, k)


@pytest.mark.parametrize(
    "vectors", [np.zeros((1, 3)), np.ones((2, 3)), np.array([[float("nan"), 1, 1]])]
)
def test_invalid_model_output_rejected(vectors):
    with pytest.raises(ValueError):
        normalize_vectors(vectors, 1, 3)


def test_unsupported_schema_rejected(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="schema"):
        CorpusStore(path)
