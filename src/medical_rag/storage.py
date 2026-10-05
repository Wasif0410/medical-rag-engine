"""Transactional SQLite corpus storage with numeric-only embedding caches."""

import io
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .models import Chunk, Textbook, validate_textbook_id

SCHEMA_VERSION = 1


class CorpusStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError(f"Unsupported database schema {version}")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS textbooks (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, source_file TEXT NOT NULL,
                    sha256 TEXT NOT NULL, edition TEXT NOT NULL, year TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, textbook_id TEXT NOT NULL REFERENCES textbooks(id)
                    ON DELETE CASCADE, textbook_title TEXT NOT NULL, chapter TEXT NOT NULL,
                    page INTEGER NOT NULL CHECK(page >= 1), text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_textbook ON chunks(textbook_id);
                CREATE TABLE IF NOT EXISTS corpus_state (
                    id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO corpus_state VALUES (1, 0);
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    fingerprint TEXT PRIMARY KEY, vectors BLOB NOT NULL
                );
                PRAGMA user_version=1;
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def replace_textbook(self, textbook: Textbook, chunks: list[Chunk]) -> None:
        if not chunks or any(chunk.textbook_id != textbook.id for chunk in chunks):
            raise ValueError("Textbook must have chunks with matching textbook IDs")
        # DELETE and replacement commit together; failed inserts roll back the old corpus.
        with self._connection() as connection:
            connection.execute("DELETE FROM textbooks WHERE id=?", (textbook.id,))
            connection.execute(
                "INSERT INTO textbooks VALUES (:id,:title,:source_file,:sha256,:edition,:year)",
                asdict(textbook),
            )
            connection.executemany(
                "INSERT INTO chunks VALUES (:id,:textbook_id,:textbook_title,:chapter,:page,:text)",
                [asdict(chunk) for chunk in chunks],
            )
            connection.execute("UPDATE corpus_state SET revision=revision+1 WHERE id=1")
            connection.execute("DELETE FROM embedding_cache")

    def delete_textbook(self, textbook_id: str) -> bool:
        validate_textbook_id(textbook_id)
        with self._connection() as connection:
            deleted = (
                connection.execute("DELETE FROM textbooks WHERE id=?", (textbook_id,)).rowcount > 0
            )
            if deleted:
                connection.execute("UPDATE corpus_state SET revision=revision+1 WHERE id=1")
                connection.execute("DELETE FROM embedding_cache")
            return deleted

    def list_textbooks(self) -> list[dict]:
        with self._connection() as connection:
            return [
                dict(row)
                for row in connection.execute("""
                SELECT t.*, COUNT(c.id) AS chunk_count FROM textbooks t
                LEFT JOIN chunks c ON c.textbook_id=t.id GROUP BY t.id ORDER BY t.id
            """)
            ]

    def revision(self) -> int:
        with self._connection() as connection:
            return connection.execute("SELECT revision FROM corpus_state WHERE id=1").fetchone()[0]

    def snapshot(self) -> tuple[int, list[Chunk]]:
        with self._connection() as connection:
            connection.execute("BEGIN")
            revision = connection.execute(
                "SELECT revision FROM corpus_state WHERE id=1"
            ).fetchone()[0]
            chunks = [
                Chunk(**dict(row))
                for row in connection.execute("SELECT * FROM chunks ORDER BY textbook_id, page, id")
            ]
            return revision, chunks

    def load_vectors(self, fingerprint: str) -> np.ndarray | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT vectors FROM embedding_cache WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
        if row is None:
            return None
        try:
            vectors = np.load(io.BytesIO(row[0]), allow_pickle=False)
            if not isinstance(vectors, np.ndarray) or vectors.dtype != np.float32:
                return None
            return vectors
        except (ValueError, OSError, EOFError):
            return None

    def save_vectors(self, fingerprint: str, vectors: np.ndarray, revision: int) -> None:
        buffer = io.BytesIO()
        np.save(buffer, vectors, allow_pickle=False)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT revision FROM corpus_state WHERE id=1").fetchone()[
                0
            ]
            # A concurrent ingestion may have invalidated this snapshot while encoding.
            if current == revision:
                connection.execute(
                    "INSERT OR REPLACE INTO embedding_cache VALUES (?,?)",
                    (fingerprint, buffer.getvalue()),
                )
