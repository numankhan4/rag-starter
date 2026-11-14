"""A small in-memory vector store with cosine similarity."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol

from .chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def __len__(self) -> int: ...
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def delete_document(self, doc_id: str) -> int: ...
    def search(
        self,
        query: list[float],
        k: int = 4,
        min_score: float = 0.0,
        doc_ids: Collection[str] | None = None,
    ) -> list[Hit]: ...
    def set_title(self, doc_id: str, title: str | None) -> None: ...
    def title(self, doc_id: str) -> str | None: ...
    def document_count(self) -> int: ...
    def doc_ids_with_title(self, needle: str) -> set[str]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []
        self._titles: dict[str, str | None] = {}

    def set_title(self, doc_id: str, title: str | None) -> None:
        self._titles[doc_id] = title

    def title(self, doc_id: str) -> str | None:
        return self._titles.get(doc_id)

    def doc_ids_with_title(self, needle: str) -> set[str]:
        """Doc ids whose title contains `needle` (case-insensitive)."""
        needle = needle.lower()
        return {d for d, t in self._titles.items() if t and needle in t.lower()}

    def document_count(self) -> int:
        return len({c.doc_id for c, _ in self._items})

    def __len__(self) -> int:
        return len(self._items)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._items.extend(zip(chunks, vectors, strict=True))

    def delete_document(self, doc_id: str) -> int:
        self._titles.pop(doc_id, None)
        before = len(self._items)
        self._items = [(c, v) for c, v in self._items if c.doc_id != doc_id]
        return before - len(self._items)

    def search(
        self,
        query: list[float],
        k: int = 4,
        min_score: float = 0.0,
        doc_ids: Collection[str] | None = None,
    ) -> list[Hit]:
        """Return the top-k chunks by cosine similarity (vectors are assumed L2-normalised).

        If `doc_ids` is given, only chunks of those documents are considered.
        """
        scored = [
            Hit(c, sum(a * b for a, b in zip(query, v, strict=True)))
            for c, v in self._items
            if doc_ids is None or c.doc_id in doc_ids
        ]
        scored = [h for h in scored if h.score > min_score]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


class SQLiteVectorStore:
    """Persistent store: chunks and JSON-encoded vectors in SQLite, brute-force cosine search."""

    def __init__(self, path: str) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                text TEXT NOT NULL,
                start INTEGER NOT NULL,
                "end" INTEGER NOT NULL,
                heading TEXT,
                vector TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id);
            CREATE TABLE IF NOT EXISTS titles (doc_id TEXT PRIMARY KEY, title TEXT);
            """
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._db:
            self._db.executemany(
                'INSERT INTO chunks (doc_id, idx, text, start, "end", heading, vector)'
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.doc_id, c.index, c.text, c.start, c.end, c.heading, json.dumps(v))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def delete_document(self, doc_id: str) -> int:
        with self._db:
            self._db.execute("DELETE FROM titles WHERE doc_id = ?", (doc_id,))
            return self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,)).rowcount

    def set_title(self, doc_id: str, title: str | None) -> None:
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO titles VALUES (?, ?)", (doc_id, title))

    def title(self, doc_id: str) -> str | None:
        row = self._db.execute("SELECT title FROM titles WHERE doc_id = ?", (doc_id,)).fetchone()
        return row[0] if row else None

    def doc_ids_with_title(self, needle: str) -> set[str]:
        """Doc ids whose title contains `needle` (case-insensitive)."""
        needle = needle.lower()
        rows = self._db.execute("SELECT doc_id, title FROM titles").fetchall()
        return {d for d, t in rows if t and needle in t.lower()}

    def document_count(self) -> int:
        return self._db.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks").fetchone()[0]

    def search(
        self,
        query: list[float],
        k: int = 4,
        min_score: float = 0.0,
        doc_ids: Collection[str] | None = None,
    ) -> list[Hit]:
        """Top-k by cosine similarity, computed over every stored vector.

        If `doc_ids` is given, only chunks of those documents are considered.
        """
        qnorm = math.sqrt(sum(a * a for a in query)) or 1.0
        hits: list[Hit] = []
        rows = self._db.execute(
            'SELECT doc_id, idx, text, start, "end", heading, vector FROM chunks ORDER BY id'
        )
        for doc_id, idx, text, start, end, heading, raw in rows:
            if doc_ids is not None and doc_id not in doc_ids:
                continue
            vec = json.loads(raw)
            norm = math.sqrt(sum(b * b for b in vec)) or 1.0
            score = sum(a * b for a, b in zip(query, vec, strict=True)) / (qnorm * norm)
            if score > min_score:
                hits.append(Hit(Chunk(doc_id, idx, text, start, end, heading), score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def create_store(kind: str, path: str) -> VectorStore:
    return SQLiteVectorStore(path) if kind == "sqlite" else InMemoryVectorStore()
