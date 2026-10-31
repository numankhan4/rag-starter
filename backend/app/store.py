"""A small in-memory vector store with cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._items.extend(zip(chunks, vectors, strict=True))

    def delete_document(self, doc_id: str) -> int:
        before = len(self._items)
        self._items = [(c, v) for c, v in self._items if c.doc_id != doc_id]
        return before - len(self._items)

    def search(self, query: list[float], k: int = 4, min_score: float = 0.0) -> list[Hit]:
        """Return the top-k chunks by cosine similarity (vectors are assumed L2-normalised)."""
        scored = [Hit(c, sum(a * b for a, b in zip(query, v, strict=True))) for c, v in self._items]
        scored = [h for h in scored if h.score > min_score]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
