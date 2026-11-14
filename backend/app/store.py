"""A small in-memory vector store with cosine similarity."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from .bm25 import BM25Index
from .chunking import Chunk

RRF_K = 60


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []
        self._bm25: BM25Index | None = None

    def __len__(self) -> int:
        return len(self._items)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._bm25 = None
        self._items.extend(zip(chunks, vectors, strict=True))

    def delete_document(self, doc_id: str) -> int:
        before = len(self._items)
        self._bm25 = None
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

    def keyword_search(self, query: str, k: int = 4, doc_ids: Collection[str] | None = None) -> list[Hit]:
        """Return the top-k chunks by BM25 score, optionally limited to `doc_ids`."""
        if self._bm25 is None:
            self._bm25 = BM25Index([c for c, _ in self._items])
        return [Hit(c, s) for c, s in self._bm25.search(query, k, doc_ids)]

    def hybrid_search(
        self,
        text: str,
        query: list[float],
        k: int = 4,
        doc_ids: Collection[str] | None = None,
    ) -> list[Hit]:
        """Merge vector and BM25 rankings with reciprocal rank fusion."""
        pool = max(k * 3, 20)
        fused: dict[tuple[str, int], float] = {}
        chunks: dict[tuple[str, int], Chunk] = {}
        for ranking in (
            self.search(query, k=pool, doc_ids=doc_ids),
            self.keyword_search(text, k=pool, doc_ids=doc_ids),
        ):
            for rank, h in enumerate(ranking, start=1):
                key = (h.chunk.doc_id, h.chunk.index)
                chunks[key] = h.chunk
                fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(fused, key=lambda key: fused[key], reverse=True)[:k]
        return [Hit(chunks[key], fused[key]) for key in top]
