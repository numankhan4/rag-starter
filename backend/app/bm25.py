"""A small BM25 keyword index (Okapi BM25) over chunks."""

from __future__ import annotations

import math
from collections import Counter

from .chunking import Chunk
from .embeddings import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = chunks
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(chunks)) if chunks else 0.0
        self._df: Counter[str] = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())

    def _idf(self, term: str) -> float:
        n, df = len(self._chunks), self._df[term]
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Top-k chunks with a positive BM25 score, best first."""
        terms = set(tokenize(query))
        scored: list[tuple[Chunk, float]] = []
        for chunk, tf, length in zip(self._chunks, self._tf, self._len, strict=True):
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = 1 - self._b + self._b * length / (self._avg or 1.0)
                    score += self._idf(t) * f * (self._k1 + 1) / (f + self._k1 * norm)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:k]
