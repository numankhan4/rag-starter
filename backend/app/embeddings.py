"""Embedding providers.

`HashingEmbedder` is deterministic and dependency-free: a hashed bag-of-words vector
(feature hashing, L2-normalised). It is good enough for tests, demos and offline use,
and keeps the starter runnable without API keys. Real providers plug in through the
`Embedder` protocol.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(token.encode(), digest_size=8).digest()
        n = int.from_bytes(h, "big")
        return n % self.dim, (1.0 if (n >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in tokenize(text):
                i, sign = self._bucket(tok)
                vec[i] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
