"""Embedding providers.

`HashingEmbedder` is deterministic and dependency-free: a hashed bag-of-words vector
(feature hashing, L2-normalised). It is good enough for tests, demos and offline use,
and keeps the starter runnable without API keys. Real providers plug in through the
`Embedder` protocol.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping
from typing import Protocol

import httpx

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


class VoyageEmbedder:
    """Voyage AI embeddings over HTTP (https://docs.voyageai.com/reference/embeddings-api)."""

    URL = "https://api.voyageai.com/v1/embeddings"
    BATCH = 128

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3.5",
        dim: int = 1024,
        client: httpx.Client | None = None,
    ) -> None:
        self.dim = dim
        self._model = model
        self._key = api_key
        self._client = client or httpx.Client(timeout=30.0)

    def _post(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post(
            self.URL,
            headers={"Authorization": f"Bearer {self._key}"},
            json={"input": texts, "model": self._model},
        )
        resp.raise_for_status()
        rows = sorted(resp.json()["data"], key=lambda r: r["index"])
        return [r["embedding"] for r in rows]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            out.extend(self._post(texts[i : i + self.BATCH]))
        if out:
            self.dim = len(out[0])
        return out


def build_embedder(name: str, env: Mapping[str, str] | None = None) -> Embedder:
    """Pick an embedder by name; `voyage` needs VOYAGE_API_KEY (optional VOYAGE_MODEL)."""
    env = os.environ if env is None else env
    if name == "voyage":
        return VoyageEmbedder(env["VOYAGE_API_KEY"].strip(), env.get("VOYAGE_MODEL") or "voyage-3.5")
    return HashingEmbedder()
