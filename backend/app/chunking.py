"""Split documents into overlapping, word-based chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str


def chunk_text(doc_id: str, text: str, size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Split `text` into chunks of `size` words, each overlapping the previous by `overlap` words."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    words = _WS.split(text.strip())
    if words == [""]:
        return []
    step = size - overlap
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        chunks.append(Chunk(doc_id=doc_id, index=i, text=" ".join(window)))
        if start + size >= len(words):
            break
    return chunks
