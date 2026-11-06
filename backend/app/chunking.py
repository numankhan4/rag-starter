"""Split documents into overlapping, word-based chunks with source spans."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"\S+")
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str
    start: int = 0
    end: int = 0
    heading: str | None = None


def _heading_at(headings: list[tuple[int, str]], pos: int) -> str | None:
    """Return the nearest Markdown heading that starts at or before `pos`."""
    found = None
    for hpos, title in headings:
        if hpos > pos:
            break
        found = title
    return found


def chunk_text(doc_id: str, text: str, size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Split `text` into chunks of `size` words, each overlapping the previous by `overlap` words.

    `start`/`end` are character offsets into `text` (``text[start:end]`` is the chunk's source
    span; chunk text has whitespace collapsed to single spaces).
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    words = list(_WORD.finditer(text))
    if not words:
        return []
    headings = [(m.start(), m.group(1)) for m in _HEADING.finditer(text)]
    step = size - overlap
    chunks: list[Chunk] = []
    for i, first in enumerate(range(0, len(words), step)):
        window = words[first : first + size]
        start, end = window[0].start(), window[-1].end()
        chunks.append(
            Chunk(
                doc_id=doc_id,
                index=i,
                text=" ".join(w.group() for w in window),
                start=start,
                end=end,
                heading=_heading_at(headings, start),
            )
        )
        if first + size >= len(words):
            break
    return chunks
