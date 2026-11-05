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


_HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> int:
    return len(text.split())


def _split_sections(text: str) -> list[str]:
    """Split before each Markdown heading line."""
    starts = [m.start() for m in _HEADING.finditer(text)]
    bounds = sorted({0, *starts, len(text)})
    sections = (text[a:b].strip() for a, b in zip(bounds, bounds[1:], strict=False))
    return [s for s in sections if s]


def _pack(units: list[str], size: int, sep: str) -> list[str]:
    """Greedily merge units (each <= size words) into pieces of at most `size` words."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        n = _words(unit)
        if current and count + n > size:
            pieces.append(sep.join(current))
            current, count = [], 0
        current.append(unit)
        count += n
    if current:
        pieces.append(sep.join(current))
    return pieces


def _word_windows(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    for start in range(0, len(words), size - overlap):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


def _split_piece(text: str, size: int, overlap: int, level: int = 0) -> list[str]:
    """Recursively split `text` on paragraphs, then sentences, then word windows."""
    if _words(text) <= size:
        return [text]
    if level >= 2:
        return _word_windows(text, size, overlap)
    pattern, sep = (_PARAGRAPH, "\n\n") if level == 0 else (_SENTENCE, " ")
    units: list[str] = []
    for part in pattern.split(text):
        if part.strip():
            units.extend(_split_piece(part.strip(), size, overlap, level + 1))
    return _pack(units, size, sep)


def chunk_recursive(doc_id: str, text: str, size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Split on headings, then paragraphs, then sentences, then word windows.

    `size` is the maximum number of words per chunk; chunks never span headings.
    `overlap` only applies to the word-window fallback for over-long sentences.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    texts = [p for sec in _split_sections(text) for p in _split_piece(sec, size, overlap)]
    return [Chunk(doc_id=doc_id, index=i, text=t) for i, t in enumerate(texts)]


def chunk_document(strategy: str, doc_id: str, text: str, size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Chunk `text` with the named strategy ("words" or "recursive")."""
    if strategy == "recursive":
        return chunk_recursive(doc_id, text, size, overlap)
    if strategy == "words":
        return chunk_text(doc_id, text, size, overlap)
    raise ValueError(f"unknown chunker {strategy!r}")
