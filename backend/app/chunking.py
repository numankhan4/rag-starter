"""Split documents into overlapping, word-based chunks with source spans."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"\S+")
_HEADING_TITLE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str
    start: int = 0
    end: int = 0
    heading: str | None = None


def _headings(text: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group(1)) for m in _HEADING_TITLE.finditer(text)]


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
    headings = _headings(text)
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
    return _with_spans(doc_id, text, texts)


def _with_spans(doc_id: str, text: str, pieces: list[str]) -> list[Chunk]:
    """Build chunks for `pieces` (in document order), locating each piece's words in `text`."""
    words = list(_WORD.finditer(text))
    tokens = [w.group() for w in words]
    headings = _headings(text)
    chunks: list[Chunk] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        target = piece.split()
        n = len(target)
        j = cursor
        while j + n <= len(tokens) and tokens[j : j + n] != target:
            j += 1
        if j + n > len(tokens):  # not found; should not happen, keep the text without a span
            chunks.append(Chunk(doc_id=doc_id, index=i, text=piece))
            continue
        start, end = words[j].start(), words[j + n - 1].end()
        chunks.append(
            Chunk(
                doc_id=doc_id,
                index=i,
                text=piece,
                start=start,
                end=end,
                heading=_heading_at(headings, start),
            )
        )
        cursor = j + 1
    return chunks


def chunk_document(strategy: str, doc_id: str, text: str, size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Chunk `text` with the named strategy ("words" or "recursive")."""
    if strategy == "recursive":
        return chunk_recursive(doc_id, text, size, overlap)
    if strategy == "words":
        return chunk_text(doc_id, text, size, overlap)
    raise ValueError(f"unknown chunker {strategy!r}")
