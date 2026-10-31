"""Answer generation.

`ExtractiveAnswerer` needs no LLM: it returns the most relevant sentences from the
retrieved chunks with numbered citations. LLM-backed answerers (e.g. Claude) implement
the same `Answerer` protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .embeddings import tokenize
from .store import Hit

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[int]


class Answerer(Protocol):
    def answer(self, question: str, hits: list[Hit]) -> Answer: ...


class ExtractiveAnswerer:
    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer("I couldn't find anything relevant in the ingested documents.", [])
        q = set(tokenize(question))
        candidates: list[tuple[float, int, str]] = []
        for n, hit in enumerate(hits, start=1):
            for sentence in _SENTENCE.split(hit.chunk.text):
                overlap = len(q & set(tokenize(sentence)))
                if overlap:
                    candidates.append((overlap + hit.score, n, sentence.strip()))
        if not candidates:
            first = hits[0].chunk.text.split(". ")[0].strip()
            return Answer(f"{first} [1]", [1])
        candidates.sort(key=lambda c: c[0], reverse=True)
        chosen = candidates[: self.max_sentences]
        text = " ".join(f"{s} [{n}]" for _, n, s in chosen)
        return Answer(text, sorted({n for _, n, _ in chosen}))
