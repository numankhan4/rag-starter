"""Turn markdown / HTML / plain text into clean text plus metadata (title, headings)."""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser

_SKIP_TAGS = {"script", "style", "noscript", "template", "head"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
    "header", "footer", "pre", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
}  # fmt: skip
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


@dataclass
class Loaded:
    text: str
    title: str | None = None
    headings: list[str] = field(default_factory=list)


def _squash(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.headings: list[str] = []
        self.title: str | None = None
        self._skip = 0
        self._in_title = False
        self._title_buf: list[str] = []
        self._heading_buf: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in _SKIP_TAGS:
            if tag != "head":
                self._skip += 1
        elif tag in _HEADINGS:
            self._heading_buf = []
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            self.title = _squash(" ".join(self._title_buf)) or None
        elif tag in _SKIP_TAGS:
            if tag != "head":
                self._skip = max(0, self._skip - 1)
        elif tag in _HEADINGS and self._heading_buf is not None:
            heading = " ".join("".join(self._heading_buf).split())
            if heading:
                self.headings.append(heading)
            self._heading_buf = None
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title or self._skip:
            if self._in_title:
                self._title_buf.append(data)
            return
        self.parts.append(data)
        if self._heading_buf is not None:
            self._heading_buf.append(data)


def load_html(html: str) -> Loaded:
    """Strip tags (and script/style content); title from <title>, falling back to the first heading."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    title = parser.title or (parser.headings[0] if parser.headings else None)
    return Loaded(text=_squash("".join(parser.parts)), title=title, headings=parser.headings)


_FENCE = re.compile(r"^\s*(```|~~~)")
_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")


def _inline(line: str) -> str:
    line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)  # images -> alt text
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # links -> label
    line = re.sub(r"`([^`]*)`", r"\1", line)
    line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)
    return re.sub(r"(?<![\w*])([*_])(.+?)\1(?![\w*])", r"\2", line)


def load_markdown(md: str) -> Loaded:
    """Strip common markdown syntax; title is the first H1 (or else the first heading)."""
    out: list[str] = []
    headings: list[tuple[int, str]] = []
    in_fence = False
    for raw in md.splitlines():
        if _FENCE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            out.append(raw)
            continue
        m = _ATX.match(raw)
        if m:
            heading = _inline(m.group(2)).strip()
            if heading:
                headings.append((len(m.group(1)), heading))
                out.append(heading)
            continue
        line = re.sub(r"^\s*>+\s?", "", raw)
        line = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", line)
        if re.fullmatch(r"\s*([-*_]\s*){3,}", line):
            continue
        out.append(_inline(line))
    title = next((h for lvl, h in headings if lvl == 1), None)
    if title is None and headings:
        title = headings[0][1]
    return Loaded(text=_squash("\n".join(out)), title=title, headings=[h for _, h in headings])


def load_text(text: str) -> Loaded:
    return Loaded(text=text.strip())


class PdfUnavailableError(RuntimeError):
    """Raised when a PDF is uploaded but the optional `pypdf` package is not installed."""


class PdfReadError(ValueError):
    """Raised when the uploaded bytes cannot be parsed as a PDF."""


def load_pdf(data: bytes) -> Loaded:
    """Extract text from a PDF (requires the optional `pypdf` dependency)."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError:
        raise PdfUnavailableError("PDF support requires pypdf (pip install pypdf)") from None
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        info_title = reader.metadata.title if reader.metadata else None
    except (PyPdfError, ValueError, OSError):
        raise PdfReadError("Could not read PDF") from None
    title = _squash(info_title) if isinstance(info_title, str) else None
    return Loaded(text=_squash("\n\n".join(pages)), title=title or None)


LOADERS: dict[str, Callable[[str], Loaded]] = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".txt": load_text,
}
