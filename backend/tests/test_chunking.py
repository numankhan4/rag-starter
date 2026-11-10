import re

import pytest

from app.chunking import chunk_document, chunk_recursive, chunk_text


def test_offsets_map_back_to_original_text_and_headings():
    text = "# Intro\nalpha  beta\n\ngamma delta\n## Usage\nepsilon zeta eta theta"
    chunks = chunk_text("d", text, size=4, overlap=1)
    assert len(chunks) > 1
    for c in chunks:
        assert re.sub(r"\s+", " ", text[c.start : c.end]) == c.text
    assert chunks[0].heading == "Intro"
    assert chunks[-1].heading == "Usage"


def test_no_heading_is_none():
    assert chunk_text("d", "just plain words")[0].heading is None


def test_chunks_overlap_and_cover_all_words():
    words = [f"w{i}" for i in range(50)]
    chunks = chunk_text("d", " ".join(words), size=20, overlap=5)
    assert [c.index for c in chunks] == [0, 1, 2]
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]
    assert chunks[-1].text.split()[-1] == "w49"


def test_empty_text_has_no_chunks():
    assert chunk_text("d", "   ") == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_sizes(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("d", "a b c", size=size, overlap=overlap)


MD = (
    "# Title\n\nIntro paragraph with a few words.\n\n## Section A\n\n"
    + " ".join(f"a{i}" for i in range(30))
    + "\n\nSecond paragraph of A. It has two sentences.\n\n## Section B\n\n"
    + " ".join(f"Sentence {i} here." for i in range(20))
    + "\n\n## Section C\n\n"
    + " ".join(f"z{i}" for i in range(100))
)


def test_recursive_never_exceeds_size_and_keeps_headings_apart():
    chunks = chunk_recursive("d", MD, size=25, overlap=5)
    assert all(len(c.text.split()) <= 25 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].text.startswith("# Title")
    assert any(c.text.startswith("## Section B") for c in chunks)
    assert all(c.text.count("## ") <= 1 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert "a29" in joined and "z99" in joined and "Sentence 19 here." in joined


def test_recursive_small_sections_are_single_chunks():
    chunks = chunk_recursive("d", "# A\n\nhello world\n\n# B\n\nbye", size=50, overlap=0)
    assert [c.text for c in chunks] == ["# A\n\nhello world", "# B\n\nbye"]


def test_recursive_empty_and_invalid():
    assert chunk_recursive("d", "  \n ") == []
    with pytest.raises(ValueError):
        chunk_recursive("d", "a", size=5, overlap=5)


def test_chunk_document_dispatch():
    assert chunk_document("words", "d", "a b c") == chunk_text("d", "a b c")
    with pytest.raises(ValueError):
        chunk_document("nope", "d", "a")


def test_recursive_chunks_have_spans_and_headings():
    text = "# Intro\n\nHello  there world.\n\n## Setup\n\nInstall it. Then run it."
    chunks = chunk_recursive("d", text, size=50, overlap=0)
    assert [c.heading for c in chunks] == ["Intro", "Setup"]
    for c in chunks:
        assert " ".join(text[c.start : c.end].split()) == " ".join(c.text.split())
