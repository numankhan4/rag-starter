import pytest

from app.chunking import chunk_text


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
