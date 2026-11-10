import pytest

from app.answer import ExtractiveAnswerer
from app.chunking import chunk_text
from app.embeddings import HashingEmbedder
from app.store import InMemoryVectorStore, SQLiteVectorStore

DOCS = {
    "fastapi": "FastAPI is a Python web framework for building APIs. It uses type hints and Pydantic.",
    "nextjs": "Next.js is a React framework for the web. It supports server components and static export.",
    "pgvector": "pgvector adds vector similarity search to Postgres for embeddings.",
}


@pytest.fixture(params=["memory", "sqlite"])
def make_store(request, tmp_path):
    return lambda: (
        InMemoryVectorStore() if request.param == "memory" else SQLiteVectorStore(str(tmp_path / "t.db"))
    )


def build(store=None):
    emb, store = HashingEmbedder(), InMemoryVectorStore() if store is None else store
    for doc_id, text in DOCS.items():
        chunks = chunk_text(doc_id, text, size=40, overlap=5)
        store.add(chunks, emb.embed([c.text for c in chunks]))
    return emb, store


def test_embeddings_are_normalised_and_deterministic():
    emb = HashingEmbedder(dim=64)
    a, b = emb.embed(["Hello world", "Hello world"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_search_ranks_the_relevant_document_first(make_store):
    emb, store = build(make_store())
    hits = store.search(emb.embed(["Which Python framework uses Pydantic type hints?"])[0], k=2)
    assert hits[0].chunk.doc_id == "fastapi"


def test_delete_document(make_store):
    _, store = build(make_store())
    assert store.delete_document("nextjs") == 1
    assert len(store) == 2


def test_extractive_answer_cites_sources(make_store):
    emb, store = build(make_store())
    q = "What does pgvector add to Postgres?"
    ans = ExtractiveAnswerer().answer(q, store.search(emb.embed([q])[0], k=2))
    assert "pgvector" in ans.text
    assert ans.citations and ans.citations[0] == 1


def test_answer_without_hits():
    ans = ExtractiveAnswerer().answer("anything", [])
    assert ans.citations == []


def test_sqlite_store_persists_across_reopen(tmp_path):
    path = str(tmp_path / "p.db")
    emb, store = build(SQLiteVectorStore(path))
    store.set_title("fastapi", "FastAPI")
    store.close()
    reopened = SQLiteVectorStore(path)
    assert len(reopened) == 3 and reopened.document_count() == 3
    assert reopened.title("fastapi") == "FastAPI"
    hits = reopened.search(emb.embed(["Python framework Pydantic"])[0], k=1)
    assert hits[0].chunk.doc_id == "fastapi"
