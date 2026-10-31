from app.answer import ExtractiveAnswerer
from app.chunking import chunk_text
from app.embeddings import HashingEmbedder
from app.store import InMemoryVectorStore

DOCS = {
    "fastapi": "FastAPI is a Python web framework for building APIs. It uses type hints and Pydantic.",
    "nextjs": "Next.js is a React framework for the web. It supports server components and static export.",
    "pgvector": "pgvector adds vector similarity search to Postgres for embeddings.",
}


def build():
    emb, store = HashingEmbedder(), InMemoryVectorStore()
    for doc_id, text in DOCS.items():
        chunks = chunk_text(doc_id, text, size=40, overlap=5)
        store.add(chunks, emb.embed([c.text for c in chunks]))
    return emb, store


def test_embeddings_are_normalised_and_deterministic():
    emb = HashingEmbedder(dim=64)
    a, b = emb.embed(["Hello world", "Hello world"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_search_ranks_the_relevant_document_first():
    emb, store = build()
    hits = store.search(emb.embed(["Which Python framework uses Pydantic type hints?"])[0], k=2)
    assert hits[0].chunk.doc_id == "fastapi"


def test_delete_document():
    _, store = build()
    assert store.delete_document("nextjs") == 1
    assert len(store) == 2


def test_extractive_answer_cites_sources():
    emb, store = build()
    q = "What does pgvector add to Postgres?"
    ans = ExtractiveAnswerer().answer(q, store.search(emb.embed([q])[0], k=2))
    assert "pgvector" in ans.text
    assert ans.citations and ans.citations[0] == 1


def test_answer_without_hits():
    ans = ExtractiveAnswerer().answer("anything", [])
    assert ans.citations == []
