from fastapi.testclient import TestClient

from app.main import create_app


def client():
    return TestClient(create_app())


def test_health():
    c = client()
    assert c.get("/health").json() == {"status": "ok", "chunks": 0, "documents": 0}


def test_ingest_query_and_delete():
    c = client()
    r = c.post(
        "/ingest",
        json={
            "doc_id": "guide",
            "title": "Guide",
            "text": "Retrieval augmented generation grounds answers in your documents.",
        },
    )
    assert r.status_code == 200 and r.json()["chunks"] == 1
    q = c.post("/query", json={"question": "What grounds answers in documents?"}).json()
    assert q["sources"][0]["doc_id"] == "guide"
    assert q["sources"][0]["title"] == "Guide"
    assert 1 in q["citations"]
    assert c.delete("/documents/guide").json() == {"removed_chunks": 1}
    assert c.delete("/documents/guide").status_code == 404


def test_reingest_replaces_document():
    c = client()
    c.post("/ingest", json={"doc_id": "d", "text": "first version"})
    c.post("/ingest", json={"doc_id": "d", "text": "second version"})
    assert c.get("/health").json()["chunks"] == 1


def test_validation():
    c = client()
    assert c.post("/ingest", json={"doc_id": "", "text": "x"}).status_code == 422
    assert c.post("/query", json={"question": "x", "k": 0}).status_code == 422
