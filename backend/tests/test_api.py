from fastapi.testclient import TestClient

from app.main import create_app


def client():
    return TestClient(create_app())


def test_query_modes():
    c = client()
    c.post("/ingest", json={"doc_id": "a", "text": "pgvector adds vector search to Postgres."})
    for mode in ("vector", "keyword", "hybrid"):
        r = c.post("/query", json={"question": "pgvector", "mode": mode})
        assert r.status_code == 200 and r.json()["sources"][0]["doc_id"] == "a"
    assert c.post("/query", json={"question": "x", "mode": "bogus"}).status_code == 422


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


def test_error_shapes_404_422_500():
    app = create_app()

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret")

    c = TestClient(app, raise_server_exceptions=False)

    r = c.delete("/documents/nope")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "not_found", "message": "Unknown document"}}

    r = c.get("/missing-route")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"

    r = c.post("/query", json={"question": "", "k": 4})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"
    assert "question" in body["error"]["message"]

    r = c.get("/boom")
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "Internal server error"}}


def test_openapi_documents_errors():
    schema = client().get("/openapi.json").json()
    responses = schema["paths"]["/query"]["post"]["responses"]
    assert {"404", "422", "500"} <= set(responses)
    assert "ErrorResponse" in schema["components"]["schemas"]


def _two_docs():
    c = client()
    for doc_id, title in (("a", "Alpha Guide"), ("b", "Beta Notes")):
        body = {"doc_id": doc_id, "title": title, "text": "Retrieval grounds answers in documents."}
        c.post("/ingest", json=body)
    return c


def _ids(c, **filters):
    r = c.post("/query", json={"question": "What grounds answers?", **filters})
    assert r.status_code == 200
    return {s["doc_id"] for s in r.json()["sources"]}


def test_query_filter_by_doc_ids():
    c = _two_docs()
    assert _ids(c) == {"a", "b"}
    assert _ids(c, doc_ids=["b"]) == {"b"}
    assert _ids(c, doc_ids=["nope"]) == set()


def test_query_filter_by_title_substring():
    c = _two_docs()
    assert _ids(c, title_contains="alpha") == {"a"}
    assert _ids(c, title_contains="zzz") == set()


def test_query_filters_combine():
    c = _two_docs()
    assert _ids(c, doc_ids=["a", "b"], title_contains="beta") == {"b"}
    assert _ids(c, doc_ids=["a"], title_contains="beta") == set()
