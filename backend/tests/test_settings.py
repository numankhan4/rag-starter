import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings


def test_defaults():
    s = Settings.from_env({})
    assert s == Settings()
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (120, 20, 4)
    assert s.cors_origins == ("http://localhost:3000",)


def test_env_overrides():
    s = Settings.from_env(
        {
            "CORS_ORIGINS": "http://a.test, http://b.test",
            "CHUNK_SIZE": "10",
            "CHUNK_OVERLAP": "2",
            "TOP_K": "7",
            "EMBEDDER": "Hashing",
            "ANSWERER": "extractive",
        }
    )
    assert s.cors_origins == ("http://a.test", "http://b.test")
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (10, 2, 7)


@pytest.mark.parametrize(
    "env",
    [
        {"CHUNK_SIZE": "abc"},
        {"CHUNK_SIZE": "0"},
        {"CHUNK_SIZE": "5", "CHUNK_OVERLAP": "5"},
        {"TOP_K": "0"},
        {"TOP_K": "21"},
        {"EMBEDDER": "nope"},
        {"ANSWERER": "nope"},
    ],
)
def test_invalid_values(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_chunker_setting():
    assert Settings.from_env({}).chunker == "words"
    assert Settings.from_env({"CHUNKER": "Recursive"}).chunker == "recursive"
    with pytest.raises(ValueError):
        Settings.from_env({"CHUNKER": "x"})


def test_recursive_chunker_applies_to_ingest():
    c = TestClient(
        create_app(Settings.from_env({"CHUNKER": "recursive", "CHUNK_SIZE": "5", "CHUNK_OVERLAP": "0"}))
    )
    r = c.post("/ingest", json={"doc_id": "d", "text": "# A\n\none two\n\n# B\n\nthree four"})
    assert r.json()["chunks"] == 2


def test_chunk_size_applies_to_ingest():
    c = TestClient(create_app(Settings.from_env({"CHUNK_SIZE": "4", "CHUNK_OVERLAP": "0"})))
    r = c.post("/ingest", json={"doc_id": "d", "text": "one two three four five six seven eight nine"})
    assert r.json()["chunks"] == 3


def test_top_k_default_applies_to_query():
    c = TestClient(create_app(Settings.from_env({"CHUNK_SIZE": "2", "CHUNK_OVERLAP": "0", "TOP_K": "2"})))
    c.post("/ingest", json={"doc_id": "d", "text": "alpha beta alpha gamma alpha delta"})
    assert len(c.post("/query", json={"question": "alpha"}).json()["sources"]) == 2
    assert len(c.post("/query", json={"question": "alpha", "k": 1}).json()["sources"]) == 1


def test_cors_origin_from_env():
    c = TestClient(create_app(Settings.from_env({"CORS_ORIGINS": "http://x.test"})))
    r = c.get("/health", headers={"Origin": "http://x.test"})
    assert r.headers["access-control-allow-origin"] == "http://x.test"


def test_store_setting_and_sqlite_persistence(tmp_path):
    assert Settings.from_env({}).store == "memory"
    with pytest.raises(ValueError):
        Settings.from_env({"STORE": "x"})
    env = {"STORE": "sqlite", "SQLITE_PATH": str(tmp_path / "api.db")}
    c = TestClient(create_app(Settings.from_env(env)))
    c.post("/ingest", json={"doc_id": "d", "title": "T", "text": "alpha beta gamma"})
    c2 = TestClient(create_app(Settings.from_env(env)))
    assert c2.get("/health").json()["documents"] == 1
    assert c2.post("/query", json={"question": "alpha"}).json()["sources"][0]["title"] == "T"
