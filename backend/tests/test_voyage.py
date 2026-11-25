import httpx
import pytest

from app.embeddings import HashingEmbedder, VoyageEmbedder, build_embedder
from app.settings import Settings


def _client(calls: list[httpx.Request]) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        import json

        texts = json.loads(req.content)["input"]
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(texts)]
        return httpx.Response(200, json={"data": list(reversed(data))})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_voyage_embed_orders_and_auth():
    calls: list[httpx.Request] = []
    emb = VoyageEmbedder("k", client=_client(calls))
    assert emb.embed(["a", "bbb"]) == [[1.0, 1.0], [3.0, 1.0]]
    assert emb.dim == 2
    assert calls[0].headers["authorization"] == "Bearer k"


def test_voyage_batches():
    calls: list[httpx.Request] = []
    emb = VoyageEmbedder("k", client=_client(calls))
    assert len(emb.embed(["x"] * 130)) == 130
    assert len(calls) == 2


def test_voyage_http_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(httpx.HTTPStatusError):
        VoyageEmbedder("k", client=client).embed(["a"])


def test_selection():
    assert Settings.from_env({}).embedder == "hashing"
    assert Settings.from_env({"VOYAGE_API_KEY": "k"}).embedder == "voyage"
    assert Settings.from_env({"VOYAGE_API_KEY": "k", "EMBEDDER": "hashing"}).embedder == "hashing"
    with pytest.raises(ValueError):
        Settings.from_env({"EMBEDDER": "voyage"})
    assert isinstance(build_embedder("hashing", {}), HashingEmbedder)
    assert isinstance(build_embedder("voyage", {"VOYAGE_API_KEY": "k"}), VoyageEmbedder)
