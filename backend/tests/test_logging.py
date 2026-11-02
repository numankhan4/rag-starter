import json
import logging

from fastapi.testclient import TestClient

from app.logging_setup import JsonFormatter, request_id_var
from app.main import create_app


def test_request_id_round_trip() -> None:
    client = TestClient(create_app())
    resp = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["X-Request-ID"] == "abc-123"


def test_request_id_generated_and_invalid_replaced() -> None:
    client = TestClient(create_app())
    generated = client.get("/health").headers["X-Request-ID"]
    assert generated
    replaced = client.get("/health", headers={"X-Request-ID": "bad id!"}).headers["X-Request-ID"]
    assert replaced != "bad id!"
    assert replaced != generated


def test_request_id_on_error_response() -> None:
    client = TestClient(create_app())
    resp = client.delete("/documents/nope", headers={"X-Request-ID": "e1"})
    assert resp.status_code == 404
    assert resp.headers["X-Request-ID"] == "e1"


def test_json_formatter_includes_request_id() -> None:
    token = request_id_var.set("rid-1")
    try:
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("x",), None)
        data = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert data["message"] == "hello x"
    assert data["level"] == "INFO"
    assert data["request_id"] == "rid-1"
