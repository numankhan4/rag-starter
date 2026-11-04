import builtins

import pytest
from fastapi.testclient import TestClient

from app.loaders import load_html, load_markdown
from app.main import create_app

HTML = """<html><head><title>My Page</title><style>p{color:red}</style></head>
<body><h1>Intro</h1><script>var secret = 1;</script><p>Hello <b>world</b>.</p>
<h2>Details</h2><p>More &amp; more.</p></body></html>"""


def test_html_strips_script_style():
    r = load_html(HTML)
    assert r.title == "My Page"
    assert r.headings == ["Intro", "Details"]
    assert "secret" not in r.text and "color" not in r.text
    assert "Hello world." in r.text and "More & more." in r.text


def test_markdown():
    r = load_markdown(
        "# Title\n\nSome **bold** and [a link](http://x).\n\n```\n# not heading\n```\n## Sub\n- item"
    )
    assert r.title == "Title"
    assert r.headings == ["Title", "Sub"]
    assert "bold" in r.text and "**" not in r.text and "http" not in r.text
    assert "# not heading" in r.text


def test_ingest_file_endpoint():
    c = TestClient(create_app())
    r = c.post("/ingest/file", files={"file": ("page.html", HTML.encode(), "text/html")})
    assert r.status_code == 200 and r.json()["doc_id"] == "page.html"
    q = c.post("/query", json={"question": "hello world"}).json()
    assert q["sources"][0]["title"] == "My Page"
    assert "secret" not in q["sources"][0]["text"]
    ok = c.post("/ingest/file", files={"file": ("a.txt", b"plain text here", "text/plain")})
    assert ok.status_code == 200
    bad = c.post("/ingest/file", files={"file": ("a.docx", b"x", "application/octet-stream")})
    assert bad.status_code == 415


def make_pdf(text: str) -> bytes:
    """Build a tiny single-page PDF containing `text` (no third-party libs)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def test_ingest_pdf():
    pytest.importorskip("pypdf")
    c = TestClient(create_app())
    pdf = make_pdf("Hello PDF world")
    r = c.post("/ingest/file", files={"file": ("doc.pdf", pdf, "application/pdf")})
    assert r.status_code == 200 and r.json() == {"doc_id": "doc.pdf", "chunks": 1}
    hit = c.post("/query", json={"question": "Hello PDF"}).json()
    assert "Hello PDF world" in hit["sources"][0]["text"]


def test_invalid_pdf_is_422():
    pytest.importorskip("pypdf")
    c = TestClient(create_app())
    r = c.post("/ingest/file", files={"file": ("bad.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unprocessable_entity"


def test_pdf_without_pypdf_is_415(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf" or name.startswith("pypdf."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    c = TestClient(create_app())
    r = c.post("/ingest/file", files={"file": ("doc.pdf", make_pdf("x"), "application/pdf")})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"
    assert "pypdf" in r.json()["error"]["message"]
