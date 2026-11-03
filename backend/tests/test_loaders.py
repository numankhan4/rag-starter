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
    bad = c.post("/ingest/file", files={"file": ("a.pdf", b"x", "application/pdf")})
    assert bad.status_code == 415
