import httpx
import pytest

from nq_qa.client import LocalBackend, QAClient
from nq_qa.utils.documents import chunk_words, extract_pdf_text, highlight_answer_span


def test_chunk_words_overlaps_and_covers_the_text():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_words(text, chunk_size=40, overlap=10, min_chars=1)
    assert [len(c.split()) for c in chunks] == [40, 40, 40]
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert chunks[-1].endswith("w99")


def test_chunk_words_short_text_and_validation():
    assert chunk_words("only a few words here in this text", chunk_size=50, min_chars=5) == [
        "only a few words here in this text"
    ]
    assert chunk_words("tiny", chunk_size=10, min_chars=30) == []
    assert chunk_words("", chunk_size=10) == []
    assert (
        len(chunk_words(" ".join(["w"] * 150), min_chars=1)) == 1
    )  # defaults: 150 words, 30 overlap
    with pytest.raises(ValueError):
        chunk_words("text", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_words("text", chunk_size=10, overlap=10)


def test_highlight_wraps_the_span_and_escapes_html():
    html = highlight_answer_span("Tom <b>Brady</b> won", 4, 16)
    assert "<mark" in html and "&lt;b&gt;Brady&lt;/b&gt;</mark>" in html
    assert html.startswith("Tom ") and html.endswith(" won")
    assert "<b>" not in html  # raw markup from the document must never reach the page


@pytest.mark.parametrize("span", [(-1, 3), (5, 5), (4, 2), (0, 999)])
def test_highlight_ignores_invalid_spans(span):
    assert highlight_answer_span("a < b", *span) == "a &lt; b"


def test_extract_pdf_text_reads_a_real_pdf(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    assert pypdf is not None
    assert extract_pdf_text(str(path)).strip() == ""


# ------------------------------------------------------------------------------ client
def mock_api(handler):
    return QAClient("http://api.test", transport=httpx.MockTransport(handler))


def test_client_sends_the_documented_payloads():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = request.content
        return httpx.Response(200, json={"ok": True})

    client = mock_api(handler)
    assert client.ask("who wrote hamlet", top_k=3, mode="bm25") == {"ok": True}
    client.ask_documents("what?", ["chunk one", "chunk two"], top_k=2)
    assert b'"mode":"bm25"' in seen["/api/v1/qa"].replace(b" ", b"")
    assert b'"contexts":["chunkone","chunktwo"]' in seen["/api/v1/qa/documents"].replace(b" ", b"")


def test_client_omits_mode_when_not_given():
    def handler(request):
        assert b"mode" not in request.content
        return httpx.Response(200, json={})

    mock_api(handler).ask("who wrote hamlet")


def test_client_health():
    client = mock_api(lambda request: httpx.Response(200, json={"status": "ok", "ready": True}))
    assert client.health()["ready"] is True


def test_client_raises_readable_errors():
    client = mock_api(lambda request: httpx.Response(400, json={"detail": "No usable text"}))
    with pytest.raises(RuntimeError, match="API error 400: No usable text"):
        client.ask_documents("q?", ["x"])


def test_client_error_without_json_body():
    client = mock_api(lambda request: httpx.Response(502, content=b""))
    with pytest.raises(RuntimeError, match="API error 502"):
        client.ask("question")


def test_local_backend_delegates_to_the_pipeline():
    class Pipeline:
        def answer_question(self, question, top_k, mode):
            return {"q": question, "k": top_k, "mode": mode}

        def answer_from_contexts(self, question, contexts, top_k):
            return {"docs": list(contexts), "k": top_k}

        def describe(self):
            return {"mode": "hybrid"}

    backend = LocalBackend(Pipeline())
    assert backend.ask("who", 4, "dense") == {"q": "who", "k": 4, "mode": "dense"}
    assert backend.ask_documents("who", ("a", "b"), 2) == {"docs": ["a", "b"], "k": 2}
    assert backend.health() == {"status": "ok", "mode": "hybrid"}
