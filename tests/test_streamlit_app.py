"""Drive the Streamlit app headlessly with a fake backend (no models, no network)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
CONTEXT = "The World Wide Web was invented by Tim Berners-Lee in 1989 while working at CERN."


def candidate(answerable=True):
    start = CONTEXT.index("Tim Berners-Lee")
    return {
        "rank": 1, "doc_id": "p1", "context": CONTEXT, "answer": "Tim Berners-Lee", "start": start,
        "end": start + len("Tim Berners-Lee"), "answerable": answerable, "reader_score": 0.91,
        "retriever_score": 1.0, "combined_score": 0.94, "dense_score": 0.7, "sparse_score": 9.1,
        "matched_question": "",
    }  # fmt: skip


class FakeBackend:
    def __init__(self, answerable=True):
        self.answerable = answerable
        self.calls = []

    def health(self):
        return {
            "status": "ok", "mode": "hybrid", "available_modes": ["bm25", "dense", "hybrid"],
            "indexed_passages": 12, "reader_model": "fake-reader", "abstention_enabled": True,
        }  # fmt: skip

    def ask(self, question, top_k=5, mode=None):
        self.calls.append((question, top_k, mode))
        return {
            "question": question, "answerable": self.answerable,
            "best_answer": "Tim Berners-Lee" if self.answerable else "", "best_score": 0.94,
            "best_context": CONTEXT, "mode": mode or "hybrid", "latency_ms": 42.0,
            "retrieval_ms": 10.0, "reader_ms": 32.0, "candidates": [candidate(self.answerable)],
        }  # fmt: skip

    def ask_documents(self, question, contexts, top_k=3):
        return self.ask(question, top_k)


def launch(backend):
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["backend"] = backend
    return app.run()


def markdown_text(app):
    return "\n".join(m.value for m in app.markdown)


def click(app, label):
    next(b for b in app.button if b.label == label).click()
    return app.run()


def test_app_renders_without_errors_and_shows_system_status():
    app = launch(FakeBackend())
    assert not app.exception
    assert app.title[0].value == "Ask Natural Questions"
    sidebar = " ".join(c.value for c in app.sidebar.caption)
    assert (
        "12 passages" in sidebar and "fake-reader" in sidebar and "Abstains when unsure" in sidebar
    )
    assert [t.label for t in app.tabs] == ["Knowledge base", "Your document"]


def test_asking_a_question_shows_the_answer_and_highlights_the_evidence():
    backend = FakeBackend()
    app = launch(backend)
    app.text_input[0].set_value("who invented the world wide web")
    app = click(app, "Ask")
    assert not app.exception
    text = markdown_text(app)
    assert '<div class="answer-text">Tim Berners-Lee</div>' in text
    assert "<mark" in text and "Read 1 passages in 42 ms" in text
    assert backend.calls == [("who invented the world wide web", 5, "hybrid")]


def test_settings_are_forwarded_to_the_backend():
    backend = FakeBackend()
    app = launch(backend)
    app.sidebar.selectbox[0].select("bm25")
    app.sidebar.slider[0].set_value(3)
    app.text_input[0].set_value("who invented the world wide web")
    click(app, "Ask")
    assert backend.calls[-1] == ("who invented the world wide web", 3, "bm25")


def test_abstention_is_explained_with_a_next_step():
    app = launch(FakeBackend(answerable=False))
    app.text_input[0].set_value("a question nobody can answer")
    app = click(app, "Ask")
    assert "No confident answer" in app.warning[0].value and "Rephrase" in app.warning[0].value
    assert '<div class="answer-text">' not in markdown_text(app)  # the CSS also mentions the class


def test_too_short_questions_are_rejected_before_calling_the_backend():
    backend = FakeBackend()
    app = launch(backend)
    app.text_input[0].set_value("hi")
    app = click(app, "Ask")
    assert "at least three characters" in app.error[0].value and backend.calls == []


def test_example_buttons_ask_immediately():
    backend = FakeBackend()
    app = launch(backend)
    example = next(b for b in app.button if b.key and b.key.startswith("example-"))
    example.click()
    app.run()
    assert len(backend.calls) == 1 and backend.calls[0][0] == example.label
    assert '<div class="answer-text">' in markdown_text(app)
