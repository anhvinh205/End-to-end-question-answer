"""Streamlit demo: ask a question and see the evidence behind the answer.

make ui                                    # loads the models in-process
QA_API_URL=http://localhost:8000 make ui   # talks to the FastAPI service instead
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from nq_qa.client import LocalBackend, QAClient
from nq_qa.data.nq_csv import read_jsonl
from nq_qa.utils.config import load_config
from nq_qa.utils.documents import chunk_words, extract_pdf_text, highlight_answer_span

st.set_page_config(page_title="Ask Natural Questions", page_icon="🔎", layout="centered")

MODE_LABELS = {
    "hybrid": "Hybrid (keyword + semantic)",
    "dense": "Semantic (dense vectors)",
    "bm25": "Keyword (BM25)",
}
FALLBACK_EXAMPLES = [
    "who invented the world wide web",
    "who played the incredible hulk in the 70s",
    "what is the name of the snowman in frozen",
    "who wrote the song if you don't know me by now",
]

STYLE = """
<style>
html, body, .stApp {
    font-family: 'Segoe UI', Arial, sans-serif;
}
.block-container { max-width: 52rem; padding-top: 2.5rem; }
h1 {
    font-family: Georgia, 'Times New Roman', serif;
    font-weight: 600;
    letter-spacing: -0.01em;
}
.answer-text {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 2rem;
    line-height: 1.25;
    font-weight: 600;
    color: #14213D;
    margin: 0.25rem 0 0.75rem 0;
}
.passage {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1.08rem;
    line-height: 1.75;
    max-width: 68ch;
    padding: 1rem 1.25rem;
    border-left: 3px solid #2F5BEA;
    background: var(--secondary-background-color, #F3F5F9);
    border-radius: 0 6px 6px 0;
    overflow-wrap: anywhere;
}
.passage mark {
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}
details[data-testid="stExpander"] summary,
details[data-testid="stExpander"] summary p,
details[data-testid="stExpander"] summary span {
    font-family: inherit !important;
    line-height: 1.5 !important;
}
.muted { color: #5B667A; font-size: 0.9rem; }
</style>
"""


@st.cache_resource(show_spinner="Loading the models (first start only)...")
def load_backend() -> Any:
    """HTTP client when ``QA_API_URL`` is set, otherwise the pipeline inside this process."""
    api_url = os.environ.get("QA_API_URL")
    if api_url:
        return QAClient(api_url)
    from nq_qa.pipeline.factory import build_pipeline

    return LocalBackend(build_pipeline(load_config()))


@st.cache_data(show_spinner=False)
def example_questions() -> list[str]:
    """Questions from the bundled sample, so the examples are answerable out of the box."""
    try:
        sample = Path(load_config().get("dataset", {}).get("sample_path", ""))
        if sample.is_file():
            return [r["question"] for r in read_jsonl(sample, limit=4)]
    except (OSError, ValueError, KeyError):
        pass
    return FALLBACK_EXAMPLES


@st.cache_data(show_spinner="Reading the document...")
def read_document(name: str, data: bytes) -> list[str]:
    import io

    if name.lower().endswith(".pdf"):
        text = extract_pdf_text(io.BytesIO(data))
    else:
        text = data.decode("utf-8", errors="ignore")
    return chunk_words(text)


def render_result(result: dict[str, Any]) -> None:
    """Answer first, then the passage it came from, then how it was chosen."""
    candidates = result["candidates"]
    if result["answerable"]:
        top = candidates[0]
        st.markdown(
            f'<div class="answer-text">{html.escape(result["best_answer"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="passage">{highlight_answer_span(top["context"], top["start"], top["end"])}</div>',
            unsafe_allow_html=True,
        )
        passage_count = len(candidates)
        passage_word = "passage" if passage_count == 1 else "passages"
        st.markdown(
            f'<p class="muted">Read {passage_count} {passage_word} in {result["latency_ms"]:.0f} ms '
            f"using {MODE_LABELS.get(result['mode'], result['mode']).lower()} retrieval.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "No confident answer in the retrieved passages. "
            "Rephrase the question, or read more passages in the sidebar."
        )

    st.subheader("How this answer was chosen")
    frame = pd.DataFrame(
        {
            "Passage": [c["rank"] for c in candidates],
            "Answer": [c["answer"] or "-" for c in candidates],
            "Reader confidence": [round(100 * c["reader_score"], 1) for c in candidates],
            "Retrieval score": [round(100 * c["retriever_score"], 1) for c in candidates],
            "Final score": [round(100 * c["combined_score"], 1) for c in candidates],
            "Confident": ["yes" if c["answerable"] else "no" for c in candidates],
        }
    )
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption(
        f"Retrieval took {result['retrieval_ms']:.0f} ms and the reader {result['reader_ms']:.0f} ms. "
        "The final score blends reader confidence with the retrieval score; the weights are set in the config."
    )
    for c in candidates[:3]:
        st.markdown(f"**Passage {c['rank']}**")
        st.markdown(
            f'<div class="passage">{highlight_answer_span(c["context"], c["start"], c["end"])}</div>',
            unsafe_allow_html=True,
        )


def knowledge_base_tab(backend: Any, mode: str, top_k: int) -> None:
    with st.form("ask-kb"):
        question = st.text_input("Question", placeholder="Who invented the world wide web?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        if len(question.strip()) < 3:
            st.error("Type a question of at least three characters.")
        else:
            st.session_state["kb_result"] = backend.ask(question.strip(), top_k=top_k, mode=mode)

    st.markdown('<p class="muted">Or try one of these:</p>', unsafe_allow_html=True)
    for example in example_questions():
        if st.button(example, key=f"example-{example}"):
            st.session_state["kb_result"] = backend.ask(example, top_k=top_k, mode=mode)

    if result := st.session_state.get("kb_result"):
        st.divider()
        st.caption(result["question"])
        render_result(result)


def document_tab(backend: Any, top_k: int) -> None:
    st.write("Search your own PDF or text file. The shared knowledge base is not modified.")
    upload = st.file_uploader("PDF or text file", type=["pdf", "txt", "md"])
    if upload is None:
        return
    chunks = read_document(upload.name, upload.getvalue())
    if not chunks:
        st.error("No text found. If this is a scanned PDF, run OCR on it first.")
        return
    st.caption(f"Split into {len(chunks)} passages of about 150 words.")
    with st.form("ask-doc"):
        question = st.text_input("Question about the document")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and len(question.strip()) >= 3:
        try:
            st.session_state["doc_result"] = backend.ask_documents(
                question.strip(), chunks, top_k=min(top_k, 5)
            )
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
    if result := st.session_state.get("doc_result"):
        st.divider()
        render_result(result)


def main() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    backend = st.session_state.get("backend") or load_backend()
    health = backend.health()

    st.title("Ask Natural Questions")
    st.write(
        "A question answering system built on Google's Natural Questions. It retrieves "
        "passages from Wikipedia and a Transformer reader marks the answer inside them."
    )

    with st.sidebar:
        st.header("Settings")
        modes = [m for m in MODE_LABELS if m in health.get("available_modes", [])] or ["bm25"]
        default = modes.index(health["mode"]) if health.get("mode") in modes else 0
        mode = st.selectbox("Retrieval", modes, index=default, format_func=MODE_LABELS.get)
        top_k = st.slider("Passages to read", 1, 10, 5)
        st.divider()
        st.caption(f"Knowledge base: {health.get('indexed_passages', 0):,} passages")
        st.caption(f"Reader: {health.get('reader_model', 'unknown')}")
        st.caption(
            "Abstains when unsure."
            if health.get("abstention_enabled")
            else "Always returns its best guess (no abstention model loaded)."
        )

    kb_tab, doc_tab = st.tabs(["Knowledge base", "Your document"])
    with kb_tab:
        knowledge_base_tab(backend, mode, top_k)
    with doc_tab:
        document_tab(backend, top_k)


main()
