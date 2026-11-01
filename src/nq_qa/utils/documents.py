"""Helpers for ad-hoc document QA: PDF text extraction, chunking, answer highlighting."""

from __future__ import annotations

import html
from typing import IO, Any


def chunk_words(
    text: str, chunk_size: int = 150, overlap: int | None = None, min_chars: int = 30
) -> list[str]:
    """Split ``text`` into overlapping windows of ``chunk_size`` words.

    ~150 words matches the length of a Natural Questions paragraph, which keeps retrieval
    granular; the overlap (default: 20 % of the chunk) stops answers from being cut at a
    chunk boundary.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap is None:
        overlap = chunk_size // 5
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    words = text.split()
    step = chunk_size - overlap
    chunks: list[str] = []
    for begin in range(0, len(words), step):
        chunk = " ".join(words[begin : begin + chunk_size])
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if begin + chunk_size >= len(words):
            break
    return chunks


def extract_pdf_text(source: str | IO[bytes] | Any) -> str:
    """Concatenate the text layer of every page (needs the ``ui`` extra: ``pypdf``)."""
    from pypdf import PdfReader  # lazy import

    reader = PdfReader(source)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def highlight_answer_span(context: str, start: int, end: int) -> str:
    """HTML-escape ``context`` and wrap ``context[start:end]`` in a ``<mark>`` element."""
    if start < 0 or end <= start or end > len(context):
        return html.escape(context)
    return (
        f"{html.escape(context[:start])}"
        "<mark style='background-color:#fde047;color:#0f172a;padding:2px 6px;"
        f"border-radius:4px;font-weight:700'>{html.escape(context[start:end])}</mark>"
        f"{html.escape(context[end:])}"
    )
