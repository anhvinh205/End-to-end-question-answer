"""End-to-end question answering: retriever -> reader -> score calibration.

Final candidate score::

    S = alpha * P_reader + (1 - alpha) * S_retriever

``P_reader`` is the span probability (already in [0, 1]); ``S_retriever`` is the retriever
score **min-max normalised over the retrieved set**.  Raw retriever scores live on a different
scale for every mode (RRF ~0.03, inner product ~0.5, BM25 ~10), so they must be put on the
reader's scale before they can be averaged with it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from nq_qa.models.reader import ExtractiveReader
from nq_qa.models.retriever import HybridRetriever, Passage, PassageIndex, RetrievedPassage

MIN_CHUNK_CHARS = 20


@dataclass(slots=True)
class Candidate:
    rank: int  # retrieval rank (1 = best passage)
    doc_id: str
    context: str
    answer: str
    start: int
    end: int
    answerable: bool
    reader_score: float
    retriever_score: float  # normalised to [0, 1] within this result set
    combined_score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    matched_question: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def min_max_normalize(values: Sequence[float]) -> list[float]:
    """Scale to [0, 1]; a constant (or single) series maps to 1.0."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


class EndToEndQAPipeline:
    """Orchestrates retrieval, reading and calibration."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reader: ExtractiveReader,
        alpha_reader_weight: float = 0.7,
    ) -> None:
        if not 0.0 <= alpha_reader_weight <= 1.0:
            raise ValueError("alpha_reader_weight must be within [0, 1]")
        self.retriever = retriever
        self.reader = reader
        self.alpha = alpha_reader_weight

    # ------------------------------------------------------------------ public API
    def answer_question(
        self, question: str, top_k: int = 5, mode: str | None = None
    ) -> dict[str, Any]:
        """Answer from the shared knowledge base (the loaded NQ index)."""
        started = time.perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=top_k, mode=mode)
        return self._answer(question, retrieved, started, mode or self.retriever.mode)

    def answer_from_contexts(
        self,
        question: str,
        contexts: Sequence[str],
        top_k: int = 3,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Answer over ad-hoc text (e.g. PDF chunks) using a throw-away index.

        The shared knowledge base is never modified, so concurrent users cannot interfere.
        """
        started = time.perf_counter()
        passages = [
            Passage(id=f"chunk_{i}", text=text.strip())
            for i, text in enumerate(contexts)
            if len(text.strip()) >= MIN_CHUNK_CHARS
        ]
        if not passages:
            raise ValueError(
                f"No usable text: every chunk is shorter than {MIN_CHUNK_CHARS} characters."
            )
        index: PassageIndex = self.retriever.build_index(passages)
        mode = mode or ("hybrid" if index.has_dense and self.retriever.encoder else "bm25")
        retrieved = self.retriever.retrieve(question, top_k=top_k, index=index, mode=mode)
        return self._answer(question, retrieved, started, mode)

    def describe(self) -> dict[str, Any]:
        """Static facts about the loaded models (used by ``/health``)."""
        index = self.retriever.index
        return {
            "mode": self.retriever.mode,
            "available_modes": list(self.retriever.available_modes()),
            "indexed_passages": len(index) if index is not None else 0,
            "reader_model": self.reader.model_name_or_path,
            "abstention_enabled": self.reader.null_threshold is not None,
            "encoder": self.retriever.encoder.spec if self.retriever.encoder else None,
        }

    # ------------------------------------------------------------------- internals
    def _answer(
        self,
        question: str,
        retrieved: list[RetrievedPassage],
        started: float,
        mode: str,
    ) -> dict[str, Any]:
        retrieval_done = time.perf_counter()
        spans = self.reader.predict_batch([question] * len(retrieved), [p.text for p in retrieved])
        reader_done = time.perf_counter()

        normalized = min_max_normalize([p.score for p in retrieved])
        candidates = [
            Candidate(
                rank=passage.rank,
                doc_id=passage.doc_id,
                context=passage.text,
                answer=span.text,
                start=span.start,
                end=span.end,
                answerable=span.answerable,
                reader_score=round(span.score, 6),
                retriever_score=round(norm, 6),
                combined_score=round(self.alpha * span.score + (1.0 - self.alpha) * norm, 6),
                dense_score=passage.dense_score,
                sparse_score=passage.sparse_score,
                matched_question=passage.question,
            )
            for passage, span, norm in zip(retrieved, spans, normalized, strict=True)
        ]
        # Candidates the reader would answer come first; ties broken by the combined score.
        candidates.sort(key=lambda c: (c.answerable, c.combined_score), reverse=True)

        best = candidates[0] if candidates and candidates[0].answerable else None
        return {
            "question": question,
            "answerable": best is not None,
            "best_answer": best.answer if best else "",
            "best_score": best.combined_score if best else 0.0,
            "best_context": best.context if best else "",
            "mode": mode,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "retrieval_ms": round((retrieval_done - started) * 1000.0, 2),
            "reader_ms": round((reader_done - retrieval_done) * 1000.0, 2),
            "candidates": [c.to_dict() for c in candidates],
        }
