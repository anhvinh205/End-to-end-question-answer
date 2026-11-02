"""End-to-end question answering: retriever -> (optional reranker) -> reader -> calibration.

Final candidate score::

    S = alpha * P_reader + (1 - alpha) * S_retriever

``P_reader`` is the span probability (already in [0, 1]).
``S_retriever`` is calibrated using mode-specific curves (or legacy min-max).
Mode-specific calibration avoids the flaw of relative min-max, where an irrelevant
passage (e.g. BM25 = 0.8) is artificially forced to 1.0 (100% confidence).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from nq_qa.models.reader import ExtractiveReader
from nq_qa.models.retriever import HybridRetriever, Passage, PassageIndex, RetrievedPassage

MIN_CHUNK_CHARS = 20


@dataclass(slots=True)
class Candidate:
    rank: int  # retrieval / reranking rank (1 = best passage)
    doc_id: str
    context: str
    answer: str
    start: int
    end: int
    answerable: bool
    reader_score: float
    retriever_score: float  # calibrated or normalised to [0, 1]
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


def calibrate_retriever_score(score: float, mode: str, rrf_k: int = 60) -> float:
    """Calibrate raw retriever score to [0, 1] without relative min-max distortion.

    Prevents irrelevant top-1 candidates (e.g. BM25 = 0.8) from being artificially scaled to 1.0.
    """
    if score <= 0.0:
        return 0.0
    if mode == "hybrid":
        max_rrf = 2.0 / (rrf_k + 1.0)
        return min(1.0, max(0.0, score / max_rrf))
    if mode == "dense":
        # Cosine similarity range ~ [0.2, 0.8] for relevant texts
        return min(1.0, max(0.0, (score - 0.2) / 0.6))
    if mode == "rerank":
        # Logistic sigmoid for cross-encoder relevance logits
        clamped = max(min(score, 15.0), -15.0)
        return 1.0 / (1.0 + math.exp(-clamped))
    # bm25 soft saturation curve: score / (score + 10.0)
    return float(score / (score + 10.0))


class EndToEndQAPipeline:
    """Orchestrates retrieval, optional cross-encoder re-ranking, reading and calibration."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reader: ExtractiveReader,
        alpha_reader_weight: float = 0.7,
        reranker: Any | None = None,
        candidate_pool: int = 50,
        score_calibration: str = "min_max",
        min_confidence: float = 0.0,
    ) -> None:
        if not 0.0 <= alpha_reader_weight <= 1.0:
            raise ValueError("alpha_reader_weight must be within [0, 1]")
        if score_calibration not in ("calibrated", "min_max"):
            raise ValueError("score_calibration must be 'calibrated' or 'min_max'")
        self.retriever = retriever
        self.reader = reader
        self.alpha = alpha_reader_weight
        self.reranker = reranker
        self.candidate_pool = candidate_pool
        self.score_calibration = score_calibration
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------ public API
    def answer_question(
        self, question: str, top_k: int = 5, mode: str | None = None
    ) -> dict[str, Any]:
        """Answer from the shared knowledge base (the loaded NQ index)."""
        started = time.perf_counter()
        query_mode = mode or self.retriever.mode
        retrieve_k = self.candidate_pool if self.reranker is not None else top_k
        retrieved = self.retriever.retrieve(question, top_k=retrieve_k, mode=mode)
        retrieval_done = time.perf_counter()

        if self.reranker is not None:
            retrieved = self.reranker.rerank(question, retrieved, top_k=top_k)
            query_mode = "rerank"
        rerank_done = time.perf_counter()

        return self._answer(
            question,
            retrieved,
            started,
            query_mode,
            retrieval_ms=round((retrieval_done - started) * 1000.0, 2),
            rerank_ms=round((rerank_done - retrieval_done) * 1000.0, 2),
        )

    def answer_from_contexts(
        self,
        question: str,
        contexts: Sequence[str],
        top_k: int = 3,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Answer over ad-hoc text (e.g. PDF chunks) using a throw-away index."""
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
        query_mode = mode or ("hybrid" if index.has_dense and self.retriever.encoder else "bm25")
        retrieve_k = self.candidate_pool if self.reranker is not None else top_k
        retrieved = self.retriever.retrieve(
            question, top_k=retrieve_k, index=index, mode=query_mode
        )
        retrieval_done = time.perf_counter()

        if self.reranker is not None:
            retrieved = self.reranker.rerank(question, retrieved, top_k=top_k)
            query_mode = "rerank"
        rerank_done = time.perf_counter()

        return self._answer(
            question,
            retrieved,
            started,
            query_mode,
            retrieval_ms=round((retrieval_done - started) * 1000.0, 2),
            rerank_ms=round((rerank_done - retrieval_done) * 1000.0, 2),
        )

    def describe(self) -> dict[str, Any]:
        """Static facts about the loaded models (used by ``/health``)."""
        index = self.retriever.index
        return {
            "mode": self.retriever.mode,
            "available_modes": list(self.retriever.available_modes()),
            "indexed_passages": len(index) if index is not None else 0,
            "reader_model": self.reader.model_name_or_path,
            "reranker_model": getattr(self.reranker, "model_name_or_path", None),
            "score_calibration": self.score_calibration,
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
        retrieval_ms: float = 0.0,
        rerank_ms: float = 0.0,
    ) -> dict[str, Any]:
        reader_start = time.perf_counter()
        spans = self.reader.predict_batch([question] * len(retrieved), [p.text for p in retrieved])
        reader_done = time.perf_counter()

        if self.score_calibration == "calibrated":
            rrf_k = int(getattr(self.retriever, "rrf_k", 60))
            normalized = [calibrate_retriever_score(p.score, mode, rrf_k=rrf_k) for p in retrieved]
        else:
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

        best = (
            candidates[0]
            if (
                candidates
                and candidates[0].answerable
                and candidates[0].combined_score >= self.min_confidence
            )
            else None
        )
        return {
            "question": question,
            "answerable": best is not None,
            "best_answer": best.answer if best else "",
            "best_score": best.combined_score if best else 0.0,
            "best_context": best.context if best else "",
            "mode": mode,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "retrieval_ms": retrieval_ms or round((reader_start - started) * 1000.0, 2),
            "rerank_ms": rerank_ms,
            "reader_ms": round((reader_done - reader_start) * 1000.0, 2),
            "candidates": [c.to_dict() for c in candidates],
        }
