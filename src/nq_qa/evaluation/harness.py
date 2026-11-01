"""Evaluation routines used by ``scripts/evaluate.py`` and ``scripts/train_reader.py``.

Three questions, three functions:

1. :func:`evaluate_retrieval` - does the retriever put the gold passage in the top k?
2. :func:`evaluate_reader`    - given the gold passage, does the reader extract the answer?
3. :func:`evaluate_end_to_end` - starting from the question only, is the final answer right?

Everything is computed on a *closed corpus* (the passages of the evaluation questions plus
distractors), which is far easier than retrieving from all of Wikipedia - see the README.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from tqdm.auto import tqdm

from nq_qa.evaluation.metrics import (
    answer_in_text,
    best_null_threshold,
    retrieval_metrics,
    squad_v2_metrics,
)
from nq_qa.models.reader import ExtractiveReader
from nq_qa.models.retriever import HybridRetriever, PassageIndex
from nq_qa.pipeline.e2e_qa import EndToEndQAPipeline

logger = logging.getLogger(__name__)

Record = dict[str, Any]


def evaluate_retrieval(
    retriever: HybridRetriever,
    index: PassageIndex,
    queries: Sequence[Record],
    modes: Sequence[str] = ("bm25",),
    ks: Sequence[int] = (1, 5, 10, 20),
    show_progress: bool = False,
) -> dict[str, dict[str, float]]:
    """Recall@k / MRR of every mode, by gold passage id and by answer string.

    ``recall@k`` asks for the exact gold passage; ``answer_recall@k`` is the more lenient
    "does any retrieved passage contain the answer" (the DPR top-k accuracy).
    """
    depth = max(ks)
    results: dict[str, dict[str, float]] = {}
    for mode in modes:
        id_hits: list[list[bool]] = []
        answer_hits: list[list[bool]] = []
        started = time.perf_counter()
        iterator = tqdm(queries, desc=f"retrieval[{mode}]", disable=not show_progress)
        for query in iterator:
            retrieved = retriever.retrieve(query["question"], depth, index=index, mode=mode)
            answers = query["answers"]["text"]
            id_hits.append([p.doc_id == query["passage_id"] for p in retrieved])
            answer_hits.append([answer_in_text(p.text, answers) for p in retrieved])
        elapsed = time.perf_counter() - started

        metrics = retrieval_metrics(id_hits, ks)
        answer_metrics = retrieval_metrics(answer_hits, ks)
        metrics.update(
            {f"answer_{k}": v for k, v in answer_metrics.items() if k.startswith("recall")}
        )
        metrics["queries"] = float(len(queries))
        metrics["avg_latency_ms"] = round(1000.0 * elapsed / max(1, len(queries)), 2)
        results[mode] = metrics
    return results


def evaluate_reader(
    reader: ExtractiveReader,
    examples: Sequence[Record],
    batch_chunk: int = 64,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Reader quality when handed the gold passage (an upper bound for the full pipeline).

    Also tunes the abstention threshold on these examples (``best_null_threshold``).
    """
    predictions: dict[str, str] = {}
    best_spans: dict[str, str] = {}
    null_diffs: dict[str, float] = {}
    references: dict[str, list[str]] = {}

    chunks = range(0, len(examples), batch_chunk)
    for begin in tqdm(chunks, desc="reader", disable=not show_progress):
        batch = examples[begin : begin + batch_chunk]
        spans = reader.predict_batch([e["question"] for e in batch], [e["context"] for e in batch])
        for example, span in zip(batch, spans, strict=True):
            key = example["id"]
            references[key] = list(example["answers"]["text"])
            best_spans[key] = span.text
            null_diffs[key] = span.null_diff
            predictions[key] = span.text if span.answerable else ""

    metrics: dict[str, Any] = squad_v2_metrics(predictions, references)
    threshold, f1_tuned, f1_never = best_null_threshold(null_diffs, best_spans, references)
    metrics.update(
        {
            "best_null_threshold": round(threshold, 4),
            "f1_at_best_threshold": f1_tuned,
            "f1_without_abstention": f1_never,
        }
    )
    return metrics


def evaluate_end_to_end(
    pipeline: EndToEndQAPipeline,
    queries: Sequence[Record],
    top_k: int = 5,
    mode: str | None = None,
    show_progress: bool = False,
) -> dict[str, float]:
    """EM / F1 of the final answer when only the question is given."""
    predictions: dict[str, str] = {}
    references: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query in tqdm(queries, desc="end-to-end", disable=not show_progress):
        result = pipeline.answer_question(query["question"], top_k=top_k, mode=mode)
        predictions[query["id"]] = result["best_answer"]
        references[query["id"]] = list(query["answers"]["text"])
        latencies.append(result["latency_ms"])

    metrics = squad_v2_metrics(predictions, references)
    metrics["avg_latency_ms"] = round(sum(latencies) / max(1, len(latencies)), 2)
    metrics["top_k"] = float(top_k)
    return metrics
