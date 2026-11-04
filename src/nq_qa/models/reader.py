"""Extractive reader: predicts an answer span inside a passage.

Inference and evaluation share **one** decoding path (:func:`decode_window`) so the numbers
reported by ``scripts/evaluate.py`` are exactly what the API serves.

Handling of "no answer" follows the SQuAD v2 recipe: the null score of an example is the
smallest ``start[CLS] + end[CLS]`` over its sliding windows and the model abstains when

    null_score - best_span_score > null_threshold

The threshold is tuned on validation data by ``scripts/train_reader.py`` and stored next
to the weights (``reader_meta.json``).  A checkpoint without that file (e.g. a model fine
tuned on SQuAD v1.1, which never saw ``[CLS]`` labels) never abstains.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)

META_FILE = "reader_meta.json"
NO_SPAN_NULL_DIFF = 1e6  # finite stand-in when a passage has no decodable span


@dataclass(slots=True)
class SpanPrediction:
    """Answer of the reader for one ``(question, passage)`` pair."""

    text: str
    start: int  # character offsets inside the passage
    end: int
    score: float  # probability of the span (start x end softmax) in its best window
    null_score: float  # probability of "no answer"
    null_diff: float  # null logit - span logit; > threshold  =>  abstain
    answerable: bool


@dataclass(slots=True)
class WindowSpan:
    """Best span of a single sliding window."""

    start_char: int
    end_char: int
    span_logit: float
    span_prob: float
    null_logit: float
    null_prob: float


def _logsumexp(values: np.ndarray) -> float:
    peak = float(np.max(values))
    return peak + float(np.log(np.exp(values - peak).sum()))


def decode_window(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offsets: Sequence[tuple[int, int] | None],
    n_best: int = 20,
    max_answer_tokens: int = 30,
) -> WindowSpan | None:
    """Best ``(start, end)`` span among the ``n_best`` start x ``n_best`` end candidates.

    ``offsets[i]`` is ``None`` for tokens that are not part of the passage (question,
    special tokens, padding); those can never be an answer.  Probabilities are softmaxes
    over ``{[CLS]} U passage tokens`` so that span and null probabilities are comparable.
    """
    context_idx = np.fromiter((i for i, o in enumerate(offsets) if o is not None), dtype=np.int64)
    if context_idx.size == 0:
        return None

    n = min(n_best, int(context_idx.size))
    top_start = context_idx[np.argsort(-start_logits[context_idx])[:n]]
    top_end = context_idx[np.argsort(-end_logits[context_idx])[:n]]

    start_grid, end_grid = top_start[:, None], top_end[None, :]
    valid = (end_grid >= start_grid) & (end_grid - start_grid + 1 <= max_answer_tokens)
    if not valid.any():
        return None
    scores = np.where(
        valid, start_logits[top_start][:, None] + end_logits[top_end][None, :], -np.inf
    )
    row, col = divmod(int(np.argmax(scores)), scores.shape[1])
    start_idx, end_idx = int(top_start[row]), int(top_end[col])
    span_logit = float(scores[row, col])

    allowed = np.concatenate(([0], context_idx))
    log_z = _logsumexp(start_logits[allowed]) + _logsumexp(end_logits[allowed])
    null_logit = float(start_logits[0] + end_logits[0])
    start_offset, end_offset = offsets[start_idx], offsets[end_idx]
    if start_offset is None or end_offset is None:
        return None
    return WindowSpan(
        start_char=int(start_offset[0]),
        end_char=int(end_offset[1]),
        span_logit=span_logit,
        span_prob=math.exp(span_logit - log_z),
        null_logit=null_logit,
        null_prob=math.exp(null_logit - log_z),
    )


def read_reader_meta(path: str | Path) -> dict[str, Any]:
    """Metadata written by the training script (empty dict if absent)."""
    meta_file = Path(path) / META_FILE
    if meta_file.is_file():
        return json.loads(meta_file.read_text(encoding="utf-8"))
    return {}


def write_reader_meta(path: str | Path, meta: dict[str, Any]) -> None:
    (Path(path) / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


class ExtractiveReader:
    """Transformer span-extraction model with batched, sliding-window inference."""

    def __init__(
        self,
        model_name_or_path: str,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        max_length: int = 384,
        stride: int = 128,
        batch_size: int = 16,
        device: str | None = None,
        null_threshold: float | None = None,
        abstain: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
        self.model.to(self.device).eval()

        meta = read_reader_meta(model_name_or_path)
        threshold = null_threshold if null_threshold is not None else meta.get("null_threshold")
        self.null_threshold: float | None = (
            float(threshold) if (abstain and threshold is not None) else None
        )

    @torch.inference_mode()
    def predict_batch(
        self, questions: Sequence[str], contexts: Sequence[str]
    ) -> list[SpanPrediction]:
        """Predict one span per ``(question, context)`` pair (long passages use windows)."""
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have the same length")
        if not questions:
            return []

        enc = self.tokenizer(
            [q.strip() for q in questions],
            list(contexts),
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sample_map = enc["overflow_to_sample_mapping"]
        keys = [k for k in self.tokenizer.model_input_names if k in enc]
        n_windows = len(enc["input_ids"])
        windows: list[list[WindowSpan | None]] = [[] for _ in questions]

        order = sorted(range(n_windows), key=lambda i: len(enc["input_ids"][i]))
        for begin in range(0, n_windows, self.batch_size):
            ids = order[begin : begin + self.batch_size]
            batch = self.tokenizer.pad(
                {k: [enc[k][i] for i in ids] for k in keys}, padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits.float().cpu().numpy()
            end_logits = outputs.end_logits.float().cpu().numpy()
            for row, i in enumerate(ids):
                length = len(enc["input_ids"][i])
                sequence_ids = enc.sequence_ids(i)
                offsets: list[tuple[int, int] | None] = [
                    (int(off[0]), int(off[1])) if sequence_ids[k] == 1 else None
                    for k, off in enumerate(enc["offset_mapping"][i])
                ]
                windows[sample_map[i]].append(
                    decode_window(
                        start_logits[row, :length],
                        end_logits[row, :length],
                        offsets,
                        self.n_best_size,
                        self.max_answer_length,
                    )
                )
        return [self._aggregate(ctx, wins) for ctx, wins in zip(contexts, windows, strict=True)]

    def _aggregate(self, context: str, windows: list[WindowSpan | None]) -> SpanPrediction:
        found = [w for w in windows if w is not None]
        if not found:
            return SpanPrediction("", 0, 0, 0.0, 1.0, NO_SPAN_NULL_DIFF, False)
        best = max(found, key=lambda w: w.span_logit)
        null_diff = min(w.null_logit for w in found) - best.span_logit
        answerable = self.null_threshold is None or null_diff <= self.null_threshold
        return SpanPrediction(
            text=context[best.start_char : best.end_char],
            start=best.start_char,
            end=best.end_char,
            score=best.span_prob,
            null_score=min(w.null_prob for w in found),
            null_diff=null_diff,
            answerable=answerable,
        )

    def predict(self, question: str, context: str) -> SpanPrediction:
        """Convenience wrapper around :meth:`predict_batch` for a single pair."""
        return self.predict_batch([question], [context])[0]
