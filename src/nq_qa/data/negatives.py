"""Negative examples for the extractive reader.

In the open-domain pipeline the reader sees the top-k passages of the retriever and most of
them do **not** contain the answer.  A reader trained only on positive pairs has never
learned to say "not here", so its confidence is meaningless for those passages.  We therefore
mix two kinds of negatives into the training set:

* **natural negatives** - NQ rows that have a long answer but no short answer, and
* **hard negatives** - for an answerable question, the best BM25 passage that is *not* its
  gold passage and does not contain the gold answer (guards against false negatives).

``negative_ratio`` is the target share of negatives in the returned set.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from typing import Any

from nq_qa.evaluation.metrics import answer_in_text
from nq_qa.models.bm25 import BM25Index

logger = logging.getLogger(__name__)


def synthesize_hard_negatives(
    positives: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    n: int,
    rng: random.Random,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Create up to ``n`` hard negatives for randomly chosen answerable ``positives``.

    ``pool`` supplies the candidate passages (each needs ``passage_id`` and ``context``).
    """
    if n <= 0 or not positives:
        return []
    unique: dict[str, str] = {}
    for rec in pool:
        unique.setdefault(rec["passage_id"], rec["context"])
    passage_ids = list(unique)
    bm25 = BM25Index().fit([unique[pid] for pid in passage_ids])

    chosen = rng.sample(list(positives), min(n, len(positives)))
    negatives: list[dict[str, Any]] = []
    for rec in chosen:
        gold_answers = rec["answers"]["text"]
        for position, _ in bm25.search(rec["question"], top_k + 1):
            pid = passage_ids[position]
            text = unique[pid]
            if pid == rec["passage_id"] or answer_in_text(text, gold_answers):
                continue
            negatives.append(
                {
                    "id": f"{rec['id']}-hardneg",
                    "question": rec["question"],
                    "context": text,
                    "passage_id": pid,
                    "answers": {"text": [], "answer_start": []},
                    "has_answer": False,
                    "context_type": "hard_negative",
                }
            )
            break
    return negatives


def build_training_records(
    records: Sequence[dict[str, Any]],
    negative_ratio: float = 0.3,
    seed: int = 42,
    pool: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return a shuffled training set where ``negative_ratio`` of the examples are negatives.

    Natural negatives found in ``records`` are used first; missing ones are synthesised as
    BM25 hard negatives drawn from ``pool`` (defaults to ``records``).
    """
    if not 0.0 <= negative_ratio < 1.0:
        raise ValueError("negative_ratio must be in [0, 1)")
    rng = random.Random(seed)
    positives = [r for r in records if r["has_answer"]]
    natural = [r for r in records if not r["has_answer"]]
    if not positives:
        return list(natural)

    target = round(len(positives) * negative_ratio / (1.0 - negative_ratio))
    rng.shuffle(natural)
    negatives = natural[:target]
    missing = target - len(negatives)
    if missing > 0:
        negatives += synthesize_hard_negatives(positives, pool or records, missing, rng)

    combined = list(positives) + negatives
    rng.shuffle(combined)
    logger.info(
        "Training set: %d positives + %d negatives (%.0f%% negatives)",
        len(positives),
        len(negatives),
        100.0 * len(negatives) / len(combined),
    )
    return combined
