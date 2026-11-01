"""Knowledge-base construction from the processed Natural Questions splits."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from nq_qa.data.nq_csv import read_jsonl
from nq_qa.data.passage import Passage


def passages_from_records(records: list[dict[str, Any]]) -> list[Passage]:
    """Unique passages (by id) of NQ records, remembering the question they were written for."""
    seen: dict[str, Passage] = {}
    for rec in records:
        pid = rec.get("passage_id") or rec["id"]
        if pid not in seen:
            seen[pid] = Passage(id=pid, text=rec["context"], question=rec.get("question", ""))
    return list(seen.values())


def select_corpus(processed_dir: str | Path, corpus_size: int, seed: int = 42) -> list[Passage]:
    """Passages of the knowledge base.

    All validation and test gold passages come first (so every evaluation question is
    answerable from the corpus); the remaining slots are filled with random training
    passages that act as distractors.
    """
    processed_dir = Path(processed_dir)
    eval_records: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        eval_records += read_jsonl(processed_dir / f"{split}.jsonl")
    passages = passages_from_records(eval_records)

    if len(passages) < corpus_size:
        known = {p.id for p in passages}
        train = [
            p
            for p in passages_from_records(read_jsonl(processed_dir / "train.jsonl"))
            if p.id not in known
        ]
        random.Random(seed).shuffle(train)
        passages += train[: corpus_size - len(passages)]
    return passages[:corpus_size]
