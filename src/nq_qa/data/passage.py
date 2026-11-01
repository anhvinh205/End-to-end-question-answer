"""Lightweight passage record (kept free of torch/faiss imports)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Passage:
    id: str
    text: str
    question: str = ""  # NQ question this passage was written for (debugging only)
