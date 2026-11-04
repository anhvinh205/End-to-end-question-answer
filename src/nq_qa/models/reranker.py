"""Cross-Encoder re-ranking for retrieved passages.

Evaluates (query, passage) pairs with full cross-attention for high-precision re-ranking.
Typically re-ranks top 30-50 candidates from Hybrid/BM25 retrieval down to top 5 for the Reader.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nq_qa.models.retriever import RetrievedPassage

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Transformer Cross-Encoder producing a relevance score per (query, passage) pair."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RERANKER_MODEL,
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def rerank(
        self,
        query: str,
        passages: Sequence[RetrievedPassage],
        top_k: int = 5,
    ) -> list[RetrievedPassage]:
        """Re-rank ``passages`` for ``query`` and return top-``top_k`` passages."""
        if not passages:
            return []
        top_k = max(1, min(top_k, len(passages)))
        pairs = [(query, p.text) for p in passages]
        scores: list[float] = []

        for start_idx in range(0, len(pairs), self.batch_size):
            batch = pairs[start_idx : start_idx + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            if logits.shape[-1] == 1:
                batch_scores = logits.squeeze(-1).float().cpu().reshape(-1).tolist()
            else:
                batch_scores = logits[:, -1].float().cpu().reshape(-1).tolist()
            scores.extend(batch_scores)

        ranked_indices = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)[:top_k]
        results: list[RetrievedPassage] = []
        for new_rank, idx in enumerate(ranked_indices, start=1):
            orig = passages[idx]
            results.append(
                RetrievedPassage(
                    rank=new_rank,
                    doc_id=orig.doc_id,
                    text=orig.text,
                    score=round(float(scores[idx]), 6),
                    dense_score=orig.dense_score,
                    sparse_score=orig.sparse_score,
                    question=orig.question,
                )
            )
        return results
