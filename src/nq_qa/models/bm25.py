"""Okapi BM25 over an inverted index.

Implemented from the formula instead of wrapping a library:

.. math::

    \\mathrm{score}(q, d) = \\sum_{t \\in q} \\mathrm{idf}(t) \\cdot
        \\frac{f(t, d)\\,(k_1 + 1)}{f(t, d) + k_1\\,(1 - b + b\\,|d| / \\mathrm{avgdl})}

with the non-negative Lucene idf :math:`\\ln(1 + (N - n_t + 0.5) / (n_t + 0.5))`.

Only the postings of the query terms are touched, so a query costs
``O(sum of posting lengths)`` rather than ``O(corpus size)`` - noticeably faster than
scanning every document and cheap enough to rebuild at start-up.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Function words and interrogatives carry no signal for passage retrieval.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "after",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "s",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Lower-case word tokens, optionally without stop-words."""
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        filtered = [t for t in tokens if t not in STOPWORDS]
        # keep the raw tokens for queries made only of stop-words ("who is who")
        return filtered or tokens
    return tokens


class BM25Index:
    """Inverted-index BM25 scorer."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.num_docs = 0
        self.avgdl = 0.0
        self._doc_len = np.zeros(0, dtype=np.float32)
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}

    def fit(self, documents: Sequence[str]) -> BM25Index:
        """Index ``documents``; document ids are their positions in the sequence."""
        doc_ids: dict[str, list[int]] = {}
        term_freqs: dict[str, list[int]] = {}
        lengths = np.zeros(len(documents), dtype=np.float32)

        for doc_id, text in enumerate(documents):
            tokens = self.tokenizer(text)
            lengths[doc_id] = len(tokens)
            for term, freq in Counter(tokens).items():
                doc_ids.setdefault(term, []).append(doc_id)
                term_freqs.setdefault(term, []).append(freq)

        self.num_docs = len(documents)
        self._doc_len = lengths
        self.avgdl = float(lengths.mean()) if self.num_docs else 0.0
        self._postings = {
            term: (np.asarray(ids, dtype=np.int32), np.asarray(term_freqs[term], dtype=np.float32))
            for term, ids in doc_ids.items()
        }
        self._idf = {
            term: math.log(1.0 + (self.num_docs - len(ids) + 0.5) / (len(ids) + 0.5))
            for term, (ids, _) in self._postings.items()
        }
        return self

    def get_scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query`` (zeros where no term matches)."""
        scores = np.zeros(self.num_docs, dtype=np.float32)
        if not self.num_docs or self.avgdl == 0.0:
            return scores
        for term in set(self.tokenizer(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            ids, tf = posting
            norm = self.k1 * (1.0 - self.b + self.b * self._doc_len[ids] / self.avgdl)
            scores[ids] += self._idf[term] * tf * (self.k1 + 1.0) / (tf + norm)
        return scores

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Top-``top_k`` ``(doc_id, score)`` pairs with a strictly positive score."""
        if top_k <= 0:
            return []
        scores = self.get_scores(query)
        positive = np.flatnonzero(scores > 0)
        if positive.size == 0:
            return []
        if positive.size > top_k:
            part = np.argpartition(scores[positive], -top_k)[-top_k:]
            positive = positive[part]
        ranked = positive[np.argsort(-scores[positive], kind="stable")]
        return [(int(i), float(scores[i])) for i in ranked]
