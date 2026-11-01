"""QA and retrieval metrics implemented from their definitions.

No ``evaluate`` / network dependency: the SQuAD normalisation, Exact Match and token-level
F1 are a few lines each, easy to unit-test and identical to the official SQuAD v2 script.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Collection, Mapping, Sequence

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation and articles, collapse whitespace (SQuAD convention)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _f1_single(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(reference).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, references: Sequence[str]) -> float:
    """1.0 if ``prediction`` equals any reference after normalisation.

    With no references (an unanswerable question) the only correct output is an empty one.
    """
    pred = normalize_answer(prediction)
    if not references:
        return float(pred == "")
    return float(any(pred == normalize_answer(ref) for ref in references))


def f1_score(prediction: str, references: Sequence[str]) -> float:
    """Best token-level F1 against any reference (empty-vs-empty counts as 1.0)."""
    if not references:
        return float(normalize_answer(prediction) == "")
    return max(_f1_single(prediction, ref) for ref in references)


def squad_v2_metrics(
    predictions: Mapping[str, str], references: Mapping[str, Sequence[str]]
) -> dict[str, float]:
    """EM / F1 (in %) overall and split into answerable (HasAns) / unanswerable (NoAns)."""
    exact: list[float] = []
    f1: list[float] = []
    has_exact: list[float] = []
    has_f1: list[float] = []
    no_exact: list[float] = []
    for key, refs in references.items():
        pred = predictions.get(key, "")
        em, f = exact_match(pred, refs), f1_score(pred, refs)
        exact.append(em)
        f1.append(f)
        (has_exact if refs else no_exact).append(em)
        if refs:
            has_f1.append(f)

    def pct(values: Sequence[float]) -> float:
        return round(100.0 * sum(values) / len(values), 2) if values else 0.0

    return {
        "exact": pct(exact),
        "f1": pct(f1),
        "total": float(len(exact)),
        "HasAns_exact": pct(has_exact),
        "HasAns_f1": pct(has_f1),
        "HasAns_total": float(len(has_exact)),
        "NoAns_exact": pct(no_exact),
        "NoAns_total": float(len(no_exact)),
    }


def best_null_threshold(
    null_diffs: Mapping[str, float],
    best_spans: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> tuple[float, float, float]:
    """Pick the abstention threshold that maximises mean F1 (SQuAD v2 ``best_f1_thresh``).

    The reader abstains when ``null_diff > threshold``.  Returns
    ``(threshold, f1_at_threshold, f1_without_abstention)`` with F1 in %.
    """
    keys = [k for k in references if k in null_diffs]
    if not keys:
        return 0.0, 0.0, 0.0
    answered = {k: f1_score(best_spans.get(k, ""), references[k]) for k in keys}
    abstained = {k: f1_score("", references[k]) for k in keys}
    base = sum(answered.values())

    order = sorted(keys, key=lambda k: null_diffs[k], reverse=True)  # most "null-ish" first
    best_gain, best_cut, running = 0.0, -1, 0.0
    for cut, key in enumerate(order):
        running += abstained[key] - answered[key]
        if running > best_gain + 1e-12:
            best_gain, best_cut = running, cut

    if best_cut < 0:  # abstaining never helps
        threshold = max(null_diffs[k] for k in keys) + 1.0
    else:
        upper = null_diffs[order[best_cut]]
        lower = null_diffs[order[best_cut + 1]] if best_cut + 1 < len(order) else upper - 2.0
        threshold = (upper + lower) / 2.0
    n = len(keys)
    return float(threshold), round(100.0 * (base + best_gain) / n, 2), round(100.0 * base / n, 2)


# ----------------------------------------------------------------------- retrieval
def answer_in_text(text: str, answers: Collection[str]) -> bool:
    """True if any answer occurs in ``text`` on word boundaries (after normalisation)."""
    haystack = f" {normalize_answer(text)} "
    return any(norm and f" {norm} " in haystack for norm in (normalize_answer(a) for a in answers))


def retrieval_metrics(
    hits_per_query: Sequence[Sequence[bool]], ks: Sequence[int] = (1, 5, 10, 20)
) -> dict[str, float]:
    """Recall@k and MRR from per-query relevance flags ordered by rank.

    ``hits_per_query[i][r]`` says whether the passage at rank ``r`` (0-based) is relevant
    for query ``i``.  Recall@k is the fraction of queries with a relevant passage in the
    top k; MRR is computed over the deepest ranking supplied.
    """
    n = len(hits_per_query)
    if n == 0:
        return {**{f"recall@{k}": 0.0 for k in ks}, "mrr": 0.0}
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = round(sum(any(h[:k]) for h in hits_per_query) / n, 4)
    reciprocal = 0.0
    for hits in hits_per_query:
        for rank, hit in enumerate(hits, start=1):
            if hit:
                reciprocal += 1.0 / rank
                break
    metrics["mrr"] = round(reciprocal / n, 4)
    return metrics
