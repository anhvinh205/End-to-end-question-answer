import pytest

from nq_qa.evaluation.metrics import (
    answer_in_text,
    best_null_threshold,
    exact_match,
    f1_score,
    normalize_answer,
    retrieval_metrics,
    squad_v2_metrics,
)


def test_normalize_answer_follows_squad():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("An apple, a day.") == "apple day"
    assert normalize_answer("") == ""


def test_exact_match_uses_best_reference_and_normalisation():
    assert exact_match("the Tom Brady.", ["Tom Brady", "Brady"]) == 1.0
    assert exact_match("Brady", ["Tom Brady"]) == 0.0


def test_exact_match_for_unanswerable_requires_empty_prediction():
    assert exact_match("", []) == 1.0
    assert exact_match("something", []) == 0.0


def test_f1_partial_overlap():
    # prediction tokens {tom, brady, qb}; gold {tom, brady} -> P=2/3, R=1, F1=0.8
    assert f1_score("Tom Brady QB", ["Tom Brady"]) == pytest.approx(0.8)
    assert f1_score("Tom Brady", ["Tom Brady"]) == 1.0
    assert f1_score("Peyton", ["Tom Brady"]) == 0.0


def test_f1_takes_max_over_references_and_handles_empty():
    assert f1_score("Brady", ["Peyton Manning", "Brady"]) == 1.0
    assert f1_score("", []) == 1.0
    assert f1_score("x", []) == 0.0
    assert f1_score("", ["the"]) == 1.0  # "the" normalises to an empty answer


def test_squad_v2_metrics_breakdown():
    refs = {"a": ["Paris"], "b": ["Rome"], "c": []}
    preds = {"a": "paris", "b": "Milan", "c": ""}
    m = squad_v2_metrics(preds, refs)
    assert m["exact"] == pytest.approx(66.67, abs=0.01)
    assert m["HasAns_exact"] == 50.0 and m["NoAns_exact"] == 100.0
    assert m["total"] == 3 and m["HasAns_total"] == 2 and m["NoAns_total"] == 1


def test_missing_prediction_counts_as_empty():
    assert squad_v2_metrics({}, {"a": ["x"]})["exact"] == 0.0


def test_best_null_threshold_abstains_only_where_it_helps():
    # ex1/ex2 answerable & well answered (low null_diff); ex3/ex4 unanswerable (high null_diff)
    refs = {"1": ["a"], "2": ["b"], "3": [], "4": []}
    spans = {"1": "a", "2": "b", "3": "junk", "4": "junk"}
    diffs = {"1": -5.0, "2": -3.0, "3": 2.0, "4": 4.0}
    threshold, f1_tuned, f1_never = best_null_threshold(diffs, spans, refs)
    assert -3.0 <= threshold < 2.0
    assert f1_tuned == 100.0 and f1_never == 50.0


def test_best_null_threshold_never_abstains_when_it_hurts():
    refs = {"1": ["a"], "2": ["b"]}
    threshold, f1_tuned, f1_never = best_null_threshold(
        {"1": 1.0, "2": 2.0}, {"1": "a", "2": "b"}, refs
    )
    assert threshold > 2.0 and f1_tuned == f1_never == 100.0


def test_best_null_threshold_empty_input():
    assert best_null_threshold({}, {}, {}) == (0.0, 0.0, 0.0)


def test_answer_in_text_respects_word_boundaries():
    assert answer_in_text("The capital is Paris, France.", ["paris"])
    assert not answer_in_text("Parisian cuisine", ["Paris"])
    assert not answer_in_text("anything", [""])
    assert not answer_in_text("anything", [])


def test_retrieval_metrics():
    hits = [[True, False, False], [False, False, True], [False, False, False]]
    m = retrieval_metrics(hits, ks=(1, 3))
    assert m["recall@1"] == pytest.approx(0.3333, abs=1e-4)
    assert m["recall@3"] == pytest.approx(0.6667, abs=1e-4)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3, abs=1e-4)


def test_retrieval_metrics_no_queries():
    assert retrieval_metrics([], ks=(1,)) == {"recall@1": 0.0, "mrr": 0.0}
