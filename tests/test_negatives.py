import random

import pytest

from nq_qa.data.negatives import (
    build_training_records,
    is_yes_no_question,
    synthesize_hard_negatives,
)
from nq_qa.evaluation.metrics import answer_in_text


def rec(i, answerable=True, topic=None):
    topic = topic or f"topic{i}"
    context = f"{topic} is discussed here with plenty of words about {topic} and its history"
    return {
        "id": f"r{i}",
        "question": f"what is {topic}",
        "context": context,
        "passage_id": f"p{i}",
        "answers": {
            "text": [topic] if answerable else [],
            "answer_start": [0] if answerable else [],
        },
        "has_answer": answerable,
        "context_type": "P",
    }


def test_natural_negatives_are_used_first():
    records = [rec(i) for i in range(10)] + [rec(100 + i, answerable=False) for i in range(5)]
    out = build_training_records(records, negative_ratio=0.2, seed=1)
    negatives = [r for r in out if not r["has_answer"]]
    assert len(out) == 12 and len(negatives) == 2  # 10 positives -> 2.5 -> round(2.5)=2 negatives
    assert all(not r["id"].endswith("hardneg") for r in negatives)


def test_hard_negatives_fill_the_gap_and_never_leak_the_answer():
    records = [
        {
            **rec(i, topic=f"alpha{i}"),
            "question": f"what is alpha{i} beta common",
            "context": f"alpha{i} is a thing. beta common words shared text {i}",
        }
        for i in range(30)
    ]
    out = build_training_records(records, negative_ratio=0.3, seed=2)
    hard = [r for r in out if r["id"].endswith("hardneg")]
    assert hard, "expected synthesised hard negatives"
    by_id = {r["id"]: r for r in records}
    for neg in hard:
        positive = by_id[neg["id"].removesuffix("-hardneg")]
        assert neg["passage_id"] != positive["passage_id"]
        assert not answer_in_text(neg["context"], positive["answers"]["text"])
        assert not neg["has_answer"] and neg["answers"]["text"] == []
    share = sum(not r["has_answer"] for r in out) / len(out)
    assert share == pytest.approx(0.3, abs=0.05)


def test_zero_ratio_returns_positives_only():
    records = [rec(i) for i in range(5)] + [rec(50, answerable=False)]
    assert all(r["has_answer"] for r in build_training_records(records, negative_ratio=0.0))


def test_deterministic_for_a_seed():
    records = [rec(i) for i in range(20)]
    a = build_training_records(records, 0.3, seed=5)
    b = build_training_records(records, 0.3, seed=5)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_invalid_ratio():
    with pytest.raises(ValueError):
        build_training_records([rec(1)], negative_ratio=1.0)


def test_only_negatives_available():
    records = [rec(i, answerable=False) for i in range(3)]
    assert len(build_training_records(records, 0.3)) == 3


def test_synthesize_with_no_positives_or_zero_n():
    assert synthesize_hard_negatives([], [rec(1)], 5, random.Random(0)) == []
    assert synthesize_hard_negatives([rec(1)], [rec(1)], 0, random.Random(0)) == []


def test_is_yes_no_question():
    assert is_yes_no_question("Is water wet?")
    assert is_yes_no_question("Can you hear me now")
    assert is_yes_no_question("Did he play the piano")
    assert not is_yes_no_question("Who played the incredible hulk")
    assert not is_yes_no_question("Where does the new movie take place")
    assert not is_yes_no_question("")


def test_filter_yes_no_negatives():
    positives = [rec(1, answerable=True), rec(10, answerable=True)]
    yn_neg = {**rec(2, answerable=False), "question": "Is the earth flat"}
    normal_neg = {**rec(3, answerable=False), "question": "who was the king"}

    # With filtering: yn_neg is skipped from natural pool
    filtered = build_training_records(
        [*positives, yn_neg, normal_neg], negative_ratio=0.5, filter_yes_no=True
    )
    assert not any(r["id"] == "r2" for r in filtered)

    # Without filtering: yn_neg is kept
    unfiltered = build_training_records(
        [*positives, yn_neg, normal_neg], negative_ratio=0.5, filter_yes_no=False
    )
    assert any(r["id"] == "r2" for r in unfiltered)
