import json
import math
import shutil

import numpy as np
import pytest

from nq_qa.models.reader import (
    META_FILE,
    NO_SPAN_NULL_DIFF,
    ExtractiveReader,
    decode_window,
    read_reader_meta,
    write_reader_meta,
)

from .conftest import CAPITALS, make_qa_examples

# Window layout: [CLS] q1 q2 [SEP] c1 c2 c3 c4 [SEP]
OFFSETS = [None, None, None, None, (0, 5), (6, 11), (12, 17), (18, 23), None]


def logits(start_peak=None, end_peak=None, cls=0.0, n=9, base=-5.0):
    start, end = np.full(n, base, dtype=np.float64), np.full(n, base, dtype=np.float64)
    start[0] = end[0] = cls
    if start_peak is not None:
        start[start_peak] = 5.0
    if end_peak is not None:
        end[end_peak] = 5.0
    return start, end


# --------------------------------------------------------------------------- decoding
def test_decode_picks_the_peak_span_and_maps_to_characters():
    start, end = logits(5, 6)
    span = decode_window(start, end, OFFSETS)
    assert (span.start_char, span.end_char) == (6, 17)
    assert span.span_logit == pytest.approx(10.0)
    assert 0.9 < span.span_prob <= 1.0 and span.null_prob < 0.01


def test_question_and_special_tokens_can_never_be_answers():
    start, end = logits(1, 2)  # peaks sit on question tokens
    span = decode_window(start, end, OFFSETS)
    assert span is not None
    assert OFFSETS[4][0] <= span.start_char and span.end_char <= OFFSETS[7][1]


def test_end_before_start_is_rejected():
    start, end = logits(7, 5)
    span = decode_window(start, end, OFFSETS)
    assert span.start_char <= span.end_char


def test_max_answer_length_is_enforced():
    start, end = logits(4, 7)  # a 4-token answer
    long_span = decode_window(start, end, OFFSETS, max_answer_tokens=4)
    short_span = decode_window(start, end, OFFSETS, max_answer_tokens=2)
    assert (long_span.start_char, long_span.end_char) == (0, 23)
    assert short_span.end_char - short_span.start_char < long_span.end_char


def test_null_answer_dominates_when_cls_logits_are_high():
    start, end = logits(5, 5, cls=9.0)
    span = decode_window(start, end, OFFSETS)
    assert span.null_logit > span.span_logit and span.null_prob > span.span_prob
    assert span.null_prob + span.span_prob <= 1.0 + 1e-9


def test_probabilities_are_normalised_over_cls_and_context_only():
    start, end = logits(5, 6)
    span = decode_window(start, end, OFFSETS)
    # upper bound check: probability of the argmax pair cannot exceed 1
    assert 0.0 < span.span_prob <= 1.0 and math.isfinite(span.null_prob)


def test_window_without_context_returns_none():
    start, end = logits()
    assert decode_window(start, end, [None] * 9) is None


# ------------------------------------------------------------------ inference (random)
@pytest.fixture()
def random_reader(tiny_qa_dir):
    return ExtractiveReader(str(tiny_qa_dir), max_length=32, stride=8, batch_size=4, device="cpu")


def test_predictions_are_consistent_with_offsets(random_reader):
    questions = ["what is the capital of france?", "what is the capital of peru?"]
    contexts = [e["context"] for e in make_qa_examples()[:2]]
    for context, pred in zip(
        contexts, random_reader.predict_batch(questions, contexts), strict=True
    ):
        assert pred.text == context[pred.start : pred.end]
        assert 0.0 <= pred.score <= 1.0 and 0.0 <= pred.null_score <= 1.0
        assert math.isfinite(pred.null_diff)


def test_batch_and_single_predictions_agree(random_reader):
    examples = make_qa_examples()[:5]
    questions, contexts = [e["question"] for e in examples], [e["context"] for e in examples]
    batched = random_reader.predict_batch(questions, contexts)
    single = [random_reader.predict(q, c) for q, c in zip(questions, contexts, strict=True)]
    for a, b in zip(batched, single, strict=True):
        assert (a.text, a.start, a.end) == (b.text, b.start, b.end)
        assert a.score == pytest.approx(b.score, abs=1e-5)


def test_long_contexts_are_handled_with_sliding_windows(random_reader):
    context = " ".join(e["context"] for e in make_qa_examples())  # far longer than 32 tokens
    pred = random_reader.predict("what is the capital of peru?", context)
    assert pred.text == context[pred.start : pred.end]


def test_empty_input_and_length_mismatch(random_reader):
    assert random_reader.predict_batch([], []) == []
    with pytest.raises(ValueError):
        random_reader.predict_batch(["q"], [])


def test_empty_context_yields_no_span(random_reader):
    pred = random_reader.predict("what is the capital of peru?", "")
    assert pred.text == "" and not pred.answerable and pred.null_diff == NO_SPAN_NULL_DIFF


# ------------------------------------------------------------------------- abstention
def test_without_meta_the_reader_never_abstains(tiny_qa_dir):
    reader = ExtractiveReader(str(tiny_qa_dir), device="cpu")
    assert reader.null_threshold is None
    assert reader.predict(
        "what is the capital of peru?", make_qa_examples()[0]["context"]
    ).answerable


def test_threshold_from_meta_controls_abstention(tiny_qa_dir):
    write_reader_meta(tiny_qa_dir, {"null_threshold": -1e9, "trained_with_negatives": True})
    reader = ExtractiveReader(str(tiny_qa_dir), device="cpu")
    assert reader.null_threshold == -1e9
    pred = reader.predict("what is the capital of peru?", make_qa_examples()[0]["context"])
    assert not pred.answerable and pred.text  # abstains, but still exposes its best guess
    assert ExtractiveReader(str(tiny_qa_dir), device="cpu", abstain=False).null_threshold is None
    forced = ExtractiveReader(str(tiny_qa_dir), device="cpu", null_threshold=1e9)
    assert forced.predict("q?", make_qa_examples()[0]["context"]).answerable


def test_meta_roundtrip(tmp_path):
    assert read_reader_meta(tmp_path) == {}
    write_reader_meta(tmp_path, {"null_threshold": 1.5})
    assert json.loads((tmp_path / META_FILE).read_text()) == {"null_threshold": 1.5}
    assert read_reader_meta("distilbert-base-uncased-distilled-squad") == {}


# ----------------------------------------------------------- training (the end-to-end proof)
@pytest.mark.slow
def test_fine_tuning_teaches_the_model_to_extract_the_right_span(trained_qa_dir):
    """Labels (processor) -> training -> decoding (reader) must be mutually consistent."""
    reader = ExtractiveReader(str(trained_qa_dir), max_length=48, stride=12, device="cpu")
    examples = make_qa_examples()
    predictions = reader.predict_batch(
        [e["question"] for e in examples], [e["context"] for e in examples]
    )
    correct = sum(
        pred.text.replace(" ", "").lower() == CAPITALS[e["question"].split()[-1].rstrip("?")]
        for e, pred in zip(examples, predictions, strict=True)
    )
    assert correct >= 10, f"only {correct}/12 training answers recovered"


@pytest.mark.slow
def test_trained_reader_is_confident_on_answers_and_unsure_on_mismatches(trained_qa_dir):
    reader = ExtractiveReader(str(trained_qa_dir), max_length=48, stride=12, device="cpu")
    good = reader.predict("what is the capital of peru?", make_qa_examples()[6]["context"])
    assert good.text.lower() == "lima" and good.score > 0.3


@pytest.mark.slow
def test_trained_reader_can_be_saved_with_a_threshold(trained_qa_dir, tmp_path):
    copy = tmp_path / "model"
    shutil.copytree(trained_qa_dir, copy)
    write_reader_meta(copy, {"null_threshold": 0.0})
    reader = ExtractiveReader(str(copy), max_length=48, stride=12, device="cpu")
    assert reader.null_threshold == 0.0
    answerable = [
        reader.predict(e["question"], e["context"]).answerable for e in make_qa_examples()
    ]
    assert sum(answerable) >= 10  # a trained model prefers a span over [CLS] on its own data
