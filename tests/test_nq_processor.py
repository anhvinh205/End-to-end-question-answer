import pytest

from nq_qa.data.nq_processor import NaturalQuestionsProcessor


def raw_record(**overrides):
    tokens = ["<P>", "In", "meteorology,", "precipitation", "is", "any", "product", "of", "the",
              "condensation", "of", "atmospheric", "water", "vapor", "that", "falls", "under",
              "gravity.", "</P>"]  # fmt: skip
    record = {
        "id": "nq_test_001",
        "question": {"text": "What causes precipitation to fall?"},
        "document": {"tokens": {"token": tokens, "is_html": [t.startswith("<") for t in tokens]}},
        "annotations": {
            "long_answer": [{"start_token": 0, "end_token": 19}],
            "short_answers": [{"start_token": [17], "end_token": [18]}],
        },
        "long_answer_candidates": {"start_token": [0], "end_token": [19], "top_level": [True]},
    }
    record.update(overrides)
    return record


def test_parse_raw_record_positive():
    parsed = NaturalQuestionsProcessor.parse_raw_nq_record(raw_record())
    assert parsed is not None and parsed["id"] == "nq_test_001" and parsed["has_answer"]
    start = parsed["answers"]["answer_start"][0]
    assert parsed["answers"]["text"] == ["gravity."]
    assert parsed["context"][start : start + 8] == "gravity."
    assert parsed["context_type"] == "P"


def test_parse_raw_record_skips_leading_non_answer_annotations():
    ann = {
        "long_answer": [{"start_token": -1, "end_token": -1}, {"start_token": 0, "end_token": 19}],
        "short_answers": [
            {"start_token": [], "end_token": []},
            {"start_token": [17], "end_token": [18]},
        ],
    }
    parsed = NaturalQuestionsProcessor.parse_raw_nq_record(raw_record(annotations=ann))
    assert parsed["answers"]["text"] == ["gravity."]


def test_parse_raw_record_without_long_answer_borrows_a_candidate():
    ann = {
        "long_answer": [{"start_token": -1, "end_token": -1}],
        "short_answers": [{"start_token": [], "end_token": []}],
    }
    parsed = NaturalQuestionsProcessor.parse_raw_nq_record(
        raw_record(annotations=ann), min_context_tokens=10
    )
    assert parsed is not None and not parsed["has_answer"]


def test_parse_raw_record_without_any_candidate_is_dropped():
    ann = {
        "long_answer": [{"start_token": -1, "end_token": -1}],
        "short_answers": [{"start_token": [], "end_token": []}],
    }
    cands = {"start_token": [], "end_token": [], "top_level": []}
    assert (
        NaturalQuestionsProcessor.parse_raw_nq_record(
            raw_record(annotations=ann, long_answer_candidates=cands)
        )
        is None
    )


def test_parse_raw_record_malformed_returns_none():
    assert NaturalQuestionsProcessor.parse_raw_nq_record({"question": {}}) is None


def test_stride_must_be_smaller_than_max_length(tiny_tokenizer):
    with pytest.raises(ValueError):
        NaturalQuestionsProcessor(tiny_tokenizer, max_length=32, stride=32)


def decoded_answer(tokenizer, features, i):
    ids = features["input_ids"][i]
    return tokenizer.decode(ids[features["start_positions"][i] : features["end_positions"][i] + 1])


def test_training_labels_point_at_the_answer(tiny_tokenizer, qa_examples):
    processor = NaturalQuestionsProcessor(tiny_tokenizer, max_length=64, stride=16)
    batch = {k: [e[k] for e in qa_examples] for k in ("id", "question", "context", "answers")}
    features = processor.preprocess_training_examples(batch)
    assert len(features["start_positions"]) == len(qa_examples)
    for i, example in enumerate(qa_examples):
        assert features["start_positions"][i] > 0
        assert (
            decoded_answer(tiny_tokenizer, features, i).replace(" ", "")
            == example["answers"]["text"][0]
        )


def test_unanswerable_example_points_at_cls(tiny_tokenizer):
    processor = NaturalQuestionsProcessor(tiny_tokenizer, max_length=64, stride=16)
    features = processor.preprocess_training_examples(
        {
            "question": ["what is the capital of peru?"],
            "context": ["the capital of france is paris and it is a large city."],
            "answers": [{"text": [], "answer_start": []}],
        }
    )
    assert features["start_positions"] == [0] and features["end_positions"] == [0]


def test_windows_without_the_answer_are_labelled_cls(tiny_tokenizer):
    processor = NaturalQuestionsProcessor(tiny_tokenizer, max_length=24, stride=4)
    filler = "the large city near the coast " * 6
    context = filler + "the capital of peru is lima."
    start = context.index("lima")
    features = processor.preprocess_training_examples(
        {
            "question": ["what is the capital of peru?"],
            "context": [context],
            "answers": [{"text": ["lima"], "answer_start": [start]}],
        }
    )
    starts = features["start_positions"]
    assert len(starts) > 2, "the passage should have been split into several windows"
    assert starts[0] == 0 and starts[-1] > 0
    assert sum(1 for s in starts if s > 0) >= 1


def test_validation_features_keep_ids_and_mask_question_offsets(tiny_tokenizer, qa_examples):
    processor = NaturalQuestionsProcessor(tiny_tokenizer, max_length=64, stride=16)
    batch = {k: [e[k] for e in qa_examples[:3]] for k in ("id", "question", "context", "answers")}
    features = processor.preprocess_validation_examples(batch)
    assert features["example_id"] == ["cap-0", "cap-1", "cap-2"]
    offsets = features["offset_mapping"][0]
    assert offsets[0] is None  # [CLS]
    assert any(o is not None for o in offsets)
    assert "overflow_to_sample_mapping" not in features


def test_iter_hf_nq_streams_parses_and_skips_malformed(monkeypatch):
    datasets = pytest.importorskip("datasets")
    stream = [raw_record(id="a"), {"broken": True}, raw_record(id="b"), raw_record(id="c")]
    seen = {}

    def fake_load_dataset(name, split, streaming):
        seen.update(name=name, split=split, streaming=streaming)
        return iter(stream)

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    records = list(NaturalQuestionsProcessor.iter_hf_nq(split="validation", max_samples=2))
    assert [r["id"] for r in records] == [
        "a",
        "b",
    ]  # malformed record skipped, max_samples respected
    assert seen == {
        "name": "google-research-datasets/natural_questions",
        "split": "validation",
        "streaming": True,
    }
