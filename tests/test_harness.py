import pytest

from nq_qa.data.passage import Passage
from nq_qa.evaluation.harness import evaluate_end_to_end, evaluate_reader, evaluate_retrieval
from nq_qa.evaluation.report import render_markdown
from nq_qa.models.reader import ExtractiveReader
from nq_qa.models.retriever import HybridRetriever
from nq_qa.pipeline.e2e_qa import EndToEndQAPipeline

from .conftest import make_qa_examples


@pytest.fixture(scope="module")
def retriever():
    r = HybridRetriever(encoder=None, mode="bm25")
    r.index = r.build_index(
        [Passage(id=e["passage_id"], text=e["context"]) for e in make_qa_examples()]
    )
    return r


def test_retrieval_metrics_on_an_easy_corpus(retriever):
    results = evaluate_retrieval(
        retriever, retriever.index, make_qa_examples(), ["bm25"], ks=(1, 3)
    )
    metrics = results["bm25"]
    assert metrics["recall@1"] == 1.0 and metrics["recall@3"] == 1.0 and metrics["mrr"] == 1.0
    assert metrics["answer_recall@1"] == 1.0
    assert metrics["queries"] == 12 and metrics["avg_latency_ms"] >= 0


def test_retrieval_metrics_detect_misses(retriever):
    broken = [{**e, "passage_id": "does-not-exist"} for e in make_qa_examples()[:4]]
    metrics = evaluate_retrieval(retriever, retriever.index, broken, ["bm25"], ks=(1,))["bm25"]
    assert metrics["recall@1"] == 0.0 and metrics["answer_recall@1"] == 1.0  # answer is still found


@pytest.mark.slow
def test_reader_evaluation_and_threshold_tuning(trained_qa_dir):
    reader = ExtractiveReader(str(trained_qa_dir), max_length=48, stride=12, device="cpu")
    positives = make_qa_examples()
    # a negative: the question asks for peru but the passage is about france
    negatives = [
        {**positives[0], "id": "neg-1", "question": "what is the capital of peru?",
         "answers": {"text": [], "answer_start": []}, "has_answer": False},
    ]  # fmt: skip
    metrics = evaluate_reader(reader, positives + negatives)
    assert (
        metrics["HasAns_exact"] >= 80.0 and metrics["total"] == 13 and metrics["NoAns_total"] == 1
    )
    assert metrics["f1_at_best_threshold"] >= metrics["f1_without_abstention"]
    assert "best_null_threshold" in metrics


@pytest.mark.slow
def test_end_to_end_evaluation(trained_qa_dir, retriever):
    reader = ExtractiveReader(str(trained_qa_dir), max_length=48, stride=12, device="cpu")
    pipeline = EndToEndQAPipeline(retriever, reader)
    metrics = evaluate_end_to_end(pipeline, make_qa_examples(), top_k=3)
    assert metrics["exact"] >= 80.0 and metrics["f1"] >= metrics["exact"] and metrics["top_k"] == 3


def test_markdown_report_contains_every_section():
    results = {
        "meta": {"split": "validation", "num_queries": 10, "corpus_passages": 100, "encoder": "enc",
                 "reader": "rd", "device": "cpu", "generated_at": "now"},
        "retrieval": {"bm25": {"recall@1": 0.5, "recall@5": 0.8, "mrr": 0.6, "answer_recall@1": 0.55,
                               "answer_recall@5": 0.85, "avg_latency_ms": 1.0}},
        "reader": {"exact": 50.0, "f1": 60.0, "total": 10.0, "HasAns_exact": 55.0, "HasAns_f1": 65.0,
                   "HasAns_total": 8.0, "NoAns_exact": 30.0, "NoAns_total": 2.0, "best_null_threshold": 0.5,
                   "f1_at_best_threshold": 62.0, "f1_without_abstention": 60.0},
        "end_to_end": {"bm25": {"exact": 40.0, "f1": 50.0, "avg_latency_ms": 12.0, "top_k": 5.0}},
    }  # fmt: skip
    markdown = render_markdown(results)
    for fragment in (
        "## Retrieval",
        "| bm25 | 50.0 | 80.0 | 0.600",
        "## Reader on gold passages",
        "## End-to-end",
        "Best null threshold: 0.5",
    ):
        assert fragment in markdown


def test_markdown_report_with_only_retrieval():
    markdown = render_markdown(
        {
            "meta": {},
            "retrieval": {
                "bm25": {"recall@1": 1.0, "mrr": 1.0, "answer_recall@1": 1.0, "avg_latency_ms": 0.5}
            },
        }
    )
    assert "## Retrieval" in markdown and "## Reader" not in markdown
