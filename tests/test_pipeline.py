import pytest

from nq_qa.data.passage import Passage
from nq_qa.models.reader import ExtractiveReader, SpanPrediction
from nq_qa.models.retriever import HybridRetriever
from nq_qa.pipeline.e2e_qa import (
    EndToEndQAPipeline,
    calibrate_retriever_score,
    min_max_normalize,
)

from .conftest import CAPITALS

PASSAGES = [
    Passage(
        id=f"p-{country}",
        text=f"the capital of {country} is {city} and it is a large city near the coast.",
    )
    for country, city in CAPITALS.items()
]


class FakeReader:
    """Returns a scripted span for each passage text; records how it was called."""

    model_name_or_path = "fake-reader"
    null_threshold = None

    def __init__(self, scripted):
        self.scripted = scripted
        self.calls = []

    def predict_batch(self, questions, contexts):
        self.calls.append((list(questions), list(contexts)))
        return [self.scripted(context) for context in contexts]


def span(text="x", score=0.5, answerable=True):
    return SpanPrediction(text, 0, len(text), score, 0.1, -1.0 if answerable else 3.0, answerable)


@pytest.fixture()
def retriever():
    r = HybridRetriever(encoder=None, mode="bm25")
    r.index = r.build_index(PASSAGES)
    return r


def test_min_max_normalize():
    assert min_max_normalize([2.0, 4.0, 3.0]) == [0.0, 1.0, 0.5]
    assert min_max_normalize([5.0]) == [1.0]
    assert min_max_normalize([3.0, 3.0]) == [1.0, 1.0]
    assert min_max_normalize([]) == []


def test_calibrate_retriever_score_prevents_forcing_weak_matches_to_one():
    # Weak BM25 score (0.8) must NOT be calibrated to 1.0
    weak_calib = calibrate_retriever_score(0.8, mode="bm25")
    assert weak_calib < 0.1, f"Expected weak BM25 score to be < 0.1, got {weak_calib}"

    # Decent BM25 score (10.0) gives 0.5
    assert calibrate_retriever_score(10.0, mode="bm25") == pytest.approx(0.5)

    # Strong BM25 score (30.0) gives 0.75
    assert calibrate_retriever_score(30.0, mode="bm25") == pytest.approx(0.75)

    # Hybrid RRF score
    assert calibrate_retriever_score(2.0 / 61.0, mode="hybrid", rrf_k=60) == pytest.approx(1.0)
    assert calibrate_retriever_score(0.0, mode="hybrid") == 0.0

    # Negative or 0 score gives 0.0
    assert calibrate_retriever_score(-1.0, mode="bm25") == 0.0


def test_pipeline_with_calibrated_scoring(retriever):
    reader = FakeReader(lambda context: span("ans", score=0.8))
    pipeline = EndToEndQAPipeline(
        retriever, reader, alpha_reader_weight=0.7, score_calibration="calibrated"
    )
    result = pipeline.answer_question("what is the capital of peru", top_k=3)
    top = result["candidates"][0]
    assert 0.0 < top["retriever_score"] <= 1.0
    assert top["retriever_score"] == pytest.approx(
        calibrate_retriever_score(top["sparse_score"], "bm25")
    )


class FakeReranker:
    model_name_or_path = "fake-reranker"

    def rerank(self, query, passages, top_k=5):
        # Reverse passages to simulate re-ordering
        rev = list(reversed(passages))[:top_k]
        for i, p in enumerate(rev, 1):
            p.rank = i
            p.score = 5.0 - i
        return rev


def test_pipeline_with_reranker(retriever):
    reader = FakeReader(lambda context: span("ans", score=0.9))
    pipeline = EndToEndQAPipeline(retriever, reader, reranker=FakeReranker(), candidate_pool=10)
    result = pipeline.answer_question("capital of peru", top_k=2)
    assert len(result["candidates"]) == 2
    assert "rerank_ms" in result
    assert result["candidates"][0]["retriever_score"] == 1.0


def test_alpha_must_be_a_probability(retriever):
    with pytest.raises(ValueError):
        EndToEndQAPipeline(retriever, FakeReader(lambda c: span()), alpha_reader_weight=1.5)


def test_combined_score_blends_reader_probability_and_normalised_retrieval(retriever):
    reader = FakeReader(lambda context: span("ans", score=0.8))
    pipeline = EndToEndQAPipeline(retriever, reader, alpha_reader_weight=0.7)
    result = pipeline.answer_question("what is the capital of peru", top_k=3)
    top = result["candidates"][0]
    assert top["retriever_score"] == 1.0  # best retrieved passage after min-max
    assert top["combined_score"] == pytest.approx(0.7 * 0.8 + 0.3 * 1.0)
    assert result["candidates"][-1]["retriever_score"] == 0.0
    combined = [c["combined_score"] for c in result["candidates"]]
    assert combined == sorted(combined, reverse=True)


def test_reader_receives_all_passages_in_one_batch(retriever):
    reader = FakeReader(lambda context: span())
    EndToEndQAPipeline(retriever, reader).answer_question("capital of peru", top_k=4)
    assert len(reader.calls) == 1
    questions, contexts = reader.calls[0]
    assert len(questions) == len(contexts) == 4 and set(questions) == {"capital of peru"}


def test_the_better_answer_from_a_lower_ranked_passage_can_win(retriever):
    def scripted(context):
        return span("lima", score=0.95) if "lima" in context else span("junk", score=0.05)

    pipeline = EndToEndQAPipeline(retriever, FakeReader(scripted), alpha_reader_weight=0.9)
    result = pipeline.answer_question("capital city coast", top_k=12)
    assert result["best_answer"] == "lima" and "lima" in result["best_context"]


def test_answerable_candidates_outrank_abstentions_even_with_higher_scores(retriever):
    def scripted(context):
        if "peru" in context:
            return span("unsure", score=0.99, answerable=False)
        return span("ok", score=0.2, answerable=True)

    result = EndToEndQAPipeline(retriever, FakeReader(scripted)).answer_question(
        "capital of peru", top_k=3
    )
    assert result["candidates"][0]["answerable"] and result["answerable"]
    assert result["candidates"][-1]["answer"] == "unsure"


def test_pipeline_abstains_when_no_passage_is_answerable(retriever):
    pipeline = EndToEndQAPipeline(
        retriever, FakeReader(lambda c: span("maybe", 0.9, answerable=False))
    )
    result = pipeline.answer_question("capital of peru", top_k=3)
    assert result["answerable"] is False
    assert (
        result["best_answer"] == "" and result["best_score"] == 0.0 and result["best_context"] == ""
    )
    assert len(result["candidates"]) == 3  # evidence is still returned for inspection


def test_result_contains_timings_and_mode(retriever):
    result = EndToEndQAPipeline(retriever, FakeReader(lambda c: span())).answer_question(
        "capital of peru"
    )
    assert result["mode"] == "bm25"
    assert result["latency_ms"] >= result["retrieval_ms"] >= 0 and result["reader_ms"] >= 0


def test_answer_from_contexts_never_touches_the_shared_index(retriever):
    shared = retriever.index
    reader = FakeReader(lambda c: span("pdf answer", 0.9))
    pipeline = EndToEndQAPipeline(retriever, reader)
    chunks = [
        "Invoices are payable within thirty days of receipt of the goods.",
        "The warranty covers manufacturing defects for twenty four months.",
        "tiny",  # too short to be a usable chunk
    ]
    result = pipeline.answer_from_contexts("how long is the warranty", chunks, top_k=2)
    assert retriever.index is shared and len(shared) == len(PASSAGES)
    contexts = reader.calls[0][1]
    assert set(contexts) <= set(chunks[:2]) and result["candidates"][0]["doc_id"].startswith(
        "chunk_"
    )
    assert "warranty" in result["candidates"][0]["context"]


def test_answer_from_contexts_with_no_usable_text(retriever):
    pipeline = EndToEndQAPipeline(retriever, FakeReader(lambda c: span()))
    with pytest.raises(ValueError, match="No usable text"):
        pipeline.answer_from_contexts("question here", ["a", "  ", "short"])


def test_describe_reports_loaded_components(retriever):
    info = EndToEndQAPipeline(retriever, FakeReader(lambda c: span())).describe()
    assert info["mode"] == "bm25" and info["indexed_passages"] == len(PASSAGES)
    assert info["reader_model"] == "fake-reader" and info["abstention_enabled"] is False
    assert info["available_modes"] == ["bm25"]


@pytest.mark.slow
def test_end_to_end_with_a_real_trained_reader(trained_qa_dir, retriever):
    reader = ExtractiveReader(str(trained_qa_dir), max_length=48, stride=12, device="cpu")
    pipeline = EndToEndQAPipeline(retriever, reader)
    correct = 0
    for country, city in CAPITALS.items():
        result = pipeline.answer_question(f"what is the capital of {country}?", top_k=3)
        correct += result["best_answer"].replace(" ", "").lower() == city
    assert correct >= 10
