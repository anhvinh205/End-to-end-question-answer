import pytest
from fastapi.testclient import TestClient

from nq_qa.api.main import create_app
from nq_qa.data.passage import Passage
from nq_qa.models.reader import SpanPrediction
from nq_qa.models.retriever import HybridRetriever
from nq_qa.pipeline.e2e_qa import EndToEndQAPipeline

from .conftest import CAPITALS


class FakeReader:
    model_name_or_path = "fake-reader"
    null_threshold = None

    def predict_batch(self, questions, contexts):
        return [SpanPrediction(c.split()[-1], 0, 1, 0.8, 0.05, -2.0, True) for c in contexts]


@pytest.fixture()
def pipeline():
    retriever = HybridRetriever(encoder=None, mode="bm25")
    retriever.index = retriever.build_index(
        [
            Passage(id=f"p{i}", text=f"the capital of {c} is {k} and it is a large city.")
            for i, (c, k) in enumerate(CAPITALS.items())
        ]
    )
    return EndToEndQAPipeline(retriever, FakeReader())


@pytest.fixture()
def client(pipeline):
    with TestClient(create_app(pipeline=pipeline)) as test_client:
        yield test_client


def test_health_reports_a_ready_pipeline(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["ready"] is True
    assert body["indexed_passages"] == 12 and body["available_modes"] == ["bm25"]
    assert body["reader_model"] == "fake-reader"


def test_health_before_startup_is_not_ready(pipeline):
    app = create_app(pipeline=pipeline)
    body = TestClient(app).get("/health").json()  # no `with` -> lifespan has not run
    assert body["ready"] is False and body["indexed_passages"] == 0


def test_qa_endpoint_returns_the_documented_schema(client):
    response = client.post(
        "/api/v1/qa", json={"question": "what is the capital of peru", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answerable"] is True and body["mode"] == "bm25"
    assert len(body["candidates"]) == 3
    candidate = body["candidates"][0]
    for key in (
        "rank",
        "doc_id",
        "context",
        "answer",
        "start",
        "end",
        "reader_score",
        "retriever_score",
        "combined_score",
        "answerable",
    ):
        assert key in candidate
    assert "peru" in body["best_context"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},  # too short
        {"question": "x" * 501},  # too long
        {"question": "valid question", "top_k": 0},
        {"question": "valid question", "top_k": 21},
        {"question": "valid question", "mode": "semantic"},
        {},
    ],
)
def test_invalid_requests_are_rejected_with_422(client, payload):
    assert client.post("/api/v1/qa", json=payload).status_code == 422


def test_unsupported_mode_is_a_client_error_not_a_crash(client):
    response = client.post("/api/v1/qa", json={"question": "capital of peru", "mode": "hybrid"})
    assert response.status_code == 400 and "dense index" in response.json()["detail"]


def test_document_endpoint_searches_only_the_supplied_text(client, pipeline):
    shared_before = pipeline.retriever.index
    response = client.post(
        "/api/v1/qa/documents",
        json={
            "question": "how long is the warranty",
            "contexts": [
                "The warranty covers defects for twenty four months.",
                "Invoices are payable in thirty days.",
            ],
            "top_k": 1,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert (
        body["candidates"][0]["doc_id"].startswith("chunk_") and "warranty" in body["best_context"]
    )
    assert pipeline.retriever.index is shared_before


def test_document_endpoint_validation_and_unusable_text(client):
    assert (
        client.post(
            "/api/v1/qa/documents", json={"question": "valid question", "contexts": []}
        ).status_code
        == 422
    )
    unusable = client.post(
        "/api/v1/qa/documents", json={"question": "valid question", "contexts": ["tiny"]}
    )
    assert unusable.status_code == 400 and "No usable text" in unusable.json()["detail"]


def test_requests_fail_with_503_while_the_pipeline_is_not_loaded(pipeline):
    app = create_app(pipeline=pipeline)
    response = TestClient(app).post("/api/v1/qa", json={"question": "capital of peru"})
    assert response.status_code == 503
