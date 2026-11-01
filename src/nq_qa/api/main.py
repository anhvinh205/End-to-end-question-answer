"""REST API for the end-to-end QA pipeline.

Endpoints are plain ``def`` functions on purpose: inference is CPU/GPU bound and blocking,
so FastAPI runs it in its worker thread-pool instead of stalling the event loop.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from nq_qa import __version__
from nq_qa.pipeline.e2e_qa import EndToEndQAPipeline
from nq_qa.utils.config import load_config
from nq_qa.utils.logging import setup_logging

logger = logging.getLogger(__name__)

RetrievalMode = Literal["bm25", "dense", "hybrid"]


class QARequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="Question in English")
    top_k: int = Field(default=5, ge=1, le=20, description="Passages to retrieve and read")
    mode: RetrievalMode | None = Field(
        default=None, description="Override the configured retrieval mode for this request"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "who invented the world wide web", "top_k": 5}]
        }
    }


class DocumentQARequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    contexts: list[str] = Field(
        ..., min_length=1, max_length=200, description="Text chunks to search (e.g. PDF pages)"
    )
    top_k: int = Field(default=3, ge=1, le=20)


class CandidateResponse(BaseModel):
    rank: int = Field(description="Retrieval rank of the passage (1 = best)")
    doc_id: str
    context: str
    answer: str
    start: int
    end: int
    answerable: bool
    reader_score: float
    retriever_score: float = Field(description="Retriever score normalised to [0, 1]")
    combined_score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    matched_question: str = ""


class QAResponse(BaseModel):
    question: str
    answerable: bool = Field(
        description="False when no retrieved passage contains a confident answer"
    )
    best_answer: str
    best_score: float
    best_context: str
    mode: str
    latency_ms: float
    retrieval_ms: float
    reader_ms: float
    candidates: list[CandidateResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    ready: bool
    mode: str | None = None
    available_modes: list[str] = []
    indexed_passages: int = 0
    reader_model: str | None = None
    abstention_enabled: bool = False


def get_pipeline(request: Request) -> EndToEndQAPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="QA pipeline is not loaded yet.")
    return pipeline


PipelineDep = Annotated[EndToEndQAPipeline, Depends(get_pipeline)]


def create_app(
    pipeline: EndToEndQAPipeline | None = None, config_path: str | None = None
) -> FastAPI:
    """Application factory.

    Pass ``pipeline`` to inject a ready-made (or fake) pipeline - this is how the tests run
    without downloading any model.  Otherwise the pipeline is built from the config at start-up.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            from nq_qa.pipeline.factory import build_pipeline  # heavy imports only when needed

            setup_logging()
            app.state.pipeline = build_pipeline(load_config(config_path))
        logger.info("QA pipeline ready: %s", app.state.pipeline.describe())
        yield

    app = FastAPI(
        title="Google Natural Questions - End-to-End QA API",
        description="Hybrid retrieval (FAISS + BM25 + RRF) and an extractive Transformer reader.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.pipeline = None

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> Any:
        loaded = getattr(request.app.state, "pipeline", None)
        if loaded is None:
            return HealthResponse(status="ok", version=__version__, ready=False)
        info = loaded.describe()
        return HealthResponse(
            status="ok",
            version=__version__,
            ready=True,
            mode=info["mode"],
            available_modes=info["available_modes"],
            indexed_passages=info["indexed_passages"],
            reader_model=info["reader_model"],
            abstention_enabled=info["abstention_enabled"],
        )

    @app.post("/api/v1/qa", response_model=QAResponse, tags=["qa"])
    def answer_question(req: QARequest, qa: PipelineDep) -> Any:
        """Answer a question from the Natural Questions knowledge base."""
        try:
            return qa.answer_question(req.question, top_k=req.top_k, mode=req.mode)
        except ValueError as exc:  # e.g. a mode the loaded index does not support
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/qa/documents", response_model=QAResponse, tags=["qa"])
    def answer_from_documents(req: DocumentQARequest, qa: PipelineDep) -> Any:
        """Answer over text supplied in the request; the shared knowledge base is untouched."""
        try:
            return qa.answer_from_contexts(req.question, req.contexts, top_k=req.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
