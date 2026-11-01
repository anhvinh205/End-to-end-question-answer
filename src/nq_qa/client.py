"""Backends used by the Streamlit UI: a thin HTTP client and an in-process wrapper.

Both expose the same three methods so the UI does not care whether it talks to the
FastAPI service (docker-compose) or loads the models itself (``make ui``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx


class QAClient:
    """HTTP client for the QA API."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def health(self) -> dict[str, Any]:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def ask(self, question: str, top_k: int = 5, mode: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question, "top_k": top_k}
        if mode:
            payload["mode"] = mode
        return self._post("/api/v1/qa", payload)

    def ask_documents(
        self, question: str, contexts: Sequence[str], top_k: int = 3
    ) -> dict[str, Any]:
        return self._post(
            "/api/v1/qa/documents",
            {"question": question, "contexts": list(contexts), "top_k": top_k},
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(path, json=payload)
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text) if response.content else ""
            raise RuntimeError(f"API error {response.status_code}: {detail}")
        return response.json()


class LocalBackend:
    """Same interface, but runs the pipeline inside the current process."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def health(self) -> dict[str, Any]:
        return {"status": "ok", **self.pipeline.describe()}

    def ask(self, question: str, top_k: int = 5, mode: str | None = None) -> dict[str, Any]:
        return self.pipeline.answer_question(question, top_k=top_k, mode=mode)

    def ask_documents(
        self, question: str, contexts: Sequence[str], top_k: int = 3
    ) -> dict[str, Any]:
        return self.pipeline.answer_from_contexts(question, contexts, top_k=top_k)
