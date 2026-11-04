"""Passage retrieval: dense (FAISS), sparse (BM25) and their Reciprocal-Rank-Fusion.

Three small pieces keep the responsibilities apart:

* :class:`DenseEncoder`  - turns text into vectors (``[CLS]`` or mean pooling).
* :class:`PassageIndex`  - the *data*: passages + FAISS index + BM25 index, persistable.
* :class:`HybridRetriever` - the *logic*: encodes the query, searches an index and fuses.

Because the index is a separate object, answering a question over an uploaded PDF builds
a throw-away :class:`PassageIndex` and never touches the shared knowledge base.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from nq_qa import __version__
from nq_qa.data.passage import Passage
from nq_qa.models.bm25 import BM25Index

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = 1
INDEX_TYPES = ("FlatL2", "FlatIP", "IVFFlat")
RETRIEVAL_MODES = ("bm25", "dense", "hybrid")
MIN_VECTORS_PER_CENTROID = 39


class IndexMismatchError(RuntimeError):
    """The persisted index was built with a different encoder than the one configured."""


@dataclass(slots=True)
class RetrievedPassage:
    rank: int
    doc_id: str
    text: str
    score: float  # fused score (RRF) for hybrid, raw score otherwise
    dense_score: float | None = None  # similarity, higher is better
    sparse_score: float | None = None  # BM25
    question: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- encoder
class DenseEncoder:
    """Transformer bi-encoder producing one vector per text.

    ``pooling="cls"`` reproduces the vanilla DistilBERT baseline (``last_hidden_state[:, 0]``),
    ``pooling="mean"`` is the masked mean pooling used by sentence-transformers models such
    as ``multi-qa-MiniLM-L6-cos-v1``.
    """

    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = "mean",
        normalize: bool = True,
        max_length: int = 256,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.model_name_or_path = model_name_or_path
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device).eval()

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def spec(self) -> dict[str, Any]:
        """Identity of the encoder, stored in the index manifest."""
        return {
            "model": self.model_name_or_path,
            "pooling": self.pooling,
            "normalize": self.normalize,
            "max_length": self.max_length,
        }

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    @torch.inference_mode()
    def encode(
        self, texts: Sequence[str], batch_size: int | None = None, show_progress: bool = False
    ) -> np.ndarray:
        """Encode ``texts`` into a ``(len(texts), dim)`` float32 array (input order kept)."""
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch_size = batch_size or self.batch_size
        # Length-sorted batches waste far less padding (large speed-up on CPU).
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        starts: Iterable[int] = range(0, len(order), batch_size)
        if show_progress:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc="Encoding", unit="batch")

        chunks: list[np.ndarray] = []
        for start in starts:
            batch = [texts[i] for i in order[start : start + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            hidden = self.model(**encoded).last_hidden_state
            vectors = self._pool(hidden, encoded["attention_mask"])
            if self.normalize:
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            chunks.append(vectors.float().cpu().numpy())

        sorted_vectors = np.vstack(chunks)
        output = np.empty_like(sorted_vectors)
        output[order] = sorted_vectors
        return np.ascontiguousarray(output, dtype=np.float32)


# ----------------------------------------------------------------------------- index
class PassageIndex:
    """Passages with a FAISS index (optional) and a BM25 index."""

    def __init__(
        self,
        passages: list[Passage],
        bm25: BM25Index,
        faiss_index: faiss.Index | None = None,
        index_type: str | None = None,
        encoder_spec: dict[str, Any] | None = None,
    ) -> None:
        self.passages = passages
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.index_type = index_type
        self.encoder_spec = encoder_spec

    def __len__(self) -> int:
        return len(self.passages)

    @property
    def has_dense(self) -> bool:
        return self.faiss_index is not None

    @property
    def metric(self) -> str | None:
        if self.faiss_index is None:
            return None
        return "ip" if self.faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT else "l2"

    # -- construction ------------------------------------------------------------
    @classmethod
    def build(
        cls,
        passages: Sequence[Passage],
        encoder: DenseEncoder | None = None,
        index_type: str = "FlatIP",
        nlist: int = 256,
        nprobe: int = 8,
        show_progress: bool = False,
    ) -> PassageIndex:
        """Index ``passages`` (duplicated texts are dropped). ``encoder=None`` -> BM25 only."""
        if index_type not in INDEX_TYPES:
            raise ValueError(f"index_type must be one of {INDEX_TYPES}, got {index_type!r}")
        unique: list[Passage] = []
        seen: set[str] = set()
        for passage in passages:
            if passage.text not in seen:
                seen.add(passage.text)
                unique.append(passage)
        if not unique:
            raise ValueError("Cannot build an index from an empty list of passages.")

        texts = [p.text for p in unique]
        bm25 = BM25Index().fit(texts)

        faiss_index = None
        if encoder is not None:
            vectors = encoder.encode(texts, show_progress=show_progress)
            faiss_index = _make_faiss_index(vectors, index_type, encoder.normalize, nlist, nprobe)
            logger.info(
                "FAISS %s index with %d vectors (dim=%d)",
                index_type,
                len(vectors),
                vectors.shape[1],
            )
        return cls(
            unique,
            bm25,
            faiss_index,
            index_type if encoder is not None else None,
            encoder.spec if encoder is not None else None,
        )

    # -- search --------------------------------------------------------------------
    def dense_search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """``(passage position, similarity)`` pairs; similarity is higher-is-better."""
        if self.faiss_index is None:
            raise RuntimeError("This index has no dense component (built without an encoder).")
        k = min(k, len(self.passages))
        scores, ids = self.faiss_index.search(
            np.ascontiguousarray(query_vector, dtype=np.float32), k
        )
        sign = 1.0 if self.metric == "ip" else -1.0  # L2 returns distances
        return [
            (int(i), sign * float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1
        ]

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, min(k, len(self.passages)))

    # -- persistence -----------------------------------------------------------------
    @staticmethod
    def manifest_path(index_path: str | Path) -> Path:
        return Path(index_path).with_suffix(".manifest.json")

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """Write ``<index>.faiss`` (if dense), passages parquet and ``<index>.manifest.json``."""
        index_path, metadata_path = Path(index_path), Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(index_path))
        pd.DataFrame([asdict(p) for p in self.passages]).to_parquet(metadata_path, index=False)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "nq_qa_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_passages": len(self.passages),
            "has_dense": self.has_dense,
            "index_type": self.index_type,
            "metric": self.metric,
            "dim": int(self.faiss_index.d) if self.faiss_index is not None else None,
            "encoder": self.encoder_spec,
        }
        self.manifest_path(index_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> PassageIndex:
        manifest_file = cls.manifest_path(index_path)
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"{manifest_file} not found - rebuild the index with `make index`."
            )
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        frame = pd.read_parquet(metadata_path)
        passages = [
            Passage(id=str(r["id"]), text=str(r["text"]), question=str(r.get("question", "") or ""))
            for r in frame.to_dict("records")
        ]
        faiss_index = faiss.read_index(str(index_path)) if manifest.get("has_dense") else None
        if faiss_index is not None and faiss_index.ntotal != len(passages):
            raise IndexMismatchError(
                f"FAISS index holds {faiss_index.ntotal} vectors but metadata has "
                f"{len(passages)} passages - the artifacts are out of sync."
            )
        bm25 = BM25Index().fit([p.text for p in passages])
        return cls(passages, bm25, faiss_index, manifest.get("index_type"), manifest.get("encoder"))


def _make_faiss_index(
    vectors: np.ndarray, index_type: str, normalized: bool, nlist: int, nprobe: int
) -> Any:
    if faiss is None:
        raise ImportError(
            "FAISS is not installed. Run `pip install faiss-cpu` to enable dense index building."
        )
    dim = vectors.shape[1]
    metric = faiss.METRIC_INNER_PRODUCT if normalized or index_type == "FlatIP" else faiss.METRIC_L2
    if index_type == "FlatL2":
        index: faiss.Index = faiss.IndexFlatL2(dim)
    elif index_type == "FlatIP":
        index = faiss.IndexFlatIP(dim)
    else:  # IVFFlat needs a training set: roughly >= 39 vectors per centroid
        nlist = max(1, min(nlist, len(vectors) // MIN_VECTORS_PER_CENTROID))
        if nlist < 2:
            logger.warning(
                "Too few vectors (%d) for IVFFlat - falling back to a flat index.", len(vectors)
            )
            index = (
                faiss.IndexFlatIP(dim)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dim)
            )
        else:
            quantizer = (
                faiss.IndexFlatIP(dim)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dim)
            )
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric)
            index.train(vectors)
            index.nprobe = nprobe
    index.add(vectors)
    return index


# --------------------------------------------------------------------------- fusion
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[int, float]:
    """Weighted Reciprocal Rank Fusion: ``score(d) = sum_r weight_r / (k + rank_r(d))`` with 1-based ranks.

    A document missing from a ranking contributes nothing for that ranking.
    """
    fused: dict[int, float] = defaultdict(float)
    w_list = list(weights) if weights is not None else [1.0] * len(rankings)
    for ranking, w in zip(rankings, w_list, strict=False):
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] += float(w) / (k + rank)
    return dict(fused)


# ------------------------------------------------------------------------ retriever
class HybridRetriever:
    """Searches a :class:`PassageIndex` with BM25, dense vectors, or both fused by RRF."""

    def __init__(
        self,
        encoder: DenseEncoder | None = None,
        index: PassageIndex | None = None,
        mode: str = "hybrid",
        rrf_k: int = 60,
        candidate_pool: int = 50,
        index_type: str = "FlatIP",
        nlist: int = 256,
        nprobe: int = 8,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
        self.encoder = encoder
        self.index = index
        self.mode = mode
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def build_index(self, passages: Sequence[Passage], show_progress: bool = False) -> PassageIndex:
        """Build an index with this retriever's encoder (does **not** replace ``self.index``)."""
        return PassageIndex.build(
            passages,
            self.encoder,
            index_type=self.index_type,
            nlist=self.nlist,
            nprobe=self.nprobe,
            show_progress=show_progress,
        )

    def available_modes(self, index: PassageIndex | None = None) -> tuple[str, ...]:
        index = index or self.index
        if index is not None and index.has_dense and self.encoder is not None:
            return RETRIEVAL_MODES
        return ("bm25",)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        index: PassageIndex | None = None,
        mode: str | None = None,
    ) -> list[RetrievedPassage]:
        """Top-``top_k`` passages for ``query``.

        ``index`` overrides the shared index (used for ad-hoc documents) and ``mode``
        overrides the default mode for this call.
        """
        index = index or self.index
        if index is None:
            raise RuntimeError("No index loaded. Build or load one first.")
        mode = mode or self.mode
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
        if mode != "bm25" and not (index.has_dense and self.encoder is not None):
            raise ValueError(f"Retrieval mode {mode!r} needs a dense index and an encoder.")

        top_k = max(1, min(top_k, len(index)))
        pool = min(max(top_k, self.candidate_pool), len(index))

        dense: list[tuple[int, float]] = []
        sparse: list[tuple[int, float]] = []
        if mode in ("dense", "hybrid"):
            if self.encoder is None:
                raise RuntimeError("Dense retrieval requires an encoder.")
            dense = index.dense_search(self.encoder.encode([query]), pool)
        if mode in ("bm25", "hybrid"):
            sparse = index.sparse_search(query, pool)

        dense_scores = dict(dense)
        sparse_scores = dict(sparse)
        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                [[d for d, _ in dense], [d for d, _ in sparse]],
                self.rrf_k,
                weights=[self.dense_weight, self.sparse_weight],
            )
            ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        elif mode == "dense":
            ranked = dense[:top_k]
        else:
            ranked = sparse[:top_k]

        results: list[RetrievedPassage] = []
        for rank, (position, score) in enumerate(ranked, start=1):
            passage = index.passages[position]
            results.append(
                RetrievedPassage(
                    rank=rank,
                    doc_id=passage.id,
                    text=passage.text,
                    score=round(float(score), 6),
                    dense_score=_round(dense_scores.get(position)),
                    sparse_score=_round(sparse_scores.get(position)),
                    question=passage.question,
                )
            )
        return results

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Nothing to save: no index has been built.")
        self.index.save(index_path, metadata_path)

    def load(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """Load a persisted index and verify it matches this retriever's encoder."""
        index = PassageIndex.load(index_path, metadata_path)
        if self.encoder is not None and index.has_dense and index.encoder_spec is not None:
            built, wanted = index.encoder_spec, self.encoder.spec
            if (built.get("model"), built.get("pooling")) != (wanted["model"], wanted["pooling"]):
                raise IndexMismatchError(
                    f"Index was built with {built['model']} ({built['pooling']} pooling) but the "
                    f"retriever uses {wanted['model']} ({wanted['pooling']} pooling). "
                    "Rebuild the index with `make index`."
                )
        self.index = index


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)
