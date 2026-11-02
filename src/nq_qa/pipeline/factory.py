"""Build the QA components from ``configs/config.yaml``.

One place decides which encoder, index and reader checkpoint are used, so the API, the
Streamlit app and the CLI scripts can never drift apart.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nq_qa.data.corpus import passages_from_records
from nq_qa.data.nq_csv import read_jsonl
from nq_qa.models.reader import ExtractiveReader
from nq_qa.models.retriever import DenseEncoder, HybridRetriever
from nq_qa.pipeline.e2e_qa import EndToEndQAPipeline

logger = logging.getLogger(__name__)


def build_encoder(cfg: Mapping[str, Any]) -> DenseEncoder:
    """Dense encoder selected by ``retriever.dense_encoder`` (``biencoder`` | ``cls``)."""
    rcfg = cfg["retriever"]
    max_length = int(rcfg.get("max_passage_tokens", 256))
    kind = rcfg.get("dense_encoder", "biencoder")
    if kind == "cls":
        return DenseEncoder(
            rcfg["cls_model_name"], pooling="cls", normalize=False, max_length=max_length
        )
    if kind == "biencoder":
        return DenseEncoder(
            rcfg["biencoder_model_name"], pooling="mean", normalize=True, max_length=max_length
        )
    raise ValueError(f"retriever.dense_encoder must be 'biencoder' or 'cls', got {kind!r}")


def build_retriever(
    cfg: Mapping[str, Any], *, load_index: bool = True, allow_sample_fallback: bool = True
) -> HybridRetriever:
    """Create the retriever and (optionally) load its index.

    Index resolution: persisted index -> bundled demo sample (``allow_sample_fallback``) ->
    error with instructions.  The demo fallback keeps ``git clone && make demo`` working.
    """
    rcfg = cfg["retriever"]
    mode = rcfg.get("mode", "hybrid")
    encoder = None if mode == "bm25" else build_encoder(cfg)
    retriever = HybridRetriever(
        encoder=encoder,
        mode=mode,
        rrf_k=int(rcfg.get("rrf_k", 60)),
        candidate_pool=int(rcfg.get("candidate_pool", 50)),
        index_type=rcfg.get("index_type", "FlatIP"),
        nlist=int(rcfg.get("nlist", 256)),
        nprobe=int(rcfg.get("nprobe", 8)),
        dense_weight=float(rcfg.get("dense_weight", 1.0)),
        sparse_weight=float(rcfg.get("sparse_weight", 1.0)),
    )
    if not load_index:
        return retriever

    index_path, meta_path = Path(rcfg["index_path"]), Path(rcfg["metadata_path"])
    manifest = index_path.with_suffix(".manifest.json")
    if manifest.exists() and meta_path.exists():
        retriever.load(index_path, meta_path)
        logger.info("Loaded index with %d passages from %s", len(retriever.index), index_path)  # type: ignore[arg-type]
        return retriever

    sample = Path(cfg.get("dataset", {}).get("sample_path", ""))
    if allow_sample_fallback and sample.is_file():
        logger.warning(
            "No index at %s - building an in-memory index from the bundled demo sample (%s). "
            "Run `make data index` for the full knowledge base.",
            index_path,
            sample,
        )
        retriever.index = retriever.build_index(passages_from_records(read_jsonl(sample)))
        return retriever

    raise FileNotFoundError(
        f"No index found at {index_path}. Build one with `make data index` "
        "(or provide dataset.sample_path for the demo corpus)."
    )


def resolve_reader_path(cfg: Mapping[str, Any]) -> str:
    """Fine-tuned checkpoint if present (and enabled), otherwise the public checkpoint."""
    rcfg = cfg["reader"]
    finetuned = Path(rcfg["output_dir"])
    if rcfg.get("use_finetuned_if_available", True) and (finetuned / "config.json").is_file():
        return str(finetuned)
    return str(rcfg["pretrained_checkpoint"])


def build_reader(cfg: Mapping[str, Any]) -> ExtractiveReader:
    rcfg = cfg["reader"]
    dcfg = cfg.get("dataset", {})
    path = resolve_reader_path(cfg)
    logger.info("Loading reader from %s", path)
    null_thresh = (
        float(rcfg["null_threshold"])
        if ("null_threshold" in rcfg and rcfg["null_threshold"] is not None)
        else None
    )
    return ExtractiveReader(
        path,
        n_best_size=int(rcfg.get("n_best_size", 20)),
        max_answer_length=int(rcfg.get("max_answer_length", 30)),
        max_length=int(dcfg.get("max_length", 384)),
        stride=int(dcfg.get("stride", 128)),
        batch_size=int(rcfg.get("inference_batch_size", 16)),
        abstain=bool(rcfg.get("abstain", True)),
        null_threshold=null_thresh,
    )


def build_reranker(cfg: Mapping[str, Any]) -> Any | None:
    """Build cross-encoder re-ranker if enabled in config."""
    rcfg = cfg.get("reranker", {})
    if not rcfg.get("enabled", False):
        return None
    from nq_qa.models.reranker import CrossEncoderReranker

    model_name = str(rcfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    max_length = int(rcfg.get("max_length", 512))
    batch_size = int(rcfg.get("batch_size", 16))
    logger.info("Loading cross-encoder re-ranker from %s", model_name)
    try:
        return CrossEncoderReranker(
            model_name_or_path=model_name,
            max_length=max_length,
            batch_size=batch_size,
        )
    except Exception as exc:
        logger.warning(
            "Could not load cross-encoder re-ranker (%s): %s. Disabling reranker.", model_name, exc
        )
        return None


def build_pipeline(
    cfg: Mapping[str, Any], *, allow_sample_fallback: bool = True
) -> EndToEndQAPipeline:
    """Full pipeline: retriever + optional reranker + index + reader, wired with score calibration."""
    retriever = build_retriever(cfg, allow_sample_fallback=allow_sample_fallback)
    reader = build_reader(cfg)
    reranker = build_reranker(cfg)
    rcfg = cfg.get("retriever", {})
    min_confidence = float(
        cfg.get("pipeline", {}).get(
            "min_confidence",
            cfg.get("reader", {}).get("min_confidence", 0.45),
        )
    )
    return EndToEndQAPipeline(
        retriever=retriever,
        reader=reader,
        alpha_reader_weight=float(rcfg.get("alpha_reader_weight", 0.7)),
        reranker=reranker,
        candidate_pool=int(cfg.get("reranker", {}).get("candidate_pool", 50)),
        score_calibration=str(rcfg.get("score_calibration", "calibrated")),
        min_confidence=min_confidence,
    )
