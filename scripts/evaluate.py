"""Evaluate retrieval, the reader and the full pipeline; write reports/metrics.{json,md}.

python scripts/evaluate.py --bm25-only              # no model download, runs anywhere
python scripts/evaluate.py                          # all retrieval modes + reader + end-to-end
python scripts/evaluate.py --num-queries 200 --skip-e2e
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from nq_qa.data.corpus import select_corpus
from nq_qa.data.nq_csv import read_jsonl
from nq_qa.evaluation.harness import evaluate_end_to_end, evaluate_reader, evaluate_retrieval
from nq_qa.evaluation.report import render_markdown
from nq_qa.models.retriever import PassageIndex
from nq_qa.pipeline.e2e_qa import EndToEndQAPipeline
from nq_qa.pipeline.factory import build_reader, build_retriever
from nq_qa.utils.config import load_config
from nq_qa.utils.logging import setup_logging

logger = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument(
        "--bm25-only", action="store_true", help="BM25 retrieval only (no models needed)"
    )
    parser.add_argument(
        "--official-dev",
        action="store_true",
        help="Evaluate on the official 7,830 NQ dev set (data/processed/dev_official.jsonl)",
    )
    parser.add_argument("--skip-reader", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    ecfg = cfg["evaluation"]
    processed = Path(cfg["dataset"]["processed_dir"])

    if args.official_dev:
        split = "dev_official"
        dev_file = processed / f"{split}.jsonl"
        if not dev_file.exists():
            raise SystemExit(
                f"{dev_file} not found.\n"
                "Run `python scripts/prepare_official_dev.py` to stream the official 7,830 NQ dev set from Hugging Face."
            )
        split_records = read_jsonl(dev_file)
    else:
        split = ecfg["split"]
        split_file = processed / f"{split}.jsonl"
        if not split_file.exists():
            raise SystemExit(f"{split_file} not found - run `make data` first.")
        split_records = read_jsonl(split_file)

    n_queries = args.num_queries or int(ecfg["num_queries"])
    queries = [r for r in split_records if r["has_answer"]][:n_queries]
    ks = tuple(int(k) for k in ecfg["ks"])
    results: dict = {}

    if args.bm25_only:
        cfg["retriever"]["mode"] = "bm25"
        retriever = build_retriever(cfg, load_index=False)
        corpus = select_corpus(
            processed,
            int(cfg["index"]["corpus_size"]),
            int(cfg["project"]["seed"]),
            eval_splits=(split,),
        )
        index = PassageIndex.build(corpus, encoder=None)
        retriever.index = index
        args.skip_reader = args.skip_e2e = True
    else:
        retriever = build_retriever(cfg, allow_sample_fallback=False)
        index = retriever.index
        assert index is not None

    modes = list(retriever.available_modes(index))
    logger.info("Evaluating %d queries on %d passages, modes=%s", len(queries), len(index), modes)
    results["retrieval"] = evaluate_retrieval(
        retriever, index, queries, modes, ks, show_progress=True
    )

    reader = None
    if not args.skip_reader or not args.skip_e2e:
        reader = build_reader(cfg)
    if not args.skip_reader:
        gold = [r for r in split_records if r["context_type"] != "hard_negative"][
            : int(cfg["dataset"]["max_val_samples"])
        ]
        results["reader"] = evaluate_reader(reader, gold, show_progress=True)
    if not args.skip_e2e:
        from nq_qa.pipeline.factory import build_pipeline

        pipeline = build_pipeline(cfg, allow_sample_fallback=False)
        pipeline.retriever = retriever
        if reader is not None:
            pipeline.reader = reader
        results["end_to_end"] = {
            mode: evaluate_end_to_end(
                pipeline, queries, int(ecfg["e2e_top_k"]), mode, show_progress=True
            )
            for mode in modes
        }

    results["meta"] = {
        "split": split,
        "num_queries": len(queries),
        "corpus_passages": len(index),
        "encoder": (retriever.encoder.spec["model"] if retriever.encoder else None),
        "reader": reader.model_name_or_path if reader else None,
        "reranker": (
            cfg.get("reranker", {}).get("model_name")
            if cfg.get("reranker", {}).get("enabled", False)
            else None
        ),
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else f"cpu ({platform.processor() or platform.machine()})"
        ),
        "torch": torch.__version__,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    report_dir = Path(ecfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = render_markdown(results)
    (report_dir / "metrics.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
