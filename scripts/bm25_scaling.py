"""How fast does BM25 retrieval degrade as the knowledge base grows?

The evaluation corpus is *closed* (it contains the gold passages of the evaluation questions),
which is much easier than retrieving from all of Wikipedia.  This script quantifies that by
re-running the BM25 evaluation on corpora of increasing size (no model needed):

    python scripts/bm25_scaling.py --sizes 12000 20000 50000 100000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nq_qa.data.corpus import select_corpus
from nq_qa.data.nq_csv import read_jsonl
from nq_qa.evaluation.harness import evaluate_retrieval
from nq_qa.models.retriever import HybridRetriever, PassageIndex
from nq_qa.utils.config import load_config
from nq_qa.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sizes", type=int, nargs="+", default=[12000, 20000, 50000, 100000])
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--out", default="reports/bm25_scaling.json")
    args = parser.parse_args()

    setup_logging("WARNING")
    cfg = load_config(args.config)
    processed = Path(cfg["dataset"]["processed_dir"])
    split = cfg["evaluation"]["split"]
    queries = [r for r in read_jsonl(processed / f"{split}.jsonl") if r["has_answer"]][
        : args.num_queries
    ]
    retriever = HybridRetriever(encoder=None, mode="bm25")

    rows = []
    for size in args.sizes:
        corpus = select_corpus(processed, size, int(cfg["project"]["seed"]))
        started = time.perf_counter()
        index = PassageIndex.build(corpus, encoder=None)
        build_s = time.perf_counter() - started
        metrics = evaluate_retrieval(retriever, index, queries, ["bm25"], ks=(1, 5, 10, 20))["bm25"]
        rows.append({"corpus_passages": len(index), "index_build_s": round(build_s, 1), **metrics})
        print(
            f"{len(index):>8,} passages  R@1={metrics['recall@1']:.3f}  R@5={metrics['recall@5']:.3f}  "
            f"R@20={metrics['recall@20']:.3f}  MRR={metrics['mrr']:.3f}  {metrics['avg_latency_ms']:.1f} ms/query"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"queries": len(queries), "split": split, "rows": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
