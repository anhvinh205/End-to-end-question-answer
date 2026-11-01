"""Build the retrieval index (FAISS + BM25 passages) from the processed Natural Questions data.

The knowledge base holds every validation and test gold passage plus random training
passages as distractors, up to ``index.corpus_size`` passages.

    python scripts/build_index.py                  # dense + BM25 (downloads the encoder once)
    python scripts/build_index.py --corpus-size 5000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nq_qa.data.corpus import select_corpus
from nq_qa.pipeline.factory import build_retriever
from nq_qa.utils.config import load_config
from nq_qa.utils.logging import setup_logging

logger = logging.getLogger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--corpus-size", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    processed = Path(cfg["dataset"]["processed_dir"])
    if not (processed / "validation.jsonl").exists():
        raise SystemExit(f"{processed}/validation.jsonl not found - run `make data` first.")

    size = args.corpus_size or int(cfg["index"]["corpus_size"])
    passages = select_corpus(processed, size, int(cfg["project"]["seed"]))
    logger.info("Indexing %d passages", len(passages))

    retriever = build_retriever(cfg, load_index=False)
    retriever.index = retriever.build_index(passages, show_progress=True)
    retriever.save(cfg["retriever"]["index_path"], cfg["retriever"]["metadata_path"])
    logger.info("Index saved to %s", cfg["retriever"]["index_path"])


if __name__ == "__main__":
    main()
