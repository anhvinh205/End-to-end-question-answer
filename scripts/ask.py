"""Ask one question from the command line.

python scripts/ask.py "who invented the world wide web"
python scripts/ask.py "who wrote hamlet" --top-k 3 --mode bm25
"""

from __future__ import annotations

import argparse
import json

from nq_qa.pipeline.factory import build_pipeline
from nq_qa.utils.config import load_config
from nq_qa.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("question")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default=None)
    parser.add_argument("--json", action="store_true", help="print the full JSON result")
    args = parser.parse_args()

    setup_logging("WARNING")
    pipeline = build_pipeline(load_config(args.config))
    result = pipeline.answer_question(args.question, top_k=args.top_k, mode=args.mode)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if result["answerable"]:
        print(f"Answer : {result['best_answer']}  (score {result['best_score']:.3f})")
        print(f"Context: {result['best_context'][:300]}")
    else:
        print("No confident answer found in the retrieved passages.")
    print(
        f"[{result['mode']} | retrieval {result['retrieval_ms']} ms | reader {result['reader_ms']} ms]"
    )


if __name__ == "__main__":
    main()
