"""Clean the Natural Questions CSV, split it by passage and write an audit report.

python scripts/prepare_data.py                       # full CSV from the config
python scripts/prepare_data.py --limit-rows 20000    # quick run
python scripts/prepare_data.py --sample-size 300     # also refresh the bundled demo sample
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from nq_qa.data.nq_csv import CsvParseConfig, prepare_splits, read_jsonl, write_jsonl
from nq_qa.utils.config import load_config
from nq_qa.utils.logging import setup_logging

logger = logging.getLogger("prepare_data")


def make_sample(validation_path: Path, out_path: Path, size: int, seed: int) -> int:
    """Small, answerable, paragraph-only subset of the validation split for the offline demo."""
    pool = [
        r
        for r in read_jsonl(validation_path)
        if r["has_answer"] and r["context_type"] == "P" and len(r["context"].split()) <= 150
    ]
    random.Random(seed).shuffle(pool)
    return write_jsonl(out_path, pool[:size])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--csv", help="override dataset.csv_path")
    parser.add_argument("--out", help="override dataset.processed_dir")
    parser.add_argument(
        "--limit-rows", type=int, default=None, help="only parse the first N CSV rows"
    )
    parser.add_argument(
        "--sample-size", type=int, default=0, help="write a demo sample of N records"
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    csv_path = Path(args.csv or dcfg["csv_path"])
    out_dir = Path(args.out or dcfg["processed_dir"])
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found. Download the Natural Questions CSV (see data/README.md) "
            "and place it there, or pass --csv."
        )

    seed = int(cfg["project"]["seed"])
    audit = prepare_splits(
        csv_path,
        out_dir,
        CsvParseConfig.from_config(dcfg),
        val_ratio=float(dcfg["val_ratio"]),
        test_ratio=float(dcfg["test_ratio"]),
        seed=seed,
        limit_rows=args.limit_rows,
    )
    report = audit.to_dict()
    reports_dir = Path(cfg.get("evaluation", {}).get("report_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Audit: %s", json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)})
    )

    if args.sample_size:
        sample_path = Path(dcfg["sample_path"])
        n = make_sample(out_dir / "validation.jsonl", sample_path, args.sample_size, seed)
        logger.info("Wrote %d demo records to %s", n, sample_path)


if __name__ == "__main__":
    main()
