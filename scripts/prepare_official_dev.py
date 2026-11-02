"""Download and process the OFFICIAL 7,830 Natural Questions dev set from Hugging Face.

This extracts:
- Multi-annotator short answers (all 5 human annotators)
- Yes/No answer flags
- Character offsets mapped to clean detokenized contexts
- Audit report (saved to reports/official_dev_audit.json)

Usage:
    python scripts/prepare_official_dev.py
    python scripts/prepare_official_dev.py --limit 1000   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from nq_qa.data.nq_csv import write_jsonl
from nq_qa.data.nq_processor import NaturalQuestionsProcessor
from nq_qa.utils.config import load_config
from nq_qa.utils.logging import setup_logging

logger = logging.getLogger("prepare_official_dev")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dev records")
    parser.add_argument("--out", default=None, help="Path to output jsonl file")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    out_path = Path(args.out or (processed_dir / "dev_official.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The `datasets` library is required for downloading from Hugging Face.\n"
            "Run: pip install datasets"
        ) from None

    logger.info(
        "Connecting to Hugging Face Hub: google-research-datasets/natural_questions (split='validation')..."
    )
    dataset_stream = load_dataset(
        "google-research-datasets/natural_questions", split="validation", streaming=True
    )

    records = []
    answerable_count = 0
    yes_no_count = 0
    multi_annotator_answers = 0
    context_type_counts: Counter[str] = Counter()

    for idx, raw in enumerate(dataset_stream):
        if args.limit is not None and idx >= args.limit:
            break
        parsed = NaturalQuestionsProcessor.parse_raw_nq_record(raw)
        if parsed is None:
            continue

        records.append(parsed)
        context_type_counts[parsed.get("context_type", "plain")] += 1
        num_answers = len(parsed["answers"]["text"])
        if num_answers > 0:
            answerable_count += 1
        if num_answers > 1:
            multi_annotator_answers += 1
        if parsed.get("has_yes_no"):
            yes_no_count += 1

        if len(records) % 500 == 0:
            logger.info(
                "Processed %d dev records (%d answerable)...", len(records), answerable_count
            )

    write_jsonl(out_path, records)

    audit = {
        "dataset": "google-research-datasets/natural_questions",
        "split": "validation",
        "total_records_processed": len(records),
        "answerable_records": answerable_count,
        "unanswerable_records": len(records) - answerable_count,
        "records_with_multiple_reference_answers": multi_annotator_answers,
        "records_with_yes_no_answer": yes_no_count,
        "context_type_counts": dict(context_type_counts),
        "output_path": str(out_path),
    }

    report_dir = Path(cfg.get("evaluation", {}).get("report_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    audit_file = report_dir / "official_dev_audit.json"
    audit_file.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info("Done preparing official dev set:")
    logger.info(" - Total records: %d", len(records))
    logger.info(
        " - Answerable: %d (%.1f%%)",
        answerable_count,
        100.0 * answerable_count / max(1, len(records)),
    )
    logger.info(" - Multi-reference answers: %d", multi_annotator_answers)
    logger.info(" - Yes/No answers detected: %d", yes_no_count)
    logger.info(" - Saved to: %s", out_path)
    logger.info(" - Audit log: %s", audit_file)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
