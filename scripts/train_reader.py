"""Fine-tune the extractive reader on Natural Questions and tune its abstention threshold.

python scripts/train_reader.py                    # settings from configs/config.yaml
python scripts/train_reader.py --max-steps 50     # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from nq_qa.data.negatives import build_training_records
from nq_qa.data.nq_csv import read_jsonl
from nq_qa.data.nq_processor import NaturalQuestionsProcessor
from nq_qa.evaluation.harness import evaluate_reader
from nq_qa.models.reader import ExtractiveReader, write_reader_meta
from nq_qa.models.training import build_training_features, train_reader
from nq_qa.utils.config import load_config
from nq_qa.utils.logging import setup_logging

logger = logging.getLogger("train_reader")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after N optimiser steps")
    parser.add_argument("--cpu", action="store_true", help="force CPU training")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    dcfg, rcfg = cfg["dataset"], cfg["reader"]
    processed = Path(dcfg["processed_dir"])
    seed = int(cfg["project"]["seed"])
    for split in ("train", "validation"):
        if not (processed / f"{split}.jsonl").exists():
            raise SystemExit(f"{processed}/{split}.jsonl not found - run `make data` first.")

    train_raw = read_jsonl(processed / "train.jsonl", limit=int(dcfg["max_train_samples"]))
    val_raw = read_jsonl(processed / "validation.jsonl", limit=int(dcfg["max_val_samples"]))
    train_records = build_training_records(train_raw, float(dcfg["negative_sampling_ratio"]), seed)

    tokenizer = AutoTokenizer.from_pretrained(rcfg["base_model"])
    processor = NaturalQuestionsProcessor(tokenizer, int(dcfg["max_length"]), int(dcfg["stride"]))
    features = build_training_features(train_records, processor)
    logger.info(
        "%d training examples -> %d sliding-window features", len(train_records), len(features)
    )

    output_dir = Path(rcfg["output_dir"])
    train_reader(
        rcfg["base_model"],
        features,
        output_dir,
        learning_rate=float(rcfg["learning_rate"]),
        num_train_epochs=float(rcfg["num_train_epochs"]),
        weight_decay=float(rcfg["weight_decay"]),
        batch_size=int(rcfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(rcfg.get("gradient_accumulation_steps", 1)),
        warmup_ratio=float(rcfg.get("warmup_ratio", 0.1)),
        fp16=bool(rcfg.get("fp16", True)),
        seed=seed,
        max_steps=args.max_steps,
        use_cpu=args.cpu,
    )

    # Evaluate with the exact decoding path the API uses, then store the tuned threshold.
    reader = ExtractiveReader(
        str(output_dir),
        n_best_size=int(rcfg["n_best_size"]),
        max_answer_length=int(rcfg["max_answer_length"]),
        max_length=int(dcfg["max_length"]),
        stride=int(dcfg["stride"]),
        batch_size=int(rcfg.get("inference_batch_size", 16)),
        abstain=False,
    )
    metrics = evaluate_reader(reader, val_raw, show_progress=True)
    write_reader_meta(
        output_dir,
        {
            "null_threshold": metrics["best_null_threshold"],
            "trained_with_negatives": True,
            "base_model": rcfg["base_model"],
            "train_examples": len(train_records),
            "validation_examples": len(val_raw),
            "validation_f1_at_threshold": metrics["f1_at_best_threshold"],
        },
    )
    report_dir = Path(cfg["evaluation"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "reader_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Validation metrics: %s", json.dumps(metrics))


if __name__ == "__main__":
    main()
