"""Run the real CLI scripts (as subprocesses) on a synthetic Natural Questions CSV.

``prepare_data -> build_index -> train_reader -> evaluate -> ask`` is the workflow a user runs
with ``make``; this checks that the scripts, the config and the package fit together using
tiny offline models.  Fixtures are module scoped, so each stage executes once.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from .conftest import CAPITALS

pytest.importorskip("datasets")
REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / name), *args],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=600,
    )  # fmt: skip
    assert result.returncode == 0, (
        f"{name} failed:\n{result.stdout[-1500:]}\n{result.stderr[-3000:]}"
    )
    return result


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, tiny_tokenizer_dir, tiny_encoder_dir) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("workflow")

    # synthetic CSV in the Natural Questions tokenisation
    csv_path = root / "nq.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "long_answers", "short_answers"])
        pairs = [(c, k) for c in CAPITALS for k in CAPITALS.values() if CAPITALS[c] != k]
        for i, (country, city) in enumerate(pairs):
            passage = f"<P> the capital of {country} is {city} and it is a large city near the coast . </P>"
            writer.writerow([f"what is the capital of {country}", passage, city if i % 4 else ""])
        writer.writerows([["no passage here", "", ""]] * 5)

    # an untrained tiny reader to fine-tune (random weights + the tiny tokenizer)
    import torch
    from transformers import AutoTokenizer, BertForQuestionAnswering

    from .conftest import _bert_config

    base = root / "base_reader"
    tokenizer = AutoTokenizer.from_pretrained(tiny_tokenizer_dir)
    torch.manual_seed(0)
    BertForQuestionAnswering(_bert_config(len(tokenizer))).save_pretrained(base)
    tokenizer.save_pretrained(base)

    cfg = yaml.safe_load((REPO / "configs" / "config.yaml").read_text())
    cfg["dataset"].update(
        csv_path=str(csv_path), processed_dir=str(root / "processed"), sample_path=str(root / "sample.jsonl"),
        min_context_words=5, val_ratio=0.2, test_ratio=0.2, max_train_samples=200, max_val_samples=40,
        max_length=64, stride=16,
    )  # fmt: skip
    cfg["index"]["corpus_size"] = 100
    cfg["retriever"].update(
        mode="hybrid", dense_encoder="biencoder", biencoder_model_name=str(tiny_encoder_dir),
        max_passage_tokens=48, candidate_pool=10,
        index_path=str(root / "artifacts" / "nq.faiss"), metadata_path=str(root / "artifacts" / "nq.parquet"),
    )  # fmt: skip
    cfg["reader"].update(
        base_model=str(base), pretrained_checkpoint=str(base), output_dir=str(root / "artifacts" / "reader"),
        learning_rate=3e-3, num_train_epochs=2, per_device_train_batch_size=4, fp16=False, n_best_size=10,
    )  # fmt: skip
    cfg["reranker"] = {"enabled": False}
    cfg["evaluation"].update(
        report_dir=str(root / "reports"), num_queries=8, ks=[1, 3], e2e_top_k=3
    )
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return {"root": root, "config": config_path, "cfg_dir": root}


@pytest.fixture(scope="module")
def prepared(workspace):
    run_script("prepare_data.py", "--config", str(workspace["config"]), "--sample-size", "10")
    return workspace["root"]


@pytest.fixture(scope="module")
def indexed(workspace, prepared):
    run_script("build_index.py", "--config", str(workspace["config"]))
    return workspace["root"] / "artifacts"


@pytest.fixture(scope="module")
def trained(workspace, prepared):
    run_script("train_reader.py", "--config", str(workspace["config"]), "--cpu")
    return workspace["root"] / "artifacts" / "reader"


def test_prepare_data_writes_splits_audit_and_sample(prepared):
    for split in ("train", "validation", "test"):
        assert (prepared / "processed" / f"{split}.jsonl").stat().st_size > 0
    audit = json.loads((prepared / "processed" / "audit.json").read_text())
    assert audit["rows_total"] == 137 and audit["no_long_answer"] == 5
    assert audit["alignment_rate"] == 1.0
    sample = [json.loads(line) for line in (prepared / "sample.jsonl").read_text().splitlines()]
    assert 1 <= len(sample) <= 10 and all(r["has_answer"] for r in sample)
    assert (
        prepared / "reports" / "data_audit.json"
    ).exists()  # copied to the configured report dir


def test_build_index_persists_index_passages_and_manifest(indexed):
    manifest = json.loads((indexed / "nq.manifest.json").read_text())
    assert (
        manifest["has_dense"]
        and manifest["encoder"]["pooling"] == "mean"
        and manifest["num_passages"] > 20
    )
    assert (indexed / "nq.faiss").exists() and (indexed / "nq.parquet").exists()


def test_train_reader_saves_model_threshold_and_metrics(trained, workspace):
    assert (trained / "config.json").exists() and (trained / "reader_meta.json").exists()
    meta = json.loads((trained / "reader_meta.json").read_text())
    assert meta["trained_with_negatives"] and "null_threshold" in meta
    metrics = json.loads((workspace["root"] / "reports" / "reader_metrics.json").read_text())
    assert metrics["total"] > 0 and "best_null_threshold" in metrics


def test_evaluate_covers_all_modes_reader_and_end_to_end(workspace, indexed, trained):
    result = run_script("evaluate.py", "--config", str(workspace["config"]))
    report = json.loads((workspace["root"] / "reports" / "metrics.json").read_text())
    assert set(report["retrieval"]) == {"bm25", "dense", "hybrid"}
    assert set(report["end_to_end"]) == {"bm25", "dense", "hybrid"}
    assert report["reader"]["total"] > 0 and report["meta"]["corpus_passages"] > 20
    assert 0.0 <= report["retrieval"]["hybrid"]["recall@3"] <= 1.0
    assert (
        "## End-to-end" in result.stdout and (workspace["root"] / "reports" / "metrics.md").exists()
    )


def test_evaluate_bm25_only_needs_no_models(workspace, prepared):
    run_script(
        "evaluate.py", "--config", str(workspace["config"]), "--bm25-only", "--num-queries", "6"
    )
    report = json.loads((workspace["root"] / "reports" / "metrics.json").read_text())
    assert (
        list(report["retrieval"]) == ["bm25"]
        and "reader" not in report
        and "end_to_end" not in report
    )
    assert report["retrieval"]["bm25"]["recall@3"] > 0.5


def test_ask_prints_a_json_result(workspace, indexed, trained):
    result = run_script(
        "ask.py",
        "what is the capital of peru",
        "--config",
        str(workspace["config"]),
        "--json",
        "--top-k",
        "3",
    )
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["question"] == "what is the capital of peru" and len(payload["candidates"]) == 3
    assert payload["mode"] == "hybrid"
