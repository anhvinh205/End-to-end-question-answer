from pathlib import Path

import pytest

from nq_qa.data.corpus import passages_from_records, select_corpus
from nq_qa.data.nq_csv import write_jsonl
from nq_qa.pipeline import factory
from nq_qa.utils.config import CONFIG_ENV_VAR, load_config

REPO = Path(__file__).resolve().parents[1]


def record(i, split_passage=None):
    return {
        "id": f"r{i}",
        "question": f"question {i}",
        "context": f"passage text number {i} with enough words to be meaningful",
        "passage_id": split_passage or f"p{i}",
        "answers": {"text": ["text"], "answer_start": [8]},
        "has_answer": True,
        "context_type": "P",
    }


# ---------------------------------------------------------------------------- config
def test_load_config_explicit_path_and_env(tmp_path, monkeypatch):
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\n")
    assert load_config(path) == {"a": 1}
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    assert load_config() == {"a": 1}


def test_load_config_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ValueError):
        load_config(bad)


def test_repository_config_has_every_key_the_code_reads():
    cfg = load_config(REPO / "configs" / "config.yaml")
    required = {
        "project": ["seed"],
        "dataset": ["csv_path", "processed_dir", "sample_path", "context_types", "val_ratio", "test_ratio",
                    "max_train_samples", "max_val_samples", "max_length", "stride", "negative_sampling_ratio"],
        "index": ["corpus_size"],
        "retriever": ["mode", "dense_encoder", "cls_model_name", "biencoder_model_name", "index_type",
                      "index_path", "metadata_path", "candidate_pool", "rrf_k", "alpha_reader_weight"],
        "reader": ["base_model", "pretrained_checkpoint", "output_dir", "learning_rate", "num_train_epochs",
                   "per_device_train_batch_size", "weight_decay", "n_best_size", "max_answer_length"],
        "evaluation": ["split", "num_queries", "ks", "e2e_top_k", "report_dir"],
    }  # fmt: skip
    for section, keys in required.items():
        for key in keys:
            assert key in cfg[section], f"configs/config.yaml is missing {section}.{key}"
    assert cfg["retriever"]["mode"] in ("bm25", "dense", "hybrid")
    assert cfg["retriever"]["dense_encoder"] in ("biencoder", "cls")


# ------------------------------------------------------------------------- factory
def base_cfg(tmp_path, **retriever):
    sample = tmp_path / "sample.jsonl"
    write_jsonl(sample, [record(i) for i in range(6)])
    return {
        "dataset": {"sample_path": str(sample)},
        "retriever": {
            "mode": "bm25",
            "index_path": str(tmp_path / "idx" / "i.faiss"),
            "metadata_path": str(tmp_path / "idx" / "m.parquet"),
            **retriever,
        },
        "reader": {"output_dir": str(tmp_path / "ft"), "pretrained_checkpoint": "public/model"},
    }


def test_build_encoder_rejects_unknown_kind():
    with pytest.raises(ValueError, match="biencoder"):
        factory.build_encoder({"retriever": {"dense_encoder": "colbert"}})


def test_retriever_falls_back_to_the_bundled_sample(tmp_path, caplog):
    retriever = factory.build_retriever(base_cfg(tmp_path))
    assert retriever.index is not None and len(retriever.index) == 6
    assert "demo sample" in caplog.text


def test_retriever_without_index_or_sample_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make data index"):
        factory.build_retriever(base_cfg(tmp_path), allow_sample_fallback=False)


def test_retriever_prefers_a_persisted_index(tmp_path):
    cfg = base_cfg(tmp_path)
    first = factory.build_retriever(cfg, load_index=False)
    first.index = first.build_index(passages_from_records([record(i) for i in range(3)]))
    first.save(cfg["retriever"]["index_path"], cfg["retriever"]["metadata_path"])
    assert len(factory.build_retriever(cfg).index) == 3  # not the 6-record sample


def test_resolve_reader_path_prefers_the_finetuned_model(tmp_path):
    cfg = base_cfg(tmp_path)
    assert factory.resolve_reader_path(cfg) == "public/model"
    (tmp_path / "ft").mkdir()
    (tmp_path / "ft" / "config.json").write_text("{}")
    assert factory.resolve_reader_path(cfg) == str(tmp_path / "ft")
    cfg["reader"]["use_finetuned_if_available"] = False
    assert factory.resolve_reader_path(cfg) == "public/model"


# --------------------------------------------------------------------------- corpus
def test_passages_from_records_deduplicates_by_passage_id():
    records = [record(1), {**record(2), "passage_id": "p1"}, record(3)]
    assert [p.id for p in passages_from_records(records)] == ["p1", "p3"]


def test_select_corpus_puts_eval_passages_first_and_fills_with_train(tmp_path):
    write_jsonl(tmp_path / "validation.jsonl", [record(i) for i in range(0, 3)])
    write_jsonl(tmp_path / "test.jsonl", [record(i) for i in range(3, 5)])
    write_jsonl(tmp_path / "train.jsonl", [record(i) for i in range(5, 40)])
    corpus = select_corpus(tmp_path, corpus_size=12, seed=1)
    ids = [p.id for p in corpus]
    assert len(ids) == 12 and len(set(ids)) == 12
    assert set(ids[:5]) == {"p0", "p1", "p2", "p3", "p4"}  # every eval passage is present
    assert select_corpus(tmp_path, corpus_size=12, seed=1) == corpus  # deterministic
    assert len(select_corpus(tmp_path, corpus_size=3)) == 3


def test_build_encoder_cls_variant(tiny_encoder_dir):
    encoder = factory.build_encoder(
        {
            "retriever": {
                "dense_encoder": "cls",
                "cls_model_name": str(tiny_encoder_dir),
                "max_passage_tokens": 32,
            }
        }
    )
    assert (
        encoder.spec["pooling"] == "cls"
        and encoder.spec["normalize"] is False
        and encoder.max_length == 32
    )


@pytest.mark.slow
def test_build_pipeline_wires_everything_from_the_config(
    tmp_path, tiny_encoder_dir, trained_qa_dir
):
    cfg = base_cfg(
        tmp_path,
        mode="hybrid",
        dense_encoder="biencoder",
        biencoder_model_name=str(tiny_encoder_dir),
        max_passage_tokens=48,
        candidate_pool=10,
        alpha_reader_weight=0.6,
    )
    cfg["dataset"].update(max_length=48, stride=12)
    cfg["reader"].update(output_dir=str(trained_qa_dir), n_best_size=10, inference_batch_size=4)
    pipeline = factory.build_pipeline(cfg)  # no persisted index -> builds one from the sample
    assert pipeline.alpha == 0.6
    assert pipeline.reader.model_name_or_path == str(trained_qa_dir)  # fine-tuned model preferred
    assert pipeline.retriever.available_modes() == ("bm25", "dense", "hybrid")
    result = pipeline.answer_question("what is the capital of peru", top_k=2)
    assert result["mode"] == "hybrid" and len(result["candidates"]) == 2
