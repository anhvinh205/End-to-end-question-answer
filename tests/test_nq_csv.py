import csv
import json
from pathlib import Path

import pytest

from nq_qa.data.nq_csv import (
    AuditReport,
    CsvParseConfig,
    assign_split,
    iter_csv_records,
    parse_row,
    prepare_splits,
    read_jsonl,
    write_jsonl,
)

PASSAGE = (
    "<P> Active quarterback Tom Brady holds the records for most wins with 220 , most regular "
    "season wins with 195 , and most postseason wins with 25 , as of Week 16 of the 2017 NFL season . </P>"
)
CFG = CsvParseConfig(min_context_words=5)


def parse(question, long_raw, short_raw, cfg=CFG):
    audit = AuditReport()
    return parse_row(0, question, long_raw, short_raw, cfg, audit), audit


def test_answerable_row_is_aligned_to_character_offsets():
    record, audit = parse("who had the most wins in the nfl", PASSAGE, "Tom Brady")
    assert record is not None and record["has_answer"]
    text, start = record["answers"]["text"][0], record["answers"]["answer_start"][0]
    assert text == "Tom Brady"
    assert record["context"][start : start + len(text)] == "Tom Brady"
    assert "<P>" not in record["context"] and "2017 NFL season." in record["context"]
    assert record["context_type"] == "P"
    assert audit.aligned == 1 and audit.unaligned == 0


def test_answer_matching_is_case_insensitive_fallback():
    record, _ = parse("q", PASSAGE, "tom brady")
    assert record["answers"]["text"] == ["Tom Brady"]


def test_multi_span_answer_keeps_every_located_span():
    record, audit = parse("q", PASSAGE, "Tom Brady, NFL season")
    assert record["answers"]["text"] == ["Tom Brady", "NFL season"]
    starts = record["answers"]["answer_start"]
    assert starts == sorted(starts)
    assert audit.aligned_multi_span == 1


def test_long_answer_without_short_answer_is_a_natural_negative():
    record, audit = parse("q", PASSAGE, "")
    assert record is not None and not record["has_answer"]
    assert record["answers"] == {"text": [], "answer_start": []}
    assert audit.unanswerable_kept == 1


def test_unanswerable_can_be_dropped():
    record, audit = parse(
        "q", PASSAGE, "", CsvParseConfig(min_context_words=5, keep_unanswered=False)
    )
    assert record is None and audit.unanswerable_dropped == 1


def test_row_without_passage_is_skipped():
    record, audit = parse("q", "", "")
    assert record is None and audit.no_long_answer == 1


def test_tables_are_filtered_by_context_type():
    record, audit = parse("q", "<Table> <Tr> <Td> a b c d e f </Td> </Tr> </Table>", "a")
    assert record is None and audit.filtered_context_type == 1


@pytest.mark.parametrize(
    ("words", "counter"), [(3, "filtered_too_short"), (400, "filtered_too_long")]
)
def test_length_filters(words, counter):
    raw = "<P> " + " ".join(["word"] * words) + " </P>"
    record, audit = parse("q", raw, "", CsvParseConfig(min_context_words=5, max_context_words=350))
    assert record is None and getattr(audit, counter) == 1


def test_unlocatable_answer_is_counted_not_guessed():
    record, audit = parse("q", PASSAGE, "completely different answer")
    assert record is None and audit.unaligned == 1


def test_overlong_answer_is_dropped():
    record, audit = parse("q", PASSAGE, "Active quarterback Tom Brady holds the records for most wins",
                          CsvParseConfig(min_context_words=5, max_answer_words=3))  # fmt: skip
    assert record is None and audit.answer_too_long == 1


def test_passage_id_is_content_addressed():
    a, _ = parse("question one", PASSAGE, "Tom Brady")
    b, _ = parse("another question", PASSAGE, "")
    assert a["passage_id"] == b["passage_id"]


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "long_answers", "short_answers"])
        writer.writerows(rows)


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    rows = []
    for i in range(60):
        passage = f"<P> Document {i} is about topic number {i} and mentions the famous person Person{i} in detail today . </P>"
        rows.append((f"question {i}", passage, f"Person{i}" if i % 3 else ""))
    rows += [
        ("no passage", "", ""),
        ("table", "<Table> <Tr> <Td> x y z a b c d e f g h </Td> </Tr> </Table>", ""),
    ]
    path = tmp_path / "nq.csv"
    write_csv(path, rows)
    return path


def test_iter_csv_records_streams_and_audits(csv_file):
    audit = AuditReport()
    records = list(iter_csv_records(csv_file, CFG, audit))
    assert len(records) == 60
    assert audit.rows_total == 62 and audit.no_long_answer == 1 and audit.filtered_context_type == 1
    assert sum(r["has_answer"] for r in records) == 40


def test_limit_rows(csv_file):
    assert len(list(iter_csv_records(csv_file, CFG, limit_rows=10))) == 10


def test_missing_columns_raise(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("question,answer\nq,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        list(iter_csv_records(bad))


def test_very_long_field_does_not_crash(tmp_path):
    path = tmp_path / "huge.csv"
    write_csv(path, [("q", "<P> " + "word " * 300_000 + "</P>", "")])
    audit = AuditReport()
    assert list(iter_csv_records(path, CFG, audit)) == []
    assert audit.filtered_too_long == 1


def test_splits_are_disjoint_by_passage_and_deterministic(csv_file, tmp_path):
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    audit = prepare_splits(csv_file, out_a, CFG, val_ratio=0.2, test_ratio=0.2, seed=7)
    prepare_splits(csv_file, out_b, CFG, val_ratio=0.2, test_ratio=0.2, seed=7)

    ids = {
        s: {r["passage_id"] for r in read_jsonl(out_a / f"{s}.jsonl")}
        for s in ("train", "validation", "test")
    }
    assert (
        not (ids["train"] & ids["validation"])
        and not (ids["train"] & ids["test"])
        and not (ids["validation"] & ids["test"])
    )
    assert sum(audit.split_counts.values()) == 60
    for split in ("train", "validation", "test"):
        assert (out_a / f"{split}.jsonl").read_text() == (out_b / f"{split}.jsonl").read_text()
    assert json.loads((out_a / "audit.json").read_text())["rows_total"] == 62


def test_same_passage_never_straddles_splits():
    splits = {assign_split("abc123", val_ratio=0.3, test_ratio=0.3, seed=1) for _ in range(5)}
    assert len(splits) == 1


def test_assign_split_ratios_are_respected():
    n = 4000
    counts = {"train": 0, "validation": 0, "test": 0}
    for i in range(n):
        counts[assign_split(f"passage-{i}", val_ratio=0.1, test_ratio=0.2, seed=3)] += 1
    assert abs(counts["validation"] / n - 0.1) < 0.03
    assert abs(counts["test"] / n - 0.2) < 0.03


def test_prepare_splits_rejects_bad_ratios(csv_file, tmp_path):
    with pytest.raises(ValueError):
        prepare_splits(csv_file, tmp_path, CFG, val_ratio=0.6, test_ratio=0.5)


def test_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1, "b": "é"}, {"a": 2, "b": "ü"}]
    assert write_jsonl(tmp_path / "x.jsonl", rows) == 2
    assert read_jsonl(tmp_path / "x.jsonl") == rows
    assert read_jsonl(tmp_path / "x.jsonl", limit=1) == rows[:1]


def test_config_from_dict_uses_defaults():
    cfg = CsvParseConfig.from_config({"max_context_words": 99, "context_types": ["P"]})
    assert (
        cfg.max_context_words == 99 and cfg.context_types == ("P",) and cfg.min_context_words == 10
    )


SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "nq_sample.jsonl"


@pytest.mark.skipif(not SAMPLE.exists(), reason="bundled demo sample not present")
def test_bundled_sample_is_internally_consistent():
    records = read_jsonl(SAMPLE)
    assert len(records) >= 100
    for rec in records:
        assert rec["has_answer"]
        for text, start in zip(rec["answers"]["text"], rec["answers"]["answer_start"], strict=True):
            assert rec["context"][start : start + len(text)] == text
