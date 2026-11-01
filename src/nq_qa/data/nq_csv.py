"""Turn the Natural Questions CSV export into clean SQuAD-v2-style records.

The CSV has three columns - ``question``, ``long_answers`` (the passage) and
``short_answers`` (the answer, possibly several comma-joined spans).  Roughly half of the
rows have no passage at all and a further ~15 % have a passage but no short answer, so
turning the file into training data needs care:

* HTML tokens are stripped and the passage is detokenised **with token -> character
  offsets**, so answers are aligned at token level and mapped to exact character spans.
* Multi-span answers are split and every span that can be located is kept
  (``answers`` is a list, exactly like SQuAD).
* Every discarded row is accounted for in an :class:`AuditReport`.
* Train / validation / test are split **by passage hash**, not by question: several
  questions share the same Wikipedia paragraph and a per-question split would leak it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nq_qa.data.text import (
    detokenize,
    find_token_span,
    first_html_tag,
    split_multi_span,
    strip_html_tokens,
)

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TYPES: tuple[str, ...] = ("P", "Ul", "Ol", "Li", "Dl", "Dd")
REQUIRED_COLUMNS = ("question", "long_answers", "short_answers")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class CsvParseConfig:
    """Filters applied while parsing the CSV."""

    context_types: tuple[str, ...] = DEFAULT_CONTEXT_TYPES
    min_context_words: int = 10
    max_context_words: int = 350
    max_answer_words: int = 20
    keep_unanswered: bool = True

    @classmethod
    def from_config(cls, dataset_cfg: Mapping[str, Any]) -> CsvParseConfig:
        defaults = cls()
        return cls(
            context_types=tuple(dataset_cfg.get("context_types", defaults.context_types)),
            min_context_words=int(dataset_cfg.get("min_context_words", defaults.min_context_words)),
            max_context_words=int(dataset_cfg.get("max_context_words", defaults.max_context_words)),
            max_answer_words=int(dataset_cfg.get("max_answer_words", defaults.max_answer_words)),
            keep_unanswered=bool(dataset_cfg.get("keep_unanswered", defaults.keep_unanswered)),
        )


@dataclass
class AuditReport:
    """Where every CSV row went.  Written next to the processed data as ``audit.json``."""

    rows_total: int = 0
    no_long_answer: int = 0
    filtered_context_type: int = 0
    filtered_too_short: int = 0
    filtered_too_long: int = 0
    candidates_with_passage: int = 0  # rows that survived the passage filters
    with_short_answer: int = 0
    unanswerable_kept: int = 0
    unanswerable_dropped: int = 0
    aligned: int = 0
    aligned_multi_span: int = 0
    unaligned: int = 0
    answer_too_long: int = 0
    context_type_counts: Counter[str] = field(default_factory=Counter)
    split_counts: Counter[str] = field(default_factory=Counter)
    split_answerable: Counter[str] = field(default_factory=Counter)

    @property
    def alignment_rate(self) -> float:
        """Share of short answers that could be located inside their passage."""
        if not self.with_short_answer:
            return 0.0
        return (self.with_short_answer - self.unaligned) / self.with_short_answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_total": self.rows_total,
            "no_long_answer": self.no_long_answer,
            "filtered_context_type": self.filtered_context_type,
            "filtered_too_short": self.filtered_too_short,
            "filtered_too_long": self.filtered_too_long,
            "candidates_with_passage": self.candidates_with_passage,
            "with_short_answer": self.with_short_answer,
            "unanswerable_kept": self.unanswerable_kept,
            "unanswerable_dropped": self.unanswerable_dropped,
            "aligned": self.aligned,
            "aligned_multi_span": self.aligned_multi_span,
            "unaligned": self.unaligned,
            "answer_too_long": self.answer_too_long,
            "alignment_rate": round(self.alignment_rate, 4),
            "context_type_counts": dict(self.context_type_counts),
            "split_counts": dict(self.split_counts),
            "split_answerable": dict(self.split_answerable),
        }


def passage_id_for(context: str) -> str:
    """Stable 16-hex-char id derived from the passage text."""
    return hashlib.sha1(context.encode("utf-8")).hexdigest()[:16]


def _locate_answer_spans(tokens: list[str], short_raw: str) -> tuple[list[tuple[int, int]], bool]:
    """Find the short answer in the passage tokens.

    Returns ``(spans, multi_span)`` with inclusive token spans sorted by position.
    The whole answer is tried first (exact, then case-insensitive); when it is not
    contiguous it is split on the NQ span separator and each part is located separately.
    """
    whole = strip_html_tokens(short_raw)
    if whole:
        for ignore_case in (False, True):
            idx = find_token_span(tokens, whole, ignore_case=ignore_case)
            if idx >= 0:
                return [(idx, idx + len(whole) - 1)], False

    parts = split_multi_span(short_raw)
    if len(parts) < 2:
        return [], False
    spans: set[tuple[int, int]] = set()
    for part in parts:
        part_tokens = strip_html_tokens(part)
        if not part_tokens:
            continue
        idx = find_token_span(tokens, part_tokens)
        if idx < 0:
            idx = find_token_span(tokens, part_tokens, ignore_case=True)
        if idx >= 0:
            spans.add((idx, idx + len(part_tokens) - 1))
    return sorted(spans), True


def parse_row(
    row_id: int,
    question: str,
    long_raw: str,
    short_raw: str,
    cfg: CsvParseConfig,
    audit: AuditReport,
) -> dict[str, Any] | None:
    """Convert one CSV row into a record, or ``None`` when it is filtered out."""
    audit.rows_total += 1
    question = " ".join(question.split())
    long_raw, short_raw = long_raw.strip(), short_raw.strip()

    if not question or not long_raw:
        audit.no_long_answer += 1
        return None

    tag = first_html_tag(long_raw)
    if tag != "plain" and tag not in cfg.context_types:
        audit.filtered_context_type += 1
        return None

    tokens = strip_html_tokens(long_raw)
    if len(tokens) < cfg.min_context_words:
        audit.filtered_too_short += 1
        return None
    if len(tokens) > cfg.max_context_words:
        audit.filtered_too_long += 1
        return None

    audit.candidates_with_passage += 1
    audit.context_type_counts[tag] += 1
    context, char_spans = detokenize(tokens)

    answer_texts: list[str] = []
    answer_starts: list[int] = []
    if short_raw:
        audit.with_short_answer += 1
        token_spans, multi_span = _locate_answer_spans(tokens, short_raw)
        if not token_spans:
            audit.unaligned += 1
            return None
        usable = [(i, j) for i, j in token_spans if j - i + 1 <= cfg.max_answer_words]
        if not usable:
            audit.answer_too_long += 1
            return None
        for i, j in usable:
            start, end = char_spans[i][0], char_spans[j][1]
            answer_texts.append(context[start:end])
            answer_starts.append(start)
        audit.aligned += 1
        audit.aligned_multi_span += int(multi_span and len(usable) > 1)
    elif cfg.keep_unanswered:
        audit.unanswerable_kept += 1
    else:
        audit.unanswerable_dropped += 1
        return None

    return {
        "id": f"nq-{row_id}",
        "question": question,
        "context": context,
        "passage_id": passage_id_for(context),
        "answers": {"text": answer_texts, "answer_start": answer_starts},
        "has_answer": bool(answer_texts),
        "context_type": tag,
    }


def _raise_csv_field_limit() -> None:
    """Passages can be far longer than the csv module's default 128 KiB field limit."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def iter_csv_records(
    path: str | Path,
    cfg: CsvParseConfig | None = None,
    audit: AuditReport | None = None,
    limit_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream cleaned records from an NQ CSV without loading it into memory."""
    cfg = cfg or CsvParseConfig()
    audit = audit if audit is not None else AuditReport()
    _raise_csv_field_limit()
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        for row_id, row in enumerate(reader):
            if limit_rows is not None and row_id >= limit_rows:
                break
            record = parse_row(
                row_id,
                row["question"] or "",
                row["long_answers"] or "",
                row["short_answers"] or "",
                cfg,
                audit,
            )
            if record is not None:
                yield record


def assign_split(
    passage_id: str, *, val_ratio: float = 0.05, test_ratio: float = 0.05, seed: int = 42
) -> str:
    """Deterministically map a passage to ``train`` / ``validation`` / ``test``."""
    digest = hashlib.sha1(f"{seed}:{passage_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "validation"
    return "train"


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write records as JSON lines and return how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read (at most ``limit``) records from a JSON-lines file."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def prepare_splits(
    csv_path: str | Path,
    out_dir: str | Path,
    cfg: CsvParseConfig | None = None,
    *,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    limit_rows: int | None = None,
) -> AuditReport:
    """Parse ``csv_path`` and write ``train/validation/test.jsonl`` + ``audit.json``."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditReport()
    handles = {name: (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") for name in SPLITS}
    try:
        for record in iter_csv_records(csv_path, cfg, audit, limit_rows):
            split = assign_split(
                record["passage_id"], val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
            )
            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            audit.split_counts[split] += 1
            audit.split_answerable[split] += int(record["has_answer"])
    finally:
        for handle in handles.values():
            handle.close()
    (out_dir / "audit.json").write_text(
        json.dumps(audit.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote %s", dict(audit.split_counts))
    return audit
