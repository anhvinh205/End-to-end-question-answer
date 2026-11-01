"""Tokenisation of QA records for Transformer span extraction.

Works on plain ``dict``-of-lists batches so it can be used directly with
``datasets.Dataset.map(batched=True)`` (training) or in unit tests without ``datasets``.

Records follow the SQuAD v2 convention::

    {"id", "question", "context", "answers": {"text": [...], "answer_start": [...]}}

An example without answers (``answers.text == []``) is *unanswerable*: its label is the
``[CLS]`` position ``(0, 0)``.  The same label is used for sliding windows that do not
contain the answer.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from typing import Any

from transformers import PreTrainedTokenizerBase

from nq_qa.data.nq_csv import passage_id_for
from nq_qa.data.text import detokenize

logger = logging.getLogger(__name__)


class NaturalQuestionsProcessor:
    """Parse raw NQ records and tokenise QA examples with a sliding window."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 384,
        stride: int = 128,
    ) -> None:
        if stride >= max_length:
            raise ValueError("stride must be smaller than max_length")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

    # ------------------------------------------------------------------ raw HF records
    @staticmethod
    def parse_raw_nq_record(
        record: dict[str, Any], min_context_tokens: int = 30
    ) -> dict[str, Any] | None:
        """Convert a record of ``google-research-datasets/natural_questions`` to QA format.

        The annotated long answer becomes the context and the first short answer that
        falls inside it becomes the label.  Examples without a long answer are kept as
        *unanswerable* by pairing the question with the first large top-level candidate.
        Returns ``None`` for malformed records.
        """
        try:
            question = record["question"]["text"].strip()
            doc = record["document"]["tokens"]
            tokens: list[str] = doc["token"]
            is_html: list[bool] = doc["is_html"]
            annotations = record["annotations"]

            # First annotation that actually has a long answer.
            long_start = long_end = -1
            ann_idx = -1
            for idx, long_answer in enumerate(annotations["long_answer"]):
                if long_answer.get("start_token", -1) != -1:
                    long_start = long_answer["start_token"]
                    long_end = long_answer["end_token"]
                    ann_idx = idx
                    break

            if long_start == -1:  # unanswerable: borrow a substantial top-level candidate
                cands = record["long_answer_candidates"]
                for s_tok, e_tok, top in zip(
                    cands["start_token"], cands["end_token"], cands["top_level"], strict=False
                ):
                    if top and (e_tok - s_tok) >= min_context_tokens:
                        long_start, long_end = s_tok, e_tok
                        break
            if long_start == -1:
                return None

            kept = [
                (idx, tokens[idx])
                for idx in range(long_start, min(long_end, len(tokens)))
                if not is_html[idx]
            ]
            if not kept:
                return None
            context, spans = detokenize([tok for _, tok in kept])
            if len(context) < 20:
                return None
            char_of = {idx: span for (idx, _), span in zip(kept, spans, strict=True)}

            answer_texts: list[str] = []
            answer_starts: list[int] = []
            if ann_idx >= 0 and ann_idx < len(annotations["short_answers"]):
                short = annotations["short_answers"][ann_idx]
                for s_tok, e_tok in zip(short["start_token"], short["end_token"], strict=False):
                    inside = [t for t in range(s_tok, e_tok) if t in char_of]
                    if not inside:
                        continue
                    start, end = char_of[inside[0]][0], char_of[inside[-1]][1]
                    answer_texts.append(context[start:end])
                    answer_starts.append(start)

            rec_id = str(record.get("id") or hashlib.md5(question.encode()).hexdigest())
            first = tokens[long_start] if long_start < len(tokens) else ""
            return {
                "id": rec_id,
                "question": question,
                "context": context,
                "passage_id": passage_id_for(context),
                "answers": {"text": answer_texts, "answer_start": answer_starts},
                "has_answer": bool(answer_texts),
                "context_type": first.strip("<>/").split("_")[0]
                if is_html[long_start]
                else "plain",
            }
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            logger.debug("Skipping malformed NQ record: %s", exc)
            return None

    @classmethod
    def iter_hf_nq(
        cls, split: str = "train", max_samples: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Stream the official dataset from the Hugging Face Hub (needs ``datasets``).

        The raw dataset is ~40 GB, hence streaming.  Prefer the CSV route
        (:mod:`nq_qa.data.nq_csv`) for local work.
        """
        from datasets import load_dataset  # lazy: only needed for this optional path

        stream = load_dataset(
            "google-research-datasets/natural_questions", split=split, streaming=True
        )
        produced = 0
        for item in stream:
            parsed = cls.parse_raw_nq_record(item)
            if parsed is None:
                continue
            yield parsed
            produced += 1
            if max_samples is not None and produced >= max_samples:
                return

    # --------------------------------------------------------------- tokenisation
    def _tokenize(self, questions: list[str], contexts: list[str], padding: str) -> Any:
        return self.tokenizer(
            questions,
            contexts,
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=padding,
        )

    def preprocess_training_examples(self, examples: dict[str, list[Any]]) -> dict[str, Any]:
        """Tokenise with a sliding window and compute ``start_positions`` / ``end_positions``.

        Windows that do not contain the answer (or examples without answer) get ``(0, 0)``,
        i.e. the ``[CLS]`` token, which teaches the model to say "not in this passage".
        """
        questions = [q.strip() for q in examples["question"]]
        inputs = self._tokenize(questions, examples["context"], padding="max_length")

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]

        start_positions: list[int] = []
        end_positions: list[int] = []

        for i, offsets in enumerate(offset_mapping):
            sequence_ids = inputs.sequence_ids(i)
            context_tokens = [k for k, sid in enumerate(sequence_ids) if sid == 1]
            answer = answers[sample_map[i]]

            if not context_tokens or len(answer["text"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            context_start, context_end = context_tokens[0], context_tokens[-1]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
                start_positions.append(0)  # answer is not (fully) inside this window
                end_positions.append(0)
                continue

            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(self, examples: dict[str, list[Any]]) -> dict[str, Any]:
        """Tokenise for evaluation, keeping ``example_id`` and context-only offsets."""
        questions = [q.strip() for q in examples["question"]]
        inputs = self._tokenize(questions, examples["context"], padding="max_length")

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids: list[str] = []

        for i in range(len(inputs["input_ids"])):
            example_ids.append(examples["id"][sample_map[i]])
            sequence_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [
                offset if sequence_ids[k] == 1 else None
                for k, offset in enumerate(inputs["offset_mapping"][i])
            ]

        inputs["example_id"] = example_ids
        return inputs
