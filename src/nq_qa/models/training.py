"""Fine-tuning of the extractive reader (needs the ``train`` extra: ``datasets``)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from nq_qa.data.nq_processor import NaturalQuestionsProcessor

logger = logging.getLogger(__name__)

QA_COLUMNS = ("id", "question", "context", "answers")


def build_training_features(
    records: Sequence[dict[str, Any]], processor: NaturalQuestionsProcessor
) -> Any:
    """Tokenise records into an Arrow-backed ``datasets.Dataset`` of training features."""
    from datasets import Dataset  # lazy: keeps the inference image free of `datasets`

    rows = [{col: rec[col] for col in QA_COLUMNS} for rec in records]
    dataset = Dataset.from_list(rows)
    return dataset.map(
        processor.preprocess_training_examples,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenising train set",
    )


def train_reader(
    base_model: str,
    train_dataset: Any,
    output_dir: str | Path,
    *,
    learning_rate: float = 3e-5,
    num_train_epochs: float = 2,
    weight_decay: float = 0.01,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.1,
    fp16: bool = True,
    seed: int = 42,
    max_steps: int = -1,
    logging_steps: int = 50,
    save_checkpoints: bool = True,
    use_cpu: bool = False,
) -> Trainer:
    """Fine-tune ``base_model`` for span extraction and save model + tokenizer.

    Mixed precision is only enabled when CUDA is present, so the same call works on a laptop.
    """
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForQuestionAnswering.from_pretrained(base_model)

    use_fp16 = bool(fp16 and torch.cuda.is_available() and not use_cpu)
    num_steps_hint = max(1, len(train_dataset) // max(1, batch_size * gradient_accumulation_steps))
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        warmup_steps=int(warmup_ratio * num_steps_hint * num_train_epochs),
        fp16=use_fp16,
        seed=seed,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_strategy="epoch" if save_checkpoints else "no",
        save_total_limit=1,
        report_to="none",
        use_cpu=use_cpu,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset, processing_class=tokenizer
    )
    logger.info(
        "Training %s on %d features (fp16=%s, %.1f epochs)",
        base_model,
        len(train_dataset),
        use_fp16,
        num_train_epochs,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer
