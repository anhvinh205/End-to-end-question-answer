"""Shared fixtures.

The suite never touches the network: a tiny WordPiece tokenizer is trained on the fly and
tiny randomly-initialised BERT models are built from a config.  They are useless as language
models but exercise exactly the same code paths as DistilBERT / MiniLM do in production.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertModel,
    PreTrainedTokenizerFast,
)

CAPITALS = {
    "france": "paris", "japan": "tokyo", "italy": "rome", "spain": "madrid",
    "egypt": "cairo", "kenya": "nairobi", "peru": "lima", "chile": "santiago",
    "norway": "oslo", "greece": "athens", "cuba": "havana", "iran": "tehran",
}  # fmt: skip
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def make_qa_examples() -> list[dict[str, Any]]:
    """Synthetic 'capital of X' QA records in the project's record format."""
    examples = []
    for i, (country, city) in enumerate(CAPITALS.items()):
        context = f"the capital of {country} is {city} and it is a large city near the coast."
        examples.append(
            {
                "id": f"cap-{i}",
                "question": f"what is the capital of {country}?",
                "context": context,
                "passage_id": f"p{i}",
                "answers": {"text": [city], "answer_start": [context.index(city)]},
                "has_answer": True,
                "context_type": "P",
            }
        )
    return examples


@pytest.fixture(scope="session")
def qa_examples() -> list[dict[str, Any]]:
    return make_qa_examples()


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny_tokenizer")
    corpus = [e["context"] for e in make_qa_examples()] + [
        e["question"] for e in make_qa_examples()
    ]
    corpus += ["who wrote the book", "tell me about the history of the world wide web"]
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(
        corpus, trainers.WordPieceTrainer(vocab_size=400, special_tokens=SPECIAL_TOKENS)
    )
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    tok.decoder = decoders.WordPiece()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]",
        model_input_names=["input_ids", "token_type_ids", "attention_mask"],
    )  # fmt: skip
    fast.save_pretrained(path)
    return path


def _bert_config(vocab_size: int) -> BertConfig:
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=256,
    )


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_tokenizer_dir)


@pytest.fixture()
def tiny_qa_dir(tmp_path: Path, tiny_tokenizer: PreTrainedTokenizerFast) -> Path:
    """A fresh, randomly initialised tiny QA model directory (function scoped: tests train it)."""
    import torch

    torch.manual_seed(0)
    BertForQuestionAnswering(_bert_config(len(tiny_tokenizer))).save_pretrained(tmp_path)
    tiny_tokenizer.save_pretrained(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def tiny_encoder_dir(
    tmp_path_factory: pytest.TempPathFactory, tiny_tokenizer: PreTrainedTokenizerFast
) -> Path:
    import torch

    path = tmp_path_factory.mktemp("tiny_encoder")
    torch.manual_seed(0)
    BertModel(_bert_config(len(tiny_tokenizer))).save_pretrained(path)
    tiny_tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def trained_qa_dir(
    tmp_path_factory: pytest.TempPathFactory, tiny_tokenizer: PreTrainedTokenizerFast
) -> Path:
    """Tiny QA model fine-tuned until it reproduces the toy training set (~10 s on CPU).

    Session scoped and treated as read-only: tests that need to modify it must copy it.
    """
    pytest.importorskip("datasets")
    import torch

    from nq_qa.data.nq_processor import NaturalQuestionsProcessor
    from nq_qa.models.training import build_training_features, train_reader

    base = tmp_path_factory.mktemp("tiny_base")
    torch.manual_seed(0)
    BertForQuestionAnswering(_bert_config(len(tiny_tokenizer))).save_pretrained(base)
    tiny_tokenizer.save_pretrained(base)

    processor = NaturalQuestionsProcessor(tiny_tokenizer, max_length=48, stride=12)
    features = build_training_features(make_qa_examples(), processor)
    output = tmp_path_factory.mktemp("tiny_trained")
    train_reader(
        str(base),
        features,
        output,
        learning_rate=3e-3,
        num_train_epochs=80,
        batch_size=6,
        weight_decay=0.0,
        warmup_ratio=0.0,
        fp16=False,
        seed=0,
        logging_steps=1000,
        save_checkpoints=False,
        use_cpu=True,
    )
    return output
