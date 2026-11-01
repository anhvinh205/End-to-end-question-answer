# Changelog

## 1.1.0

Completes the project and fixes correctness issues found while reviewing 1.0.

### Fixed
- **Fake supervision.** The "lightweight fallback" labelled the first sentence of every passage as
  the answer, so the reader learned "answer = first sentence" and its EM/F1 meant nothing. Training
  data now comes from real NQ short answers aligned to character offsets.
- **Uploading a PDF replaced the global knowledge base** for every user (`answer_from_custom_contexts`
  rebuilt the shared index). Ad-hoc documents now use a separate, throw-away index.
- **Mismatched score scales**: raw retriever scores (RRF ~0.03, `1/(1+L2)`) were blended with reader
  probabilities, so the retriever contributed almost nothing. Scores are now normalised first.
- **No-answer was trained but never predicted.** Evaluation excluded `[CLS]` from candidate spans, so
  the model could not abstain although it was trained to. Null-score handling and a tuned threshold added.
- **Train/validation leakage**: in the fallback path the "validation" split was read from the `train`
  split, so the validation rows were a subset of the training rows. Splits are now made by passage hash,
  so one Wikipedia paragraph can never appear in two splits either.
- `evaluate.load("squad_v2")` needed the network at evaluation time; EM/F1 are now implemented in pure Python.
- `transformers>=4.38` was declared but the code needs `>=4.46` (`processing_class`, `eval_strategy`).
- When no index file existed, the API downloaded a dataset from the Hugging Face Hub at start-up, and it
  always loaded `reader.pretrained_checkpoint`, so a fine-tuned reader was never served. It now loads a
  persisted index (or the bundled sample), prefers the fine-tuned checkpoint, and fails fast with an
  actionable message.
- README benchmark numbers were not produced by any code in the repository and have been removed;
  `make evaluate` writes `reports/metrics.{json,md}`.

### Added
- `dense_biencoder` retrieval (mean pooling, normalised vectors) and the `FlatIP` / `IVFFlat` index
  types that `config.yaml` already advertised; index manifest with encoder-mismatch detection.
- Pure-Python BM25 on an inverted index; standard Reciprocal Rank Fusion.
- CSV data pipeline with token-level answer alignment, multi-span answers, audit report and group split.
- Natural + BM25 hard negatives (`negative_sampling_ratio`), reader fine-tuning script with threshold tuning.
- Evaluation harness (Recall@k, MRR, EM/F1, end-to-end), `scripts/evaluate.py`, `scripts/bm25_scaling.py`, `scripts/ask.py`.
- `/api/v1/qa/documents` endpoint, `/health` with model info, request validation, dependency-injectable app.
- 206 offline tests (98 % coverage), `pyproject.toml` (src layout, extras), ruff + mypy, multi-stage Docker
  image, docker-compose (UI as a client of the API), CI matrix for Python 3.10-3.12.

### Changed
- Package imports are `nq_qa...` (installed with `pip install -e .`) instead of `src.nq_qa...`.
- `retriever.mode` is `bm25 | dense | hybrid` (query time); the encoder is chosen with `retriever.dense_encoder`
  (build time). `l2_distance` in API candidates became `dense_score` / `sparse_score`.
