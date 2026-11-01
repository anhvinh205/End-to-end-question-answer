# Design notes

Short explanations of the decisions that are not obvious from the code.

## 1. Pipeline

```
question
   |
   +--> BM25 (inverted index) ----------+
   |                                     +--> Reciprocal Rank Fusion --> top-k passages
   +--> bi-encoder --> FAISS (dense) ---+                                      |
                                                                               v
                                           reader reads all k passages in ONE batch
                                           (sliding window, n-best span decoding)
                                                                               |
                                                                               v
              S = alpha * P_reader + (1 - alpha) * S_retriever (min-max over the k passages)
                                                                               |
                                   answer, or "no confident answer" (abstention)
```

## 2. Retrieval

* **Dense and sparse are fused with RRF**, `score(d) = sum_r 1 / (60 + rank_r(d))`. RRF only needs
  ranks, so no score calibration between BM25 (~10) and cosine similarity (~0.5) is needed. A
  document missing from one ranking contributes nothing for it (the textbook definition).
* **BM25 is implemented from the formula** (`models/bm25.py`, ~100 lines, Lucene idf, inverted index)
  rather than wrapping `rank_bm25`: a query only touches the postings of its terms (about 2 ms on
  20k passages), the index builds in seconds at start-up, there is one dependency fewer, and the
  implementation is unit-tested against a hand computation and cross-checked against `rank_bm25`.
* **Encoder choice is a build-time decision, retrieval mode a query-time one.** The index manifest
  stores the encoder name and pooling; loading an index built with a different encoder raises
  `IndexMismatchError` instead of silently returning garbage similarities.
* **The vanilla DistilBERT `[CLS]` vector is kept only as a baseline** (`dense_encoder: cls`). A
  language model that was never trained contrastively is a poor retriever; the default is a
  sentence-transformers bi-encoder with mean pooling. *Caveat:* `multi-qa-MiniLM-L6-cos-v1` was
  trained on QA data that includes Natural Questions, so dense numbers on NQ are optimistic.
* **Answering over an uploaded document builds a throw-away `PassageIndex`.** The shared knowledge
  base is never mutated, so concurrent users cannot corrupt each other's results.

## 3. Reader

* **One decoding path** (`decode_window`) is used by the API, `scripts/train_reader.py` and
  `scripts/evaluate.py`, so reported metrics describe what is actually served.
* **Training windows without the answer, and unanswerable examples, are labelled `[CLS]`** (SQuAD v2).
  The reader therefore learns to say "not in this passage", which matters because most of the top-k
  retrieved passages do not contain the answer.
* **Negatives**: natural ones (a long answer exists, a short answer does not) plus **BM25 hard
  negatives**: for an answerable question, the best-scoring passage that is not the gold passage and
  does not contain the gold answer (false-negative guard). `negative_sampling_ratio` is the share of
  negatives in the training set.
* **Abstention** follows the SQuAD v2 recipe: `null_diff = min_windows(start[CLS] + end[CLS]) - best_span_logit`;
  abstain when `null_diff > threshold`. The threshold maximises F1 on validation data
  (`best_null_threshold`) and is stored in `reader_meta.json` next to the weights. A checkpoint
  without that file (e.g. the public SQuAD v1.1 model, which never saw `[CLS]` labels) never abstains.
* **Span probability** is the product of start and end softmaxes over `{[CLS]} + passage tokens`
  of the window, so span and null probabilities are directly comparable.

## 4. Score calibration

Raw retriever scores live on a different scale for each mode, so `alpha * reader + (1 - alpha) * retriever`
on raw numbers is dominated by whichever is larger. The retriever score is min-max normalised over
the retrieved set first. Candidates the reader would answer are ranked ahead of abstentions.

## 5. Evaluation protocol and its limits

* `evaluate_retrieval`: Recall@k (gold passage id) and answer-recall@k (any retrieved passage contains
  the answer, the DPR "top-k accuracy"), plus MRR.
* `evaluate_reader`: EM / F1 with the SQuAD v2 normalisation, on gold passages (an upper bound).
* `evaluate_end_to_end`: question only; EM / F1 of the final answer.
* **The corpus is closed**: it contains the gold passages of the evaluation questions plus distractors.
  Real open-domain NQ retrieves from ~21M Wikipedia passages. `scripts/bm25_scaling.py` shows how
  quickly BM25 recall falls when the corpus grows (see the README table), so absolute numbers must not
  be compared with published open-domain results.
* Each question has one reference string (possibly several spans). The official NQ metric uses up to
  five annotators; EM / F1 here are therefore stricter than the official ones.

## 6. Testing strategy

* No test needs the network or a GPU. A WordPiece tokenizer is trained on the fly and tiny randomly
  initialised BERT models exercise the same code as DistilBERT / MiniLM.
* `test_fine_tuning_teaches_the_model_to_extract_the_right_span` fine-tunes a 2-layer model on a toy
  set and asserts the decoder recovers the labelled spans: this checks processor labels, training
  loop and decoding against each other, which is where QA code usually breaks.
* `tests/test_scripts.py` runs `prepare_data -> build_index -> train_reader -> evaluate -> ask` as real
  subprocesses on a synthetic CSV, so wiring errors between scripts, config and package are caught.
* The Streamlit app is driven headlessly with `streamlit.testing.v1.AppTest`.
