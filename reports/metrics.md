# Evaluation report

- Split: `dev_official` | queries: 1000 | corpus: 7429 passages
- Encoder: `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | reader: `artifacts/models/roberta-finetuned-nq` | device: cuda | generated: 2026-11-02 13:37 UTC

## Retrieval (closed corpus)

| Mode | R@1 | R@5 | R@10 | R@20 | MRR | Ans@1 | Ans@5 | Ans@10 | Ans@20 | ms/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 51.8 | 69.1 | 73.3 | 77.9 | 0.598 | 56.6 | 73.9 | 78.6 | 82.8 | 3.5 |
| dense | 64.0 | 81.7 | 85.7 | 89.8 | 0.719 | 68.9 | 85.5 | 88.7 | 91.4 | 9.3 |
| hybrid | 60.2 | 78.3 | 83.5 | 88.5 | 0.685 | 65.7 | 82.8 | 87.2 | 91.2 | 9.3 |

R@k = gold passage in top-k (%). Ans@k = any top-k passage contains the answer (%).

## Reader on gold passages

| Subset | Exact Match | F1 | Examples |
| --- | --- | --- | --- |
| All | 82.95 | 85.96 | 2000 |
| Answerable | 76.87 | 82.65 | 1042 |
| Unanswerable | 89.56 | - | 958 |

Best null threshold: -3.3239 (F1 86.2 vs 46.9 without abstention)

## End-to-end (question only)

| Retrieval mode | Exact Match | F1 | Avg latency (ms) | Top-k |
| --- | --- | --- | --- | --- |
| bm25 | 48.70 | 53.79 | 153 | 5 |
| dense | 54.90 | 60.67 | 167 | 5 |
| hybrid | 52.20 | 57.40 | 150 | 5 |
