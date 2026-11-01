# Evaluation report

- Split: `validation` | queries: 4135 | corpus: 20000 passages
- Encoder: `None` | reader: `None` | device: cpu (x86_64) | generated: 2026-11-01 09:35 UTC

## Retrieval (closed corpus)

| Mode | R@1 | R@5 | R@10 | R@20 | MRR | Ans@1 | Ans@5 | Ans@10 | Ans@20 | ms/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 46.5 | 70.5 | 76.4 | 80.9 | 0.570 | 52.6 | 74.7 | 80.2 | 84.4 | 1.9 |

R@k = gold passage in top-k (%). Ans@k = any top-k passage contains the answer (%).
