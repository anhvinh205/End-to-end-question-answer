# Evaluation report

- Split: `dev_official` | queries: 1000 | corpus: 7429 passages
- Encoder: `None` | reader: `None` | device: cpu (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel) | generated: 2026-11-02 04:39 UTC

## Retrieval (closed corpus)

| Mode | R@1 | R@5 | R@10 | R@20 | MRR | Ans@1 | Ans@5 | Ans@10 | Ans@20 | ms/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 52.1 | 69.1 | 73.3 | 77.9 | 0.599 | 56.6 | 73.9 | 78.6 | 82.8 | 4.3 |

R@k = gold passage in top-k (%). Ans@k = any top-k passage contains the answer (%).
