"""Render evaluation results as a Markdown report."""

from __future__ import annotations

from typing import Any


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_markdown(results: dict[str, Any]) -> str:
    """Human-readable summary of the dictionary written to ``metrics.json``."""
    meta = results.get("meta", {})
    lines = ["# Evaluation report", ""]
    lines.append(
        f"- Split: `{meta.get('split')}` | queries: {meta.get('num_queries')} | "
        f"corpus: {meta.get('corpus_passages')} passages"
    )
    lines.append(
        f"- Encoder: `{meta.get('encoder')}` | reader: `{meta.get('reader')}` | "
        f"device: {meta.get('device')} | generated: {meta.get('generated_at')}"
    )
    lines.append("")

    retrieval = results.get("retrieval")
    if retrieval:
        ks = sorted(
            {int(k.split("@")[1]) for m in retrieval.values() for k in m if k.startswith("recall@")}
        )
        lines += ["## Retrieval (closed corpus)", ""]
        header = (
            ["Mode"] + [f"R@{k}" for k in ks] + ["MRR"] + [f"Ans@{k}" for k in ks] + ["ms/query"]
        )
        lines.append(_row(header))
        lines.append(_row(["---"] * len(header)))
        for mode, m in retrieval.items():
            cells = [mode]
            cells += [f"{100 * m[f'recall@{k}']:.1f}" for k in ks]
            cells.append(f"{m['mrr']:.3f}")
            cells += [f"{100 * m[f'answer_recall@{k}']:.1f}" for k in ks]
            cells.append(f"{m['avg_latency_ms']:.1f}")
            lines.append(_row(cells))
        lines += [
            "",
            "R@k = gold passage in top-k (%). Ans@k = any top-k passage contains the answer (%).",
            "",
        ]

    reader = results.get("reader")
    if reader:
        lines += ["## Reader on gold passages", ""]
        lines.append(_row(["Subset", "Exact Match", "F1", "Examples"]))
        lines.append(_row(["---", "---", "---", "---"]))
        lines.append(
            _row(
                ["All", f"{reader['exact']:.2f}", f"{reader['f1']:.2f}", f"{int(reader['total'])}"]
            )
        )
        lines.append(
            _row(
                [
                    "Answerable",
                    f"{reader['HasAns_exact']:.2f}",
                    f"{reader['HasAns_f1']:.2f}",
                    f"{int(reader['HasAns_total'])}",
                ]
            )
        )
        lines.append(
            _row(
                [
                    "Unanswerable",
                    f"{reader['NoAns_exact']:.2f}",
                    "-",
                    f"{int(reader['NoAns_total'])}",
                ]
            )
        )
        lines += [
            "",
            f"Best null threshold: {reader['best_null_threshold']} "
            f"(F1 {reader['f1_at_best_threshold']} vs {reader['f1_without_abstention']} without abstention)",
            "",
        ]

    e2e = results.get("end_to_end")
    if e2e:
        lines += ["## End-to-end (question only)", ""]
        lines.append(_row(["Retrieval mode", "Exact Match", "F1", "Avg latency (ms)", "Top-k"]))
        lines.append(_row(["---"] * 5))
        for mode, m in e2e.items():
            lines.append(
                _row(
                    [
                        mode,
                        f"{m['exact']:.2f}",
                        f"{m['f1']:.2f}",
                        f"{m['avg_latency_ms']:.0f}",
                        f"{int(m['top_k'])}",
                    ]
                )
            )
        lines.append("")
    return "\n".join(lines)
