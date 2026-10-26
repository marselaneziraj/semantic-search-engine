"""Turn a results table into a markdown table and a comparison chart."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: works in CI and over SSH
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METRIC_ORDER = ["nDCG@10", "Recall@100", "MRR@10", "MAP@100"]
SYSTEM_ORDER = ["bm25", "dense", "hybrid", "hybrid+rerank"]


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    key = {name: i for i, name in enumerate(SYSTEM_ORDER)}
    frame = frame.copy()
    frame["_order"] = frame["system"].map(lambda s: key.get(s, 99))
    return frame.sort_values("_order").drop(columns="_order")


def render_markdown_table(frame: pd.DataFrame) -> str:
    """Render the results as a GitHub-flavoured markdown table."""
    frame = _ordered(frame)
    metrics = [m for m in METRIC_ORDER if m in frame.columns]
    columns = ["system", *metrics, "mean_latency_ms", "p95_latency_ms", "index_seconds"]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) and column in metrics else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_results(frame: pd.DataFrame, path: str | Path, title: str = "") -> Path:
    """Grouped metric bars + a latency panel, saved as PNG."""
    frame = _ordered(frame)
    metrics = [m for m in METRIC_ORDER[:3] if m in frame.columns]
    systems = frame["system"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    width = 0.8 / max(len(metrics), 1)
    for i, metric in enumerate(metrics):
        offsets = [x + i * width - 0.4 + width / 2 for x in range(len(systems))]
        bars = axes[0].bar(offsets, frame[metric].tolist(), width=width, label=metric)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    axes[0].set_xticks(range(len(systems)))
    axes[0].set_xticklabels(systems, rotation=15)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("score")
    axes[0].set_title(title or "Retrieval quality")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    if "mean_latency_ms" in frame.columns:
        bars = axes[1].bar(systems, frame["mean_latency_ms"].tolist(), color="#8c8c8c", label="mean")
        axes[1].bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
        if "p95_latency_ms" in frame.columns:
            axes[1].plot(systems, frame["p95_latency_ms"].tolist(), "o--", color="#d62728", label="p95")
        axes[1].set_ylabel("latency per query (ms)")
        axes[1].set_yscale("log")
        axes[1].set_title("Query latency (log scale)")
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(frame: pd.DataFrame, results_dir: str | Path, dataset: str, config_line: str = "") -> dict:
    """Write results.csv-adjacent markdown + png for one dataset."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch if ch.isalnum() else "-" for ch in dataset).strip("-")

    table = render_markdown_table(frame)
    png = plot_results(frame, results_dir / f"{slug}_comparison.png", title=f"Retrieval quality - {dataset}")
    md_path = results_dir / f"{slug}_results.md"
    md_path.write_text(
        f"# Results - {dataset}\n\n"
        f"{('Configuration: `' + config_line + '`') if config_line else ''}\n\n"
        f"{table}\n\n"
        f"![comparison]({png.name})\n",
        encoding="utf-8",
    )
    return {"markdown": md_path, "figure": png, "table": table}
