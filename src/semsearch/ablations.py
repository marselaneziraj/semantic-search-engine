"""Ablation runner: change one knob at a time and plot what it costs.

Four sweeps, each answering a question an interviewer will ask:

    chunk  - "what chunk size did you pick and why?"
             Sweeps chunking off / 128 / 256 / 512 tokens.
    topk   - "how deep do you rerank?"
             Sweeps the cross-encoder candidate pool: 10 / 25 / 50 / 100.
    model  - "does a bigger embedding model pay for itself?"
             Sweeps MiniLM-L6 (384d) vs MiniLM-L12 vs mpnet-base (768d).
    index  - "what does quantising the index actually buy?"
             Sweeps FAISS flat vs IVF-PQ for recall / latency / memory.

Every sweep writes a CSV and a PNG into results/ablations/.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from semsearch.config import Settings
from semsearch.data.loaders import Dataset
from semsearch.evaluation.harness import evaluate_systems, results_to_frame

CHUNK_SIZES = [0, 128, 256, 512]          # 0 = chunking disabled
RERANK_DEPTHS = [10, 25, 50, 100]
MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6 (384d)"),
    ("sentence-transformers/all-MiniLM-L12-v2", "MiniLM-L12 (384d)"),
    ("sentence-transformers/all-mpnet-base-v2", "mpnet-base (768d)"),
]


def _clone(settings: Settings) -> Settings:
    return copy.deepcopy(settings)


def ablate_chunk_size(dataset: Dataset, settings: Settings, sizes: Sequence[int] = CHUNK_SIZES,
                      systems: Sequence[str] = ("bm25", "dense")) -> pd.DataFrame:
    rows = []
    for size in sizes:
        cfg = _clone(settings)
        cfg.chunking.enabled = size > 0
        if size > 0:
            cfg.chunking.size = size
            cfg.chunking.overlap = min(cfg.chunking.overlap, size // 4)
        frame = results_to_frame(evaluate_systems(dataset, cfg, include=systems, write_runs=False))
        frame["ablation"] = "chunk_size"
        frame["setting"] = "off" if size == 0 else str(size)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def ablate_rerank_depth(dataset: Dataset, settings: Settings,
                        depths: Sequence[int] = RERANK_DEPTHS) -> pd.DataFrame:
    rows = []
    for depth in depths:
        cfg = _clone(settings)
        cfg.rerank.candidates = depth
        frame = results_to_frame(
            evaluate_systems(dataset, cfg, include=["hybrid+rerank"], write_runs=False)
        )
        frame["ablation"] = "rerank_depth"
        frame["setting"] = str(depth)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def ablate_model_size(dataset: Dataset, settings: Settings,
                      models: Sequence[tuple[str, str]] = MODELS) -> pd.DataFrame:
    rows = []
    for model_name, label in models:
        cfg = _clone(settings)
        cfg.dense.backend = "sentence-transformers"
        cfg.dense.model_name = model_name
        frame = results_to_frame(evaluate_systems(dataset, cfg, include=["dense"], write_runs=False))
        frame["ablation"] = "model_size"
        frame["setting"] = label
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def ablate_index_type(dataset: Dataset, settings: Settings) -> pd.DataFrame:
    """Compare FAISS flat vs IVF-PQ on quality, latency and vector memory."""
    from semsearch.data.chunking import chunk_corpus
    from semsearch.evaluation.harness import run_queries
    from semsearch.evaluation.metrics import evaluate_run
    from semsearch.retrieval.dense import DenseRetriever

    chunks = chunk_corpus(
        dataset.documents,
        size=settings.chunking.size,
        overlap=settings.chunking.overlap,
        enabled=settings.chunking.enabled,
    )
    rows = []
    for index_type in ("flat", "ivfpq"):
        retriever = DenseRetriever(
            backend=settings.dense.backend,
            model_name=settings.dense.model_name,
            batch_size=settings.dense.batch_size,
            hashing_dim=settings.dense.hashing_dim,
            cache_dir=settings.dense.cache_dir,
            index_type=index_type,
        )
        retriever.index(chunks)
        run, latencies, _ = run_queries(retriever, dataset.queries, depth=settings.eval_depth)
        metrics = evaluate_run(run, dataset.qrels, ndcg_k=settings.top_k, recall_k=settings.eval_depth)
        row = {"system": f"dense-{index_type}", "ablation": "index_type", "setting": index_type}
        row.update({k: round(v, 4) for k, v in metrics.items()})
        row["mean_latency_ms"] = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        row["vectors_mb"] = round(retriever.memory_mb, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ablation(frame: pd.DataFrame, path: str | Path, metric: str = "nDCG@10",
                  title: str = "") -> Path:
    """Quality vs. setting on the left axis, latency on the right."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for system, group in frame.groupby("system"):
        ax.plot(group["setting"], group[metric], "o-", label=f"{system} {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("setting")
    ax.set_ylim(0, max(1.0, float(frame[metric].max()) * 1.2))
    ax.grid(alpha=0.3)
    ax.set_title(title or frame["ablation"].iloc[0])

    if "mean_latency_ms" in frame.columns:
        twin = ax.twinx()
        for system, group in frame.groupby("system"):
            twin.plot(group["setting"], group["mean_latency_ms"], "s--", alpha=0.5,
                      label=f"{system} latency")
        twin.set_ylabel("mean latency per query (ms)")

    handles, labels = ax.get_legend_handles_labels()
    if "mean_latency_ms" in frame.columns:
        h2, l2 = twin.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, fontsize=8, loc="best")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
