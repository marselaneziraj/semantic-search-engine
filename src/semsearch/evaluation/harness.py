"""The evaluation harness: one command runs all four systems on one dataset.

What "properly evaluated" means here, and what most demo repos get wrong:

* The same chunks, the same queries and the same judgements go to every system.
* Retrieval is done to `eval_depth` (100) so Recall@100 is measurable, and the
  ranking is cut to top_k only for the nDCG/MRR cutoffs.
* Results are reported per document, not per chunk: with chunking on, several
  chunks of the same paper would otherwise fill the top 10 and inflate nDCG.
* Latency is reported as mean *and* p95 over per-query timings. A mean hides the
  tail, and the tail is what users feel.
* Index build time is recorded separately from query time, because they are paid
  at different moments (once offline vs. on every request).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from semsearch.config import Settings
from semsearch.data.chunking import Chunk, chunk_corpus
from semsearch.data.loaders import Dataset
from semsearch.evaluation.metrics import evaluate_run, evaluate_run_ir_measures
from semsearch.retrieval.base import Hit, dedupe_by_document, timed_search
from semsearch.retrieval.bm25 import BM25Retriever
from semsearch.retrieval.dense import DenseRetriever
from semsearch.retrieval.hybrid import HybridRetriever
from semsearch.retrieval.rerank import CrossEncoderReranker, RerankingRetriever


@dataclass
class EvaluationResult:
    """One row of the results table."""

    system: str
    dataset: str
    n_queries: int
    n_documents: int
    n_chunks: int
    metrics: dict[str, float] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    index_seconds: float = 0.0
    notes: str = ""

    def to_row(self) -> dict:
        row = {
            "system": self.system,
            "dataset": self.dataset,
            "n_queries": self.n_queries,
            "n_documents": self.n_documents,
            "n_chunks": self.n_chunks,
        }
        row.update({k: round(v, 4) for k, v in self.metrics.items()})
        row["mean_latency_ms"] = round(self.mean_latency_ms, 2)
        row["p95_latency_ms"] = round(self.p95_latency_ms, 2)
        row["index_seconds"] = round(self.index_seconds, 2)
        row["notes"] = self.notes
        return row


def build_systems(
    settings: Settings,
    chunks: Sequence[Chunk],
    include: Sequence[str] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, object], dict[str, float]]:
    """Build the requested retrievers, sharing one BM25 and one dense index.

    Returns (systems, index_seconds_per_system). Build cost is attributed to
    every system that depends on it, which is why hybrid's index time is the sum
    of BM25's and dense's.
    """
    include = list(include) if include else ["bm25", "dense", "hybrid", "hybrid+rerank"]
    needs_dense = any(name in include for name in ("dense", "hybrid", "hybrid+rerank"))
    needs_lexical = any(name in include for name in ("bm25", "hybrid", "hybrid+rerank"))

    systems: dict[str, object] = {}
    cost: dict[str, float] = {}
    bm25_seconds = dense_seconds = 0.0

    bm25 = dense = None
    if needs_lexical:
        if verbose:
            print(f"[index] BM25 over {len(chunks):,} chunks ...", flush=True)
        start = time.perf_counter()
        bm25 = BM25Retriever()
        bm25.index(chunks)
        bm25_seconds = time.perf_counter() - start
        if verbose:
            print(f"[index] BM25 done in {bm25_seconds:.1f}s", flush=True)

    if needs_dense:
        if verbose:
            print(f"[index] dense ({settings.dense.backend}) over {len(chunks):,} chunks ...", flush=True)
        start = time.perf_counter()
        dense = DenseRetriever(
            backend=settings.dense.backend,
            model_name=settings.dense.model_name,
            batch_size=settings.dense.batch_size,
            normalize=settings.dense.normalize,
            hashing_dim=settings.dense.hashing_dim,
            cache_dir=settings.dense.cache_dir,
        )
        dense.index(chunks, show_progress=verbose)
        dense_seconds = time.perf_counter() - start
        if verbose:
            cached = getattr(dense.embedder, "last_cache_hit", None)
            print(
                f"[index] dense done in {dense_seconds:.1f}s "
                f"({'cache hit' if cached else 'embedded fresh'}, {dense.memory_mb:.1f} MB of vectors)",
                flush=True,
            )

    if "bm25" in include:
        systems["bm25"] = bm25
        cost["bm25"] = bm25_seconds
    if "dense" in include:
        systems["dense"] = dense
        cost["dense"] = dense_seconds
    if "hybrid" in include or "hybrid+rerank" in include:
        hybrid = HybridRetriever(bm25, dense, rrf_k=settings.hybrid.rrf_k, depth=settings.hybrid.depth)
        if "hybrid" in include:
            systems["hybrid"] = hybrid
            cost["hybrid"] = bm25_seconds + dense_seconds
        if "hybrid+rerank" in include:
            reranker = CrossEncoderReranker(
                backend=settings.rerank.backend,
                model_name=settings.rerank.model_name,
                batch_size=settings.rerank.batch_size,
            )
            systems["hybrid+rerank"] = RerankingRetriever(
                hybrid, reranker, candidates=settings.rerank.candidates
            )
            cost["hybrid+rerank"] = bm25_seconds + dense_seconds
    return systems, cost


def run_queries(
    retriever,
    queries: Mapping[str, str],
    depth: int = 100,
    top_k: int = 10,
) -> tuple[dict[str, list[str]], list[float], dict[str, list[Hit]]]:
    """Execute every query, returning (run, latencies_ms, hits_per_query)."""
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    hits_by_query: dict[str, list[Hit]] = {}
    for qid, text in queries.items():
        hits, elapsed_ms = timed_search(retriever, text, top_k=depth)
        latencies.append(elapsed_ms)
        documents = dedupe_by_document(hits, depth)
        run[qid] = [hit.doc_id for hit in documents]
        hits_by_query[qid] = documents[:top_k]
    return run, latencies, hits_by_query


def evaluate_systems(
    dataset: Dataset,
    settings: Settings,
    include: Sequence[str] | None = None,
    write_runs: bool = True,
    verbose: bool = True,
) -> list[EvaluationResult]:
    """Index, search and score every system on one dataset."""
    if not dataset.has_judgements:
        raise ValueError(
            f"{dataset.name} has no qrels, so it cannot be evaluated. "
            "Use `semsearch search` for an unjudged corpus."
        )

    chunks = chunk_corpus(
        dataset.documents,
        size=settings.chunking.size,
        overlap=settings.chunking.overlap,
        min_chars=settings.chunking.min_chars,
        enabled=settings.chunking.enabled,
    )
    systems, index_cost = build_systems(settings, chunks, include=include, verbose=verbose)

    results: list[EvaluationResult] = []
    runs_dir = Path(settings.results_dir) / "runs"
    if write_runs:
        runs_dir.mkdir(parents=True, exist_ok=True)

    for name, retriever in systems.items():
        if verbose:
            print(f"[search] {name}: {len(dataset.queries)} queries ...", flush=True)
        run, latencies, _ = run_queries(
            retriever, dataset.queries, depth=settings.eval_depth, top_k=settings.top_k
        )
        metrics = evaluate_run(run, dataset.qrels, ndcg_k=settings.top_k, recall_k=settings.eval_depth)
        strict = evaluate_run_ir_measures(
            run, dataset.qrels, ndcg_k=settings.top_k, recall_k=settings.eval_depth
        )
        notes = ""
        if strict:
            # ir_measures is the reference implementation; if the two ever drift,
            # the reference wins and the difference is recorded in the table.
            drift = max(abs(strict[k] - metrics[k]) for k in strict)
            metrics.update(strict)
            notes = f"ir_measures agreement: max delta {drift:.4f}"
        else:
            notes = "internal metrics (ir_measures not installed)"

        results.append(
            EvaluationResult(
                system=name,
                dataset=dataset.name,
                n_queries=len(dataset.queries),
                n_documents=len(dataset.documents),
                n_chunks=len(chunks),
                metrics=metrics,
                mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
                p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
                index_seconds=index_cost.get(name, 0.0),
                notes=notes,
            )
        )
        if write_runs:
            _write_trec_run(runs_dir / f"{_slug(dataset.name)}_{_slug(name)}.run", run, name)
    return results


def results_to_frame(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in results])


def save_results(results: Sequence[EvaluationResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_to_frame(results).to_csv(path, index=False)
    return path


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text).strip("-").replace("--", "-")


def _write_trec_run(path: Path, run: Mapping[str, Sequence[str]], system: str) -> None:
    """Write a TREC-format run file: qid Q0 docid rank score tag.

    This is the lingua franca of IR evaluation - trec_eval, pytrec_eval and
    ir_measures all read it, so anyone can re-score your results independently.
    """
    with path.open("w", encoding="utf-8") as handle:
        for qid, documents in run.items():
            for rank, doc_id in enumerate(documents, start=1):
                handle.write(f"{qid} Q0 {doc_id} {rank} {1000 - rank} {system}\n")
