"""Hybrid retrieval by Reciprocal Rank Fusion (RRF).

The problem: BM25 scores are unbounded sums of IDF terms (0 ... 40+), cosine
similarities live in [-1, 1]. Adding them, or even min-max normalising them per
query, mixes two incompatible scales and the blend weight has to be tuned per
dataset.

RRF throws the scores away and keeps only the ranks:

    RRF(d) = SUM over systems  1 / (k + rank_system(d))        k = 60

Properties worth being able to say out loud in an interview:
* Scale-free: no normalisation, no tuning, works with any number of systems.
* Rank 1 contributes 1/61, rank 2 contributes 1/62 - a tiny gap, so the fusion
  trusts "both systems liked it" more than "one system loved it".
* The k constant flattens the curve; small k makes the top rank dominate, large
  k makes the lists nearly equal-weighted. 60 is the value from Cormack et al.
  (2009) and is almost never worth tuning.
* A document found by only one system can still surface, which is why hybrid
  usually beats both parents on recall.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from semsearch.data.chunking import Chunk
from semsearch.retrieval.base import Hit


def reciprocal_rank_fusion(
    runs: Sequence[Sequence[Hit]],
    k: int = 60,
    depth: int = 200,
    weights: Sequence[float] | None = None,
    top_k: int = 10,
) -> list[Hit]:
    """Fuse several ranked lists into one.

    Args:
        runs: ranked lists, each already ordered best-first.
        k: RRF smoothing constant.
        depth: how far down each list to read.
        weights: optional per-system weights (default: all 1.0).
        top_k: size of the fused list.
    """
    if weights is None:
        weights = [1.0] * len(runs)
    if len(weights) != len(runs):
        raise ValueError("weights must have one entry per run")

    fused: dict[str, float] = {}
    payload: dict[str, Hit] = {}
    for run, weight in zip(runs, weights):
        for rank, hit in enumerate(run[:depth], start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + weight / (k + rank)
            payload.setdefault(hit.chunk_id, hit)

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    hits: list[Hit] = []
    for rank, (chunk_id, score) in enumerate(ordered[:top_k], start=1):
        source = payload[chunk_id]
        hits.append(Hit(chunk_id, source.doc_id, float(score), source.text, rank))
    return hits


class HybridRetriever:
    """BM25 + dense, fused with RRF."""

    name = "hybrid"

    def __init__(self, lexical, dense, rrf_k: int = 60, depth: int = 200,
                 weights: Sequence[float] | None = None) -> None:
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k
        self.depth = depth
        self.weights = weights

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Index both children. Call only if they are not already indexed -
        the CLI and the harness share one BM25 and one dense index on purpose."""
        self.lexical.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        lexical_hits = self.lexical.search(query, top_k=self.depth)
        dense_hits = self.dense.search(query, top_k=self.depth)
        return reciprocal_rank_fusion(
            [lexical_hits, dense_hits],
            k=self.rrf_k,
            depth=self.depth,
            weights=self.weights,
            top_k=top_k,
        )
