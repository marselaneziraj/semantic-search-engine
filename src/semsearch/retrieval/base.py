"""Shared retrieval types.

Every retriever in this project implements the same two methods, so the
evaluation harness, the API and the UI never need to know which one they hold:

    index(chunks)            -> build whatever structure the retriever needs
    search(query, top_k)     -> list[Hit] ordered best-first
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from semsearch.data.chunking import Chunk


@dataclass
class Hit:
    """One retrieved chunk."""

    chunk_id: str
    doc_id: str
    score: float
    text: str = ""
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": round(float(self.score), 6),
            "rank": self.rank,
            "text": self.text,
        }


@runtime_checkable
class Retriever(Protocol):
    """Structural interface shared by BM25, dense, hybrid and reranked retrieval."""

    name: str

    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, top_k: int = 10) -> list[Hit]: ...


def timed_search(retriever: "Retriever", query: str, top_k: int = 10) -> tuple[list[Hit], float]:
    """Run a search and return (hits, elapsed_milliseconds).

    time.perf_counter is used rather than time.time because it is monotonic and
    has a much finer resolution - on Windows time.time() can be as coarse as
    16 ms, which is the same order as a BM25 query on a small corpus.
    """
    start = time.perf_counter()
    hits = retriever.search(query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return hits, elapsed_ms


def dedupe_by_document(hits: Sequence[Hit], top_k: int) -> list[Hit]:
    """Keep only the best-scoring chunk per parent document.

    With chunking on, one document can occupy several of the top 10 slots. For
    document-level evaluation (and for a sane UI) only its best chunk counts.
    """
    seen: set[str] = set()
    kept: list[Hit] = []
    for hit in hits:
        if hit.doc_id in seen:
            continue
        seen.add(hit.doc_id)
        kept.append(hit)
        if len(kept) >= top_k:
            break
    for rank, hit in enumerate(kept, start=1):
        hit.rank = rank
    return kept
