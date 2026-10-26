"""FastAPI service exposing /search over the configured corpus.

Start it with:
    semsearch serve --dataset toy --system hybrid
    # or: uvicorn semsearch.api:app --reload

The index is built once at startup (FastAPI lifespan) and kept in memory - a
per-request rebuild would be the classic demo-repo mistake.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from semsearch.config import Settings
from semsearch.data.chunking import chunk_corpus
from semsearch.data.loaders import load_dataset
from semsearch.evaluation.harness import build_systems
from semsearch.retrieval.base import dedupe_by_document, timed_search
from semsearch.snippets import make_snippet

STATE: dict[str, Any] = {"systems": {}, "settings": None, "dataset": None}


class SearchHit(BaseModel):
    rank: int
    doc_id: str
    chunk_id: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    system: str
    took_ms: float = Field(description="Server-side retrieval time, excluding HTTP overhead.")
    total: int
    hits: list[SearchHit]


def build_state(settings: Settings | None = None, system: str | None = None) -> dict[str, Any]:
    """Load the corpus and build the retrievers. Called at startup and in tests."""
    settings = settings or Settings.from_yaml(os.environ.get("SEMSEARCH_CONFIG"))
    system = system or os.environ.get("SEMSEARCH_SYSTEM", "hybrid")
    dataset = load_dataset(settings.dataset, max_docs=settings.max_docs,
                           max_queries=settings.max_queries, split=settings.split_queries)
    chunks = chunk_corpus(
        dataset.documents,
        size=settings.chunking.size,
        overlap=settings.chunking.overlap,
        min_chars=settings.chunking.min_chars,
        enabled=settings.chunking.enabled,
    )
    include = [system] if system != "all" else ["bm25", "dense", "hybrid", "hybrid+rerank"]
    systems, _ = build_systems(settings, chunks, include=include, verbose=False)
    STATE.update({"systems": systems, "settings": settings, "dataset": dataset, "chunks": len(chunks)})
    return STATE


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_state()
    yield
    STATE["systems"] = {}


app = FastAPI(
    title="semantic-search-engine",
    version="0.1.0",
    description="BM25, dense, hybrid (RRF) and cross-encoder reranked retrieval behind one endpoint.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if STATE["systems"] else "empty",
        "systems": sorted(STATE["systems"]),
        "documents": len(STATE["dataset"].documents) if STATE.get("dataset") else 0,
        "chunks": STATE.get("chunks", 0),
    }


@app.get("/config")
def config() -> dict:
    settings: Settings | None = STATE.get("settings")
    return {"config": settings.describe() if settings else "not loaded"}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="The query text."),
    system: str | None = Query(None, description="bm25 | dense | hybrid | hybrid+rerank"),
    top_k: int = Query(10, ge=1, le=100),
    highlight: bool = Query(True, description="Wrap query terms in <mark> tags."),
) -> SearchResponse:
    if not STATE["systems"]:
        build_state()
    name = system or next(iter(STATE["systems"]))
    if name not in STATE["systems"]:
        raise HTTPException(status_code=400, detail=f"Unknown system '{name}'. Available: {sorted(STATE['systems'])}")

    retriever = STATE["systems"][name]
    hits, took_ms = timed_search(retriever, q, top_k=max(top_k * 3, top_k))
    hits = dedupe_by_document(hits, top_k)
    return SearchResponse(
        query=q,
        system=name,
        took_ms=round(took_ms, 2),
        total=len(hits),
        hits=[
            SearchHit(
                rank=hit.rank,
                doc_id=hit.doc_id,
                chunk_id=hit.chunk_id,
                score=round(hit.score, 6),
                snippet=make_snippet(hit.text, q, markup=highlight),
            )
            for hit in hits
        ],
    )