"""Cross-encoder reranking.

A bi-encoder has to commit to one vector per passage before it has ever seen the
query, so query and passage never interact inside the network. A cross-encoder
concatenates them - [CLS] query [SEP] passage [SEP] - and every attention layer
sees both at once, which is what lets it judge "does this passage answer *this*
query" rather than "is this passage about roughly the same topic".

The price is arithmetic: one transformer forward pass per (query, passage) pair,
with nothing precomputable. Scoring 5M passages per query is impossible; scoring
the top 50 from a cheap first stage costs about 50 forward passes, i.e. a few
hundred milliseconds on a CPU. That is the whole retrieve-then-rerank pattern:
a cheap recall-oriented stage followed by an expensive precision-oriented one.

Tuning knob: `candidates`. Too few and the reranker cannot fix the first stage's
recall; too many and latency grows linearly with no accuracy left to gain.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from semsearch.data.chunking import Chunk
from semsearch.retrieval.base import Hit


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder; 'identity' keeps the input order."""

    def __init__(
        self,
        backend: str = "cross-encoder",
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        if backend == "cross-encoder":
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - optional extra
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking.\n"
                    "    pip install sentence-transformers\n"
                    "or run with --rerank-backend identity"
                ) from exc
            self._model = CrossEncoder(model_name, device="cpu")
        elif backend != "identity":
            raise ValueError(f"Unknown rerank backend: {backend}")

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        if self._model is None:  # identity backend: preserve incoming order
            return np.arange(len(passages), 0, -1, dtype=np.float32)
        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int = 10) -> list[Hit]:
        if not hits:
            return []
        scores = self.score(query, [hit.text for hit in hits])
        order = np.argsort(-scores)
        out: list[Hit] = []
        for rank, idx in enumerate(order[:top_k], start=1):
            source = hits[int(idx)]
            out.append(Hit(source.chunk_id, source.doc_id, float(scores[int(idx)]), source.text, rank))
        return out


class RerankingRetriever:
    """First stage (usually hybrid) + cross-encoder second stage."""

    name = "hybrid+rerank"

    def __init__(self, first_stage, reranker: CrossEncoderReranker, candidates: int = 50) -> None:
        self.first_stage = first_stage
        self.reranker = reranker
        self.candidates = candidates

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.first_stage.index(chunks)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        pool = self.first_stage.search(query, top_k=max(self.candidates, top_k))
        return self.reranker.rerank(query, pool, top_k=top_k)
