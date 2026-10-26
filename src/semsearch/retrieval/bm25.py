"""Lexical retrieval with Okapi BM25 (rank_bm25).

BM25 scores a document d for a query q as a sum over query terms:

    score(q, d) = SUM_t IDF(t) * ( f(t,d) * (k1 + 1) ) /
                  ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

Two ideas do all the work:

* Term-frequency saturation (k1): the 10th occurrence of a word adds far less
  than the 2nd. Without it, a document that repeats a word 500 times would win
  every time.
* Length normalisation (b): long documents contain more words by accident, so
  their term frequencies are discounted by |d| / avgdl.

IDF makes rare terms count more than common ones. The result is a bag-of-words
model: it cannot match "heart attack" to "myocardial infarction", but it is
exact, fast, needs no training, and is still the retriever to beat on keyword
and identifier queries.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from semsearch.data.chunking import Chunk
from semsearch.retrieval.base import Hit

TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short, explicit stop list. Kept small on purpose: BM25's IDF already
# down-weights frequent words, so aggressive stop-wording mostly costs recall.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "s", "such", "t", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "were",
    "will", "with", "what", "how", "why", "when", "does", "do", "you", "your",
}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, strip punctuation, optionally drop stop words."""
    tokens = TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Retriever:
    """BM25 over chunk texts."""

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, remove_stopwords: bool = True) -> None:
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        corpus = [tokenize(c.text, self.remove_stopwords) for c in self._chunks]
        # rank_bm25 cannot handle a document with zero tokens; guard with a sentinel.
        corpus = [tokens or ["\x00empty"] for tokens in corpus]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        if self._bm25 is None:
            raise RuntimeError("index() must be called before search()")
        tokens = tokenize(query, self.remove_stopwords)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            # argpartition is O(n); a full sort of 100k+ scores per query is wasteful.
            part = np.argpartition(-scores, top_k)[:top_k]
            order = part[np.argsort(-scores[part])]
        hits: list[Hit] = []
        for rank, idx in enumerate(order, start=1):
            if scores[idx] <= 0:
                continue
            chunk = self._chunks[int(idx)]
            hits.append(Hit(chunk.chunk_id, chunk.doc_id, float(scores[idx]), chunk.text, rank))
        return hits

    def __len__(self) -> int:
        return len(self._chunks)
