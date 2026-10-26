"""IR metrics, implemented transparently and cross-checked against ir_measures.

The implementations here are deliberately readable - this is the part of the
project you must be able to defend in an interview. `tests/test_metrics.py`
asserts them against hand-computed values and, when ir_measures is installed,
against pytrec_eval's numbers on a shared run.

--------------------------------------------------------------------------
nDCG@k - Normalised Discounted Cumulative Gain
--------------------------------------------------------------------------
    DCG@k  = SUM over ranks i<=k   rel_i / log2(i + 1)
    IDCG@k = the same sum for the best possible ordering of the judgements
    nDCG@k = DCG@k / IDCG@k            -> 1.0 means "perfectly ordered"

Three things it does that Precision@k does not:
  1. It is graded. A "highly relevant" document (rel=2) counts more than a
     "somewhat relevant" one (rel=1).
  2. It is rank-weighted. Moving the right answer from rank 5 to rank 1 raises
     the score; Precision@10 would not move at all.
  3. It is normalised, so a query with 2 relevant documents and one with 50 are
     on the same 0-1 scale and can be averaged.
The discount is logarithmic rather than linear because user attention falls off
steeply but never quite to zero.

--------------------------------------------------------------------------
Recall@k
--------------------------------------------------------------------------
    |relevant documents retrieved in the top k| / |relevant documents total|
It ignores order completely. Recall@100 is the ceiling for a reranker: a
cross-encoder can only reorder what the first stage actually handed it.

--------------------------------------------------------------------------
MRR@k - Mean Reciprocal Rank
--------------------------------------------------------------------------
    1 / rank of the first relevant result   (0 if none in the top k)
The right metric when there is exactly one correct answer and the user stops
reading as soon as they find it. Brutal: rank 1 = 1.00, rank 2 = 0.50.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Mapping, Sequence

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of an ordered gain vector (rank 1 first)."""
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int = 10) -> float:
    """nDCG@k for one query."""
    if not relevance:
        return 0.0
    gains = [float(relevance.get(doc_id, 0)) for doc_id in ranked_ids[:k]]
    ideal = sorted((float(v) for v in relevance.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int = 100) -> float:
    """Recall@k for one query, counting rel > 0 as relevant."""
    total_relevant = sum(1 for v in relevance.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    found = sum(1 for doc_id in ranked_ids[:k] if relevance.get(doc_id, 0) > 0)
    return found / total_relevant


def reciprocal_rank(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int = 10) -> float:
    """Reciprocal rank of the first relevant document inside the top k."""
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def average_precision(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int = 100) -> float:
    """Average precision at k - included because MAP is still the TREC default."""
    total_relevant = sum(1 for v in relevance.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if relevance.get(doc_id, 0) > 0:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(total_relevant, k)


def evaluate_run(
    run: Run,
    qrels: Qrels,
    ndcg_k: int = 10,
    recall_k: int = 100,
    mrr_k: int = 10,
) -> dict[str, float]:
    """Macro-average the metrics over every judged query.

    Only queries present in qrels are scored (an unjudged query has no ground
    truth, and scoring it as 0 would punish systems for the assessors' budget).
    A judged query that the system returned nothing for *is* scored, as 0.
    """
    judged = [qid for qid in qrels if qrels[qid]]
    if not judged:
        return {f"nDCG@{ndcg_k}": 0.0, f"Recall@{recall_k}": 0.0, f"MRR@{mrr_k}": 0.0, "MAP@100": 0.0}

    ndcg_scores, recall_scores, rr_scores, ap_scores = [], [], [], []
    for qid in judged:
        ranked = list(run.get(qid, []))
        relevance = qrels[qid]
        ndcg_scores.append(ndcg_at_k(ranked, relevance, ndcg_k))
        recall_scores.append(recall_at_k(ranked, relevance, recall_k))
        rr_scores.append(reciprocal_rank(ranked, relevance, mrr_k))
        ap_scores.append(average_precision(ranked, relevance, 100))

    return {
        f"nDCG@{ndcg_k}": mean(ndcg_scores),
        f"Recall@{recall_k}": mean(recall_scores),
        f"MRR@{mrr_k}": mean(rr_scores),
        "MAP@100": mean(ap_scores),
    }


def evaluate_run_ir_measures(
    run: Run,
    qrels: Qrels,
    ndcg_k: int = 10,
    recall_k: int = 100,
    mrr_k: int = 10,
) -> dict[str, float] | None:
    """Same metrics via ir_measures/pytrec_eval. Returns None if not installed.

    Scores are passed as descending integers derived from the rank, which is all
    pytrec_eval needs - it re-sorts by score and ties are broken by doc id.
    """
    try:
        import ir_measures
        from ir_measures import MRR, RR, nDCG, R
    except ImportError:
        return None

    measures = [nDCG @ ndcg_k, R @ recall_k, RR @ mrr_k]
    run_tuples = [
        ir_measures.ScoredDoc(qid, doc_id, float(len(ranked) - rank))
        for qid, ranked in run.items()
        for rank, doc_id in enumerate(ranked)
    ]
    qrel_tuples = [
        ir_measures.Qrel(qid, doc_id, int(rel))
        for qid, rels in qrels.items()
        for doc_id, rel in rels.items()
    ]
    scores = ir_measures.calc_aggregate(measures, qrel_tuples, run_tuples)
    return {
        f"nDCG@{ndcg_k}": float(scores[nDCG @ ndcg_k]),
        f"Recall@{recall_k}": float(scores[R @ recall_k]),
        f"MRR@{mrr_k}": float(scores[RR @ mrr_k]),
    }
