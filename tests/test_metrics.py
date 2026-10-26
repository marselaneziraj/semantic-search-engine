"""Metric tests: hand-computed values first, then agreement with ir_measures."""

import math

import pytest

from semsearch.evaluation.metrics import (
    average_precision,
    evaluate_run,
    evaluate_run_ir_measures,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_ndcg_perfect_ranking_is_one():
    relevance = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], relevance, k=10) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # Ranking: b (rel 1) then a (rel 2).
    # DCG  = 1/log2(2) + 2/log2(3) = 1.0 + 1.26186 = 2.26186
    # IDCG = 2/log2(2) + 1/log2(3) = 2.0 + 0.63093 = 2.63093
    relevance = {"a": 2, "b": 1}
    expected = (1 / math.log2(2) + 2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], relevance, k=10) == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(0.8597, abs=1e-4)


def test_ndcg_rewards_putting_the_graded_document_first():
    relevance = {"a": 3, "b": 1}
    assert ndcg_at_k(["a", "b"], relevance, 10) > ndcg_at_k(["b", "a"], relevance, 10)


def test_ndcg_cutoff_is_respected():
    relevance = {"a": 1}
    assert ndcg_at_k(["x", "y", "a"], relevance, k=2) == 0.0
    assert ndcg_at_k(["x", "y", "a"], relevance, k=3) == pytest.approx(1 / math.log2(4))


def test_recall_counts_graded_documents_as_relevant():
    relevance = {"a": 2, "b": 1, "c": 0}
    assert recall_at_k(["a", "z"], relevance, k=2) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], relevance, k=2) == pytest.approx(1.0)
    assert recall_at_k(["c"], relevance, k=2) == 0.0  # rel 0 is not relevant


def test_recall_ignores_order():
    relevance = {"a": 1, "b": 1}
    assert recall_at_k(["a", "b"], relevance, 10) == recall_at_k(["b", "a"], relevance, 10)


def test_reciprocal_rank_positions():
    relevance = {"a": 1}
    assert reciprocal_rank(["a", "x"], relevance, 10) == pytest.approx(1.0)
    assert reciprocal_rank(["x", "a"], relevance, 10) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y", "a"], relevance, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], relevance, 10) == 0.0


def test_average_precision_hand_computed():
    relevance = {"a": 1, "b": 1}
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / 2 = 0.8333
    assert average_precision(["a", "x", "b"], relevance, 100) == pytest.approx((1 + 2 / 3) / 2)


def test_evaluate_run_macro_averages_over_queries():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": ["a"], "q2": ["x", "b"]}
    scores = evaluate_run(run, qrels, ndcg_k=10, recall_k=10, mrr_k=10)
    assert scores["MRR@10"] == pytest.approx((1.0 + 0.5) / 2)
    assert scores["Recall@10"] == pytest.approx(1.0)


def test_missing_query_in_run_scores_zero_not_crash():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    scores = evaluate_run({"q1": ["a"]}, qrels, ndcg_k=10, recall_k=10, mrr_k=10)
    assert scores["MRR@10"] == pytest.approx(0.5)


def test_agreement_with_ir_measures():
    """The internal implementation must match pytrec_eval on the same run."""
    qrels = {
        "q1": {"a": 2, "b": 1, "c": 0},
        "q2": {"d": 1, "e": 2},
        "q3": {"f": 1},
    }
    run = {
        "q1": ["c", "a", "z", "b"],
        "q2": ["e", "z", "d"],
        "q3": ["x", "y", "z"],
    }
    strict = evaluate_run_ir_measures(run, qrels, ndcg_k=10, recall_k=10, mrr_k=10)
    if strict is None:
        pytest.skip("ir_measures not installed")
    ours = evaluate_run(run, qrels, ndcg_k=10, recall_k=10, mrr_k=10)
    for metric, value in strict.items():
        assert ours[metric] == pytest.approx(value, abs=1e-6), metric
