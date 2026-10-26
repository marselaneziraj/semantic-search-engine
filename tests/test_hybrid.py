import pytest

from semsearch.retrieval.base import Hit, dedupe_by_document
from semsearch.retrieval.bm25 import BM25Retriever
from semsearch.retrieval.dense import DenseRetriever
from semsearch.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from semsearch.retrieval.rerank import CrossEncoderReranker, RerankingRetriever


def _hits(doc_ids):
    return [Hit(d, d, 1.0 / (i + 1), f"text of {d}", i + 1) for i, d in enumerate(doc_ids)]


def test_rrf_score_is_hand_computable():
    fused = reciprocal_rank_fusion([_hits(["a", "b"]), _hits(["b", "a"])], k=60, top_k=2)
    # a: 1/61 + 1/62, b: 1/62 + 1/61 -> identical, tie broken by chunk_id
    assert {hit.doc_id for hit in fused} == {"a", "b"}
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_prefers_the_document_both_systems_ranked():
    # 'b' is 2nd in both lists; 'a' is 1st in one and absent from the other.
    fused = reciprocal_rank_fusion([_hits(["a", "b"]), _hits(["c", "b"])], k=60, top_k=3)
    assert fused[0].doc_id == "b"
    assert fused[0].score == pytest.approx(2 / 62)


def test_rrf_is_scale_free():
    """Multiplying one system's scores by 1000 must not change the fusion."""
    lexical = _hits(["a", "b", "c"])
    inflated = [Hit(h.chunk_id, h.doc_id, h.score * 1000, h.text, h.rank) for h in lexical]
    dense = _hits(["c", "b", "a"])
    assert [h.doc_id for h in reciprocal_rank_fusion([lexical, dense], top_k=3)] == \
           [h.doc_id for h in reciprocal_rank_fusion([inflated, dense], top_k=3)]


def test_rrf_depth_truncates_the_input_lists():
    fused = reciprocal_rank_fusion([_hits(["a", "b", "c"])], k=60, depth=2, top_k=5)
    assert [hit.doc_id for hit in fused] == ["a", "b"]


def test_rrf_weights_shift_the_balance():
    runs = [_hits(["a", "x"]), _hits(["b", "y"])]
    assert reciprocal_rank_fusion(runs, weights=[5.0, 1.0], top_k=1)[0].doc_id == "a"
    assert reciprocal_rank_fusion(runs, weights=[1.0, 5.0], top_k=1)[0].doc_id == "b"


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([_hits(["a"])], weights=[1.0, 1.0])


def test_hybrid_returns_the_union_of_both_parents(toy_dataset, toy_chunks, tmp_path):
    lexical = BM25Retriever()
    lexical.index(toy_chunks)
    dense = DenseRetriever(backend="hashing", hashing_dim=256, cache_dir=tmp_path)
    dense.index(toy_chunks)
    hybrid = HybridRetriever(lexical, dense, rrf_k=60, depth=50)

    query = "how do you combine two ranked lists"
    # The parents must be read at the same depth the fusion reads them (50),
    # otherwise the fused list legitimately contains documents from deeper down.
    lexical_ids = {hit.doc_id for hit in lexical.search(query, top_k=50)}
    dense_ids = {hit.doc_id for hit in dense.search(query, top_k=50)}
    hybrid_ids = {hit.doc_id for hit in hybrid.search(query, top_k=20)}
    assert hybrid_ids <= (lexical_ids | dense_ids)
    assert hybrid_ids & lexical_ids and hybrid_ids & dense_ids


def test_identity_reranker_preserves_order(toy_chunks, tmp_path):
    lexical = BM25Retriever()
    lexical.index(toy_chunks)
    dense = DenseRetriever(backend="hashing", hashing_dim=256, cache_dir=tmp_path)
    dense.index(toy_chunks)
    hybrid = HybridRetriever(lexical, dense, depth=50)
    reranked = RerankingRetriever(hybrid, CrossEncoderReranker(backend="identity"), candidates=20)

    query = "what does nDCG measure"
    assert [h.doc_id for h in reranked.search(query, top_k=5)] == \
           [h.doc_id for h in hybrid.search(query, top_k=5)]


def test_dedupe_keeps_the_best_chunk_per_document():
    hits = [
        Hit("d1#0", "d1", 9.0, "a", 1),
        Hit("d1#1", "d1", 8.0, "b", 2),
        Hit("d2#0", "d2", 7.0, "c", 3),
    ]
    deduped = dedupe_by_document(hits, top_k=10)
    assert [hit.chunk_id for hit in deduped] == ["d1#0", "d2#0"]
    assert [hit.rank for hit in deduped] == [1, 2]
