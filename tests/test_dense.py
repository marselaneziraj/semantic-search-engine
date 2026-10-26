import numpy as np
import pytest

from semsearch.embeddings import CachedEmbedder, HashingEmbedder, l2_normalize
from semsearch.retrieval.dense import DenseRetriever


def test_hashing_embeddings_are_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=64)
    first = embedder.encode(["reciprocal rank fusion", "dense retrieval"])
    second = embedder.encode(["reciprocal rank fusion", "dense retrieval"])
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)


def test_l2_normalize_handles_zero_rows():
    matrix = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    normalised = l2_normalize(matrix)
    assert np.allclose(normalised[0], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(normalised[1]), 1.0)


def test_embedding_cache_is_used_on_the_second_call(tmp_path):
    embedder = CachedEmbedder(HashingEmbedder(dim=32), cache_dir=tmp_path)
    texts = ["alpha beta", "gamma delta"]
    first = embedder.encode(texts)
    assert embedder.last_cache_hit is False
    second = embedder.encode(texts)
    assert embedder.last_cache_hit is True
    assert np.allclose(first, second)
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_cache_key_changes_with_the_text(tmp_path):
    embedder = CachedEmbedder(HashingEmbedder(dim=32), cache_dir=tmp_path)
    embedder.encode(["alpha"])
    embedder.encode(["beta"])
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_dense_retrieval_returns_ranked_hits(toy_chunks, tmp_path):
    retriever = DenseRetriever(backend="hashing", hashing_dim=256, cache_dir=tmp_path)
    retriever.index(toy_chunks)
    hits = retriever.search("reciprocal rank fusion ranked lists", top_k=5)
    assert hits
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_faiss_flat_matches_numpy_brute_force(toy_chunks, tmp_path):
    """IndexFlatIP is exact, so it must agree with a plain matrix multiply."""
    retriever = DenseRetriever(backend="hashing", hashing_dim=256, cache_dir=tmp_path)
    retriever.index(toy_chunks)
    faiss_hits = retriever.search("cross encoder reranking", top_k=5)

    retriever._index = None  # force the numpy fallback path
    numpy_hits = retriever.search("cross encoder reranking", top_k=5)

    # The scores must agree to float32 precision. The *order* of a tie may differ
    # between FAISS and argsort, so compare the score vectors, not the ids.
    assert [h.doc_id for h in faiss_hits][:3] == [h.doc_id for h in numpy_hits][:3]
    for a, b in zip(faiss_hits, numpy_hits):
        assert a.score == pytest.approx(b.score, abs=1e-5)


def test_save_and_load_roundtrip(toy_chunks, tmp_path):
    retriever = DenseRetriever(backend="hashing", hashing_dim=128, cache_dir=tmp_path)
    retriever.index(toy_chunks)
    before = [hit.doc_id for hit in retriever.search("nDCG discount", top_k=5)]
    path = tmp_path / "index.npz"
    retriever.save(path)

    restored = DenseRetriever(backend="hashing", hashing_dim=128, cache_dir=tmp_path)
    restored.load(path)
    after = [hit.doc_id for hit in restored.search("nDCG discount", top_k=5)]
    assert before == after
    assert len(restored) == len(toy_chunks)


def test_ivfpq_falls_back_to_flat_on_a_tiny_corpus(toy_chunks, tmp_path):
    """12 vectors cannot train a product quantiser; the index must degrade, not crash."""
    retriever = DenseRetriever(backend="hashing", hashing_dim=256, cache_dir=tmp_path,
                               index_type="ivfpq")
    with pytest.warns(RuntimeWarning, match="falling back"):
        retriever.index(toy_chunks)
    assert retriever.search("reciprocal rank fusion", top_k=3)
