from semsearch.data.chunking import chunk_corpus
from semsearch.retrieval.bm25 import BM25Retriever, tokenize


def test_tokenizer_lowercases_and_strips_punctuation():
    assert tokenize("BM25, the Okapi function!", remove_stopwords=False) == ["bm25", "the", "okapi", "function"]
    assert "the" not in tokenize("BM25, the Okapi function!")


def test_three_sample_queries_return_the_expected_documents(toy_dataset):
    retriever = BM25Retriever()
    retriever.index(chunk_corpus(toy_dataset.documents))

    assert retriever.search("reciprocal rank fusion", top_k=3)[0].doc_id == "d1"
    assert retriever.search("product quantisation latency memory", top_k=3)[0].doc_id == "d7"
    assert retriever.search("inverted index terms documents", top_k=3)[0].doc_id == "d6"


def test_scores_are_descending_and_positive(toy_dataset):
    retriever = BM25Retriever()
    retriever.index(chunk_corpus(toy_dataset.documents))
    hits = retriever.search("discounting gains logarithmically", top_k=5)
    assert hits
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert all(hit.score > 0 for hit in hits)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_query_with_no_matching_terms_returns_nothing(toy_dataset):
    retriever = BM25Retriever()
    retriever.index(chunk_corpus(toy_dataset.documents))
    assert retriever.search("zzzzz qqqqq", top_k=5) == []


def test_vocabulary_mismatch_is_bm25s_weak_spot(toy_dataset):
    """The honest baseline comparison: no shared tokens, no lexical match."""
    retriever = BM25Retriever()
    retriever.index(chunk_corpus(toy_dataset.documents))
    hits = retriever.search("myocardial infarction", top_k=5)
    assert hits == []
