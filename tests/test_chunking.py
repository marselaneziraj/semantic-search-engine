import pytest

from semsearch.data.chunking import chunk_document, chunk_corpus, iter_batches
from semsearch.data.loaders import Document


def test_short_document_is_one_chunk_with_parent_id():
    chunks = chunk_document("d1", "one two three four five", size=256)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "d1"
    assert chunks[0].is_whole_document


def test_long_document_is_split_with_overlap():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_document("d1", text, size=100, overlap=20)
    # stride = size - overlap = 80, so windows start at 0, 80 and 160; the third
    # window already reaches token 250, so no fourth window is emitted.
    assert len(chunks) == 3
    assert [c.chunk_id for c in chunks] == ["d1#0", "d1#1", "d1#2"]
    assert all(c.doc_id == "d1" for c in chunks)

    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()
    assert first_tokens[-20:] == second_tokens[:20]  # the overlap is real


def test_no_token_is_lost():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_document("d1", text, size=64, overlap=8)
    covered = set()
    for chunk in chunks:
        covered.update(chunk.text.split())
    assert covered == set(text.split())


def test_empty_and_whitespace_documents_are_dropped():
    assert chunk_document("d1", "") == []
    assert chunk_document("d1", "   \n  ") == []


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_document("d1", "a b c", size=10, overlap=10)


def test_chunking_disabled_keeps_one_chunk_per_document():
    documents = [Document("d1", " ".join(["word"] * 1000)), Document("d2", " ".join(["x"] * 500))]
    chunks = chunk_corpus(documents, size=100, enabled=False)
    assert len(chunks) == 2
    assert {c.chunk_id for c in chunks} == {"d1", "d2"}


def test_iter_batches_covers_everything():
    items = list(range(10))
    batches = list(iter_batches(items, 3))
    assert [len(b) for b in batches] == [3, 3, 3, 1]
    assert [x for b in batches for x in b] == items
