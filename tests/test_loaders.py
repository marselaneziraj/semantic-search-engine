from semsearch.data.loaders import load_dataset


def test_toy_dataset_loads_with_judgements(toy_dataset):
    assert len(toy_dataset.documents) == 12
    assert len(toy_dataset.queries) == 5
    assert toy_dataset.has_judgements
    assert toy_dataset.qrels["q1"]["d1"] == 2
    assert "documents" in toy_dataset.summary()


def test_max_docs_truncates():
    dataset = load_dataset("toy", max_docs=3)
    assert len(dataset.documents) == 3


def test_document_display_text_joins_title_and_body(toy_dataset):
    document = toy_dataset.documents[0]
    assert document.title
    assert document.display_text.startswith(document.title)


# --- ir_datasets split resolution ------------------------------------------
# These check the registry only (no corpus download), so they run in CI.

import pytest

ir_datasets = pytest.importorskip("ir_datasets", reason="ir_datasets is an optional extra")

from semsearch.data.loaders import _resolve_ir_dataset  # noqa: E402


def test_parent_beir_id_resolves_to_the_judged_split():
    """`beir/scifact` has no qrels of its own - they live on `beir/scifact/test`."""
    assert ir_datasets.load("beir/scifact").has_qrels() is False
    resolved, handle = _resolve_ir_dataset("beir/scifact", split="test")
    assert resolved == "beir/scifact/test"
    assert handle.has_qrels()


def test_explicit_split_is_kept_as_given():
    resolved, handle = _resolve_ir_dataset("beir/scifact/train", split="test")
    assert resolved == "beir/scifact/train"
    assert handle.has_qrels()


def test_requested_split_is_honoured():
    resolved, _ = _resolve_ir_dataset("beir/scifact", split="train")
    assert resolved == "beir/scifact/train"


def test_dataset_whose_parent_already_has_qrels_is_untouched():
    """beir/trec-covid carries its judgements on the parent id and has no splits."""
    resolved, handle = _resolve_ir_dataset("beir/trec-covid", split="test")
    assert resolved == "beir/trec-covid"
    assert handle.has_qrels()


def test_unknown_split_does_not_crash():
    resolved, _ = _resolve_ir_dataset("beir/nfcorpus", split="nonexistent")
    assert resolved == "beir/nfcorpus/test"