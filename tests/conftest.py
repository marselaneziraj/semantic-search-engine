import pytest

from semsearch.config import Settings
from semsearch.data.chunking import chunk_corpus
from semsearch.data.loaders import load_dataset


@pytest.fixture(scope="session")
def toy_dataset():
    return load_dataset("toy")


@pytest.fixture(scope="session")
def offline_settings(tmp_path_factory):
    """Settings that never touch the network: hashed embeddings, identity rerank."""
    settings = Settings.from_yaml()
    settings.dataset = "toy"
    settings.dense.backend = "hashing"
    settings.dense.hashing_dim = 256
    settings.rerank.backend = "identity"
    settings.results_dir = tmp_path_factory.mktemp("results")
    settings.dense.cache_dir = tmp_path_factory.mktemp("embeddings")
    return settings


@pytest.fixture(scope="session")
def toy_chunks(toy_dataset, offline_settings):
    return chunk_corpus(toy_dataset.documents, size=offline_settings.chunking.size,
                        overlap=offline_settings.chunking.overlap)
