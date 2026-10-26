"""Config precedence tests."""

import pytest

from semsearch.config import DEFAULT_CONFIG, Settings


def test_yaml_values_are_loaded():
    settings = Settings.from_yaml(DEFAULT_CONFIG)
    assert settings.top_k == 10
    assert settings.hybrid.rrf_k == 60
    assert settings.chunking.size == 256


def test_environment_overrides_the_yaml_file(monkeypatch):
    monkeypatch.setenv("SEMSEARCH_TOP_K", "25")
    monkeypatch.setenv("SEMSEARCH_DATASET", "toy")
    settings = Settings.from_yaml(DEFAULT_CONFIG)
    assert settings.top_k == 25
    assert settings.dataset == "toy"


def test_nested_environment_override_keeps_sibling_keys(monkeypatch):
    monkeypatch.setenv("SEMSEARCH_DENSE__BACKEND", "hashing")
    settings = Settings.from_yaml(DEFAULT_CONFIG)
    assert settings.dense.backend == "hashing"
    assert settings.dense.batch_size == 64  # still from the YAML file


def test_explicit_overrides_win_over_everything(monkeypatch):
    monkeypatch.setenv("SEMSEARCH_TOP_K", "25")
    settings = Settings.from_yaml(DEFAULT_CONFIG, top_k=7)
    assert settings.top_k == 7


def test_missing_config_file_falls_back_to_defaults(tmp_path):
    settings = Settings.from_yaml(tmp_path / "nope.yaml")
    assert settings.top_k == 10
    assert settings.dataset == "beir/scifact"


def test_describe_mentions_the_key_knobs():
    text = Settings.from_yaml(DEFAULT_CONFIG).describe()
    assert "dataset=" in text and "rrf_k=" in text
