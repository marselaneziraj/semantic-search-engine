"""Configuration: YAML experiment file + environment overrides via pydantic-settings.

Precedence (highest first):
    1. explicit keyword arguments in code (which is how the CLI applies its flags)
    2. environment variables prefixed with SEMSEARCH_ (e.g. SEMSEARCH_TOP_K=50,
       or SEMSEARCH_DENSE__BACKEND=hashing for a nested field)
    3. values in the YAML experiment file
    4. the defaults declared below
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def _drop_env_overridden_keys(data: dict[str, Any], prefix: str = "SEMSEARCH_") -> dict[str, Any]:
    """Remove YAML keys that are also set in the environment (env takes priority).

    Nested keys use the double-underscore convention, e.g.
    SEMSEARCH_DENSE__BACKEND=hashing overrides only `dense.backend` and leaves the
    rest of the `dense` block from the file untouched.
    """
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested = {
                sub_key: sub_value
                for sub_key, sub_value in value.items()
                if f"{prefix}{key.upper()}__{sub_key.upper()}" not in os.environ
            }
            cleaned[key] = nested
        elif f"{prefix}{key.upper()}" not in os.environ:
            cleaned[key] = value
    return cleaned


class ChunkingConfig(BaseSettings):
    """How a long document is split into retrievable passages."""

    enabled: bool = True
    size: int = Field(default=256, description="Target chunk length in whitespace tokens.")
    overlap: int = Field(default=32, description="Token overlap between consecutive chunks.")
    min_chars: int = Field(default=20, description="Drop chunks shorter than this.")


class DenseConfig(BaseSettings):
    """Dense bi-encoder settings."""

    model_config = SettingsConfigDict(protected_namespaces=())

    backend: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True
    hashing_dim: int = Field(default=256, description="Only used by the offline 'hashing' backend.")
    cache_dir: Path = PROJECT_ROOT / ".cache" / "embeddings"


class RerankConfig(BaseSettings):
    """Cross-encoder reranker settings."""

    model_config = SettingsConfigDict(protected_namespaces=())

    backend: Literal["cross-encoder", "identity"] = "cross-encoder"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    candidates: int = Field(default=50, description="How many fused hits the cross-encoder rescores.")


class HybridConfig(BaseSettings):
    """Reciprocal rank fusion settings."""

    rrf_k: int = Field(default=60, description="RRF smoothing constant; 60 is the TREC default.")
    depth: int = Field(default=200, description="How deep each ranked list is read before fusion.")


class Settings(BaseSettings):
    """Top-level experiment configuration."""

    model_config = SettingsConfigDict(env_prefix="SEMSEARCH_", env_nested_delimiter="__", extra="ignore")

    dataset: str = Field(default="beir/scifact", description="ir_datasets id, or jsonl:<path>.")
    split_queries: str = Field(default="test", description="Query split for ir_datasets (unused for jsonl).")
    max_docs: int | None = Field(default=None, description="Truncate the corpus (smoke tests).")
    max_queries: int | None = Field(default=None, description="Truncate the query set (smoke tests).")

    top_k: int = Field(default=10, description="Hits returned to the user / scored at cutoff.")
    eval_depth: int = Field(default=100, description="Ranked-list depth written to the run file.")

    chunking: ChunkingConfig = ChunkingConfig()
    dense: DenseConfig = DenseConfig()
    rerank: RerankConfig = RerankConfig()
    hybrid: HybridConfig = HybridConfig()

    results_dir: Path = PROJECT_ROOT / "results"
    cache_dir: Path = PROJECT_ROOT / ".cache"
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, **overrides: Any) -> "Settings":
        """Load settings from a YAML file, letting environment variables win over it.

        pydantic-settings gives init keyword arguments the highest priority, so
        simply passing the parsed YAML as kwargs would make the file beat the
        environment - the opposite of the documented order. Any YAML key that has
        a matching SEMSEARCH_ variable is therefore dropped before construction,
        which hands that key back to pydantic-settings' env loader.
        """
        path = Path(path) if path else DEFAULT_CONFIG
        data: dict[str, Any] = {}
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data = _drop_env_overridden_keys(data)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def describe(self) -> str:
        return (
            f"dataset={self.dataset} top_k={self.top_k} eval_depth={self.eval_depth} "
            f"chunk={self.chunking.size}/{self.chunking.overlap} "
            f"dense={self.dense.backend}:{self.dense.model_name} "
            f"rerank={self.rerank.backend} rrf_k={self.hybrid.rrf_k}"
        )
