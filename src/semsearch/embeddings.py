"""Embedding backends with an on-disk cache.

Embedding 100k passages with all-MiniLM-L6-v2 on a CPU takes tens of minutes.
Doing it again for every ablation would be absurd, so every call is cached:

    .cache/embeddings/<sha1 of (backend, model, dim, normalise, texts)>.npy

The key is a hash of the exact text list, so changing the chunk size changes the
key and the cache cannot go stale behind your back.

Two backends:

* "sentence-transformers" - the real bi-encoder (downloads ~90 MB once).
* "hashing" - a deterministic hashed bag-of-words projection. No model, no
  download, no network. It is a weak retriever, but it exercises the entire
  dense/FAISS/fusion/rerank code path, which is what lets the tests and CI run
  offline in seconds.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _cache_key(backend: str, model_name: str, dim: int, normalize: bool, texts: Sequence[str]) -> str:
    digest = hashlib.sha1()
    digest.update(f"{backend}|{model_name}|{dim}|{normalize}|{len(texts)}".encode("utf-8"))
    for text in texts:
        digest.update(text.encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length so that dot product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic hashed bag-of-words vectors (offline fallback / CI backend)."""

    backend = "hashing"

    def __init__(self, dim: int = 256, normalize: bool = True) -> None:
        self.dim = dim
        self.normalize = normalize
        self.model_name = f"hashing-{dim}"

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                bucket = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little") % self.dim
                matrix[row, bucket] += 1.0
        matrix = np.log1p(matrix)
        return l2_normalize(matrix) if self.normalize else matrix


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers, CPU-only by default."""

    backend = "sentence-transformers"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "sentence-transformers is not installed. Either install it:\n"
                "    pip install sentence-transformers\n"
                "or run with the offline backend: --dense-backend hashing"
            ) from exc
        self.model_name = model_name
        self.normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        matrix = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )
        return np.asarray(matrix, dtype=np.float32)


def build_embedder(backend: str = "sentence-transformers", model_name: str = "", dim: int = 256,
                   normalize: bool = True):
    """Factory used by the CLI and the retrievers."""
    if backend == "hashing":
        return HashingEmbedder(dim=dim, normalize=normalize)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            normalize=normalize,
        )
    raise ValueError(f"Unknown embedding backend: {backend}")


class CachedEmbedder:
    """Wraps an embedder and memoises whole-list encodings on disk as .npy."""

    def __init__(self, embedder, cache_dir: str | Path = ".cache/embeddings") -> None:
        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_cache_hit: bool | None = None

    @property
    def model_name(self) -> str:
        return getattr(self.embedder, "model_name", "unknown")

    @property
    def dim(self) -> int:
        return int(getattr(self.embedder, "dim", 0))

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False,
               use_cache: bool = True) -> np.ndarray:
        if not use_cache:
            self.last_cache_hit = False
            return self.embedder.encode(texts, batch_size=batch_size, show_progress=show_progress)
        key = _cache_key(
            getattr(self.embedder, "backend", "unknown"),
            self.model_name,
            int(getattr(self.embedder, "dim", 0)),
            bool(getattr(self.embedder, "normalize", True)),
            texts,
        )
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            self.last_cache_hit = True
            return np.load(path)
        matrix = self.embedder.encode(texts, batch_size=batch_size, show_progress=show_progress)
        np.save(path, matrix)
        self.last_cache_hit = False
        return matrix
