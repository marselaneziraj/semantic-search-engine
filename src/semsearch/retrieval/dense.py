"""Dense retrieval: bi-encoder embeddings + a FAISS index.

A bi-encoder encodes the query and every passage *independently* into one vector
each, so all passage vectors can be computed offline and searched with a single
matrix operation. That independence is exactly why it is fast - and exactly why
it is less accurate than a cross-encoder, which gets to look at the query and
the passage together.

Index types:
    flat    - IndexFlatIP, exact brute-force inner product. Ground truth.
    ivfpq   - IndexIVFPQ, coarse clustering + product quantisation. Much smaller
              and faster, slightly lower recall. Use it above ~1M vectors.

With normalised embeddings, inner product == cosine similarity.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from semsearch.data.chunking import Chunk
from semsearch.embeddings import CachedEmbedder, build_embedder, l2_normalize
from semsearch.retrieval.base import Hit


class DenseRetriever:
    """Bi-encoder retrieval over chunk embeddings."""

    name = "dense"

    def __init__(
        self,
        backend: str = "sentence-transformers",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
        hashing_dim: int = 256,
        cache_dir: str | Path = ".cache/embeddings",
        index_type: str = "flat",
        nlist: int = 100,
        m: int = 16,
        nbits: int = 8,
        nprobe: int = 10,
    ) -> None:
        self.embedder = CachedEmbedder(
            build_embedder(backend, model_name, dim=hashing_dim, normalize=normalize),
            cache_dir=cache_dir,
        )
        self.batch_size = batch_size
        self.normalize = normalize
        self.index_type = index_type
        self.nlist, self.m, self.nbits, self.nprobe = nlist, m, nbits, nprobe
        self._chunks: list[Chunk] = []
        self._index = None
        self._matrix: np.ndarray | None = None

    # ------------------------------------------------------------------ index
    def index(self, chunks: Sequence[Chunk], show_progress: bool = False) -> None:
        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]
        matrix = self.embedder.encode(texts, batch_size=self.batch_size, show_progress=show_progress)
        if self.normalize:
            matrix = l2_normalize(matrix)
        self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self._index = self._build_faiss(self._matrix)

    def _build_faiss(self, matrix: np.ndarray):
        try:
            import faiss
        except ImportError:
            # No FAISS? Fall back to numpy brute force. Identical results, just
            # slower above a few hundred thousand vectors.
            return None
        dim = matrix.shape[1]
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(dim)
            index.add(matrix)
            return index
        if self.index_type == "ivfpq":
            n = matrix.shape[0]
            # FAISS trains 2**nbits centroids per sub-quantizer, so IVF-PQ needs
            # at least that many training vectors. Below that FAISS raises, and a
            # search engine must not crash on a small corpus - drop back to exact.
            nbits = self.nbits
            while nbits > 2 and 2 ** nbits > n:
                nbits -= 1
            if 2 ** nbits > n or n < 64:
                warnings.warn(
                    f"IVF-PQ needs at least {2 ** self.nbits} vectors to train and the corpus "
                    f"has {n}; falling back to an exact flat index.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                index = faiss.IndexFlatIP(dim)
                index.add(matrix)
                return index
            nlist = max(1, min(self.nlist, max(1, n // 39)))
            m = self.m
            while m > 1 and dim % m != 0:
                m -= 1
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
            index.train(matrix)
            index.add(matrix)
            index.nprobe = min(self.nprobe, nlist)
            return index
        raise ValueError(f"Unknown index_type: {self.index_type}")

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        if self._matrix is None:
            raise RuntimeError("index() must be called before search()")
        vector = self.embedder.encode([query], batch_size=1)
        if self.normalize:
            vector = l2_normalize(vector)
        vector = np.ascontiguousarray(vector, dtype=np.float32)

        k = min(top_k, len(self._chunks))
        if self._index is not None:
            scores, indices = self._index.search(vector, k)
            scores, indices = scores[0], indices[0]
        else:
            sims = (self._matrix @ vector[0]).astype(np.float32)
            indices = np.argsort(-sims)[:k]
            scores = sims[indices]

        hits: list[Hit] = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0:
                continue
            chunk = self._chunks[int(idx)]
            hits.append(Hit(chunk.chunk_id, chunk.doc_id, float(score), chunk.text, rank))
        return hits

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        """Persist the embedding matrix (the FAISS index is cheap to rebuild)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            matrix=self._matrix,
            chunk_ids=np.array([c.chunk_id for c in self._chunks]),
            doc_ids=np.array([c.doc_id for c in self._chunks]),
            texts=np.array([c.text for c in self._chunks]),
        )

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path), allow_pickle=False)
        self._matrix = np.ascontiguousarray(data["matrix"], dtype=np.float32)
        self._chunks = [
            Chunk(str(cid), str(did), str(text), position)
            for position, (cid, did, text) in enumerate(
                zip(data["chunk_ids"], data["doc_ids"], data["texts"])
            )
        ]
        self._index = self._build_faiss(self._matrix)

    @property
    def memory_mb(self) -> float:
        """Raw size of the stored vectors in megabytes (excludes FAISS overhead)."""
        return 0.0 if self._matrix is None else self._matrix.nbytes / (1024 * 1024)

    def __len__(self) -> int:
        return len(self._chunks)
