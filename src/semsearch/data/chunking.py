"""Document chunking.

Why chunk at all? A bi-encoder squeezes a whole passage into one fixed-length
vector. The longer the passage, the more topics get averaged into that single
vector and the blurrier it becomes ("embedding dilution"). Splitting a long
document into overlapping windows keeps each vector about one thing. The cost is
index size: 3x more chunks means 3x more vectors to store and scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text."""

    chunk_id: str
    doc_id: str
    text: str
    position: int

    @property
    def is_whole_document(self) -> bool:
        return self.chunk_id == self.doc_id


def chunk_document(
    doc_id: str,
    text: str,
    size: int = 256,
    overlap: int = 32,
    min_chars: int = 20,
) -> list[Chunk]:
    """Split one document into overlapping windows of whitespace tokens.

    Args:
        doc_id: identifier of the parent document.
        text: the document text.
        size: window length in tokens.
        overlap: how many tokens the next window repeats from the previous one.
        min_chars: windows shorter than this (in characters) are dropped.

    Returns:
        A list of Chunk objects with ids of the form "<doc_id>#<position>".
        A document that fits in one window keeps the parent id unchanged, so
        short-document corpora (SciFact, NFCorpus) behave exactly as if chunking
        were off.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    tokens = text.split()
    if not tokens:
        return []
    if len(tokens) <= size:
        body = " ".join(tokens)
        return [Chunk(doc_id, doc_id, body, 0)] if len(body) >= min_chars else []

    stride = size - overlap
    chunks: list[Chunk] = []
    position = 0
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + size]
        body = " ".join(window)
        if len(body) >= min_chars:
            chunks.append(Chunk(f"{doc_id}#{position}", doc_id, body, position))
            position += 1
        if start + size >= len(tokens):
            break
    return chunks


def chunk_corpus(
    documents: Iterable["DocumentLike"],
    size: int = 256,
    overlap: int = 32,
    min_chars: int = 20,
    enabled: bool = True,
) -> list[Chunk]:
    """Chunk a whole corpus. With enabled=False every document becomes one chunk."""
    out: list[Chunk] = []
    for doc in documents:
        text = doc.text
        if not enabled:
            if len(text) >= min_chars:
                out.append(Chunk(doc.doc_id, doc.doc_id, text, 0))
            continue
        out.extend(chunk_document(doc.doc_id, text, size=size, overlap=overlap, min_chars=min_chars))
    return out


class DocumentLike:
    """Structural type: anything with .doc_id and .text."""

    doc_id: str
    text: str


def iter_batches(items: Sequence, batch_size: int) -> Iterator[Sequence]:
    """Yield consecutive slices of `items` of length `batch_size`."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
