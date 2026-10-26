"""Corpus / query / qrels loading.

Three sources are supported, chosen by the `dataset` string:

    "beir/scifact"        -> ir_datasets (downloads once into ~/.ir_datasets);
                             the judged split is resolved automatically, so both
                             "beir/scifact" and "beir/scifact/test" work
    "jsonl:data/mine"     -> a folder with corpus.jsonl, queries.jsonl, qrels.tsv
    "toy"                 -> the tiny bundled corpus in data/toy (no network)

The JSONL format is the BEIR format, so anything you scrape can be dropped in:
    corpus.jsonl   {"_id": "d1", "title": "...", "text": "..."}
    queries.jsonl  {"_id": "q1", "text": "..."}
    qrels.tsv      query-id<TAB>corpus-id<TAB>score   (header row allowed)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from semsearch.config import PROJECT_ROOT


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""

    @property
    def display_text(self) -> str:
        return f"{self.title}. {self.text}".strip(". ") if self.title else self.text


@dataclass
class Dataset:
    """A corpus plus (optionally) queries and relevance judgements."""

    name: str
    documents: list[Document]
    queries: dict[str, str] = field(default_factory=dict)
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def has_judgements(self) -> bool:
        return bool(self.qrels)

    def summary(self) -> str:
        return (
            f"{self.name}: {len(self.documents):,} documents, "
            f"{len(self.queries):,} queries, "
            f"{sum(len(v) for v in self.qrels.values()):,} judgements"
        )


def load_jsonl_corpus(path: str | Path, max_docs: int | None = None) -> list[Document]:
    """Read a BEIR-style corpus.jsonl (or any .jsonl with _id/text fields)."""
    path = Path(path)
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("_id") or row.get("id") or row.get("doc_id") or len(documents))
            text = row.get("text") or row.get("body") or row.get("contents") or ""
            documents.append(Document(doc_id=doc_id, text=text, title=row.get("title", "")))
            if max_docs and len(documents) >= max_docs:
                break
    return documents


def _load_jsonl_dataset(folder: str | Path, max_docs: int | None, max_queries: int | None) -> Dataset:
    folder = Path(folder)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    corpus_path = folder / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"No corpus.jsonl in {folder}")
    documents = load_jsonl_corpus(corpus_path, max_docs=max_docs)

    queries: dict[str, str] = {}
    queries_path = folder / "queries.jsonl"
    if queries_path.exists():
        with queries_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                queries[str(row.get("_id") or row.get("id"))] = row.get("text", "")

    qrels: dict[str, dict[str, int]] = {}
    qrels_path = folder / "qrels.tsv"
    if qrels_path.exists():
        with qrels_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                if row[0].strip().lower() in {"query-id", "qid"}:
                    continue  # header
                qid, did, score = row[0].strip(), row[1].strip(), row[2].strip()
                qrels.setdefault(qid, {})[did] = int(float(score))

    if max_queries:
        keep = list(queries)[:max_queries] if queries else list(qrels)[:max_queries]
        queries = {q: queries[q] for q in keep if q in queries}
        qrels = {q: qrels[q] for q in keep if q in qrels}

    return Dataset(name=f"jsonl:{folder.name}", documents=documents, queries=queries, qrels=qrels)


def _resolve_ir_dataset(dataset_id: str, split: str = "test"):
    """Return (resolved_id, handle), picking the split that actually has qrels.

    In ir_datasets a BEIR entry is a tree: `beir/scifact` holds the documents and
    the queries, while the relevance judgements live on its splits
    (`beir/scifact/test`, `beir/scifact/train`). Calling qrels_iter() on the
    parent raises AttributeError, so the split is resolved here instead of making
    the user remember which datasets are split and which are not - `beir/trec-covid`,
    for instance, carries its qrels on the parent id and has no splits at all.
    """
    import ir_datasets

    handle = ir_datasets.load(dataset_id)
    if handle.has_qrels():
        return dataset_id, handle

    for candidate_split in dict.fromkeys([split, "test", "dev", "train"]):
        candidate = f"{dataset_id}/{candidate_split}"
        try:
            candidate_handle = ir_datasets.load(candidate)
        except KeyError:
            continue
        if candidate_handle.has_qrels():
            return candidate, candidate_handle

    # No judged split anywhere: return the parent and let the caller decide. The
    # corpus is still searchable, it just cannot be scored.
    return dataset_id, handle


def _load_ir_datasets(
    dataset_id: str,
    max_docs: int | None,
    max_queries: int | None,
    split: str = "test",
) -> Dataset:
    try:
        import ir_datasets  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "ir_datasets is required for BEIR datasets. Install it with:\n"
            "    pip install ir_datasets"
        ) from exc

    resolved_id, handle = _resolve_ir_dataset(dataset_id, split)

    documents: list[Document] = []
    for doc in handle.docs_iter():
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or getattr(doc, "body", "") or ""
        documents.append(Document(doc_id=doc.doc_id, text=text, title=title))
        if max_docs and len(documents) >= max_docs:
            break

    queries = {q.query_id: q.text for q in handle.queries_iter()}

    qrels: dict[str, dict[str, int]] = {}
    if handle.has_qrels():
        for qrel in handle.qrels_iter():
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)

    # Keep only judged queries: an unjudged query contributes 0 to every metric
    # and would silently drag the averages down.
    if qrels:
        queries = {qid: text for qid, text in queries.items() if qid in qrels}
    if max_queries:
        keep = list(queries)[:max_queries]
        queries = {q: queries[q] for q in keep}
        qrels = {q: qrels[q] for q in keep if q in qrels}

    return Dataset(name=resolved_id, documents=documents, queries=queries, qrels=qrels)


def load_dataset(
    dataset: str = "beir/scifact",
    max_docs: int | None = None,
    max_queries: int | None = None,
    split: str = "test",
) -> Dataset:
    """Load a dataset by id. See module docstring for the accepted forms.

    For ir_datasets ids the `split` is only a preference: if the id given already
    has qrels it is used as-is, otherwise the judged split is looked up, so both
    `beir/scifact` and `beir/scifact/test` work and resolve to the same data.
    """
    if dataset == "toy":
        return _load_jsonl_dataset(PROJECT_ROOT / "data" / "toy", max_docs, max_queries)
    if dataset.startswith("jsonl:"):
        return _load_jsonl_dataset(dataset.split(":", 1)[1], max_docs, max_queries)
    return _load_ir_datasets(dataset, max_docs, max_queries, split=split)