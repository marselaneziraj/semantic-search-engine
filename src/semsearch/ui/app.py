"""Minimal Streamlit search UI.

    streamlit run src/semsearch/ui/app.py
    # or: semsearch ui

Shows the four systems side by side with highlighted snippets and the measured
latency, which is the honest way to present the quality/latency trade-off.
"""

from __future__ import annotations

import time

import streamlit as st

from semsearch.config import Settings
from semsearch.data.chunking import chunk_corpus
from semsearch.data.loaders import load_dataset
from semsearch.evaluation.harness import build_systems
from semsearch.retrieval.base import dedupe_by_document, timed_search
from semsearch.snippets import make_snippet

st.set_page_config(page_title="semantic-search-engine", page_icon="*", layout="wide")


@st.cache_resource(show_spinner="Building indexes (first run only) ...")
def load_engine(dataset_id: str, dense_backend: str, chunk_size: int, system: str):
    settings = Settings.from_yaml()
    settings.dataset = dataset_id
    settings.dense.backend = dense_backend
    settings.chunking.enabled = chunk_size > 0
    if chunk_size > 0:
        settings.chunking.size = chunk_size
    dataset = load_dataset(settings.dataset, max_docs=settings.max_docs, split=settings.split_queries)
    chunks = chunk_corpus(
        dataset.documents,
        size=settings.chunking.size,
        overlap=settings.chunking.overlap,
        enabled=settings.chunking.enabled,
    )
    include = ["bm25", "dense", "hybrid", "hybrid+rerank"] if system == "all" else [system]
    systems, _ = build_systems(settings, chunks, include=include, verbose=False)
    return settings, dataset, chunks, systems


st.title("semantic-search-engine")
st.caption("BM25 vs dense vs hybrid (RRF) vs hybrid + cross-encoder rerank")

with st.sidebar:
    st.header("Index")
    dataset_id = st.text_input("Dataset", value="toy", help="toy | beir/scifact | jsonl:data/mine")
    dense_backend = st.selectbox(
        "Dense backend", ["hashing", "sentence-transformers"],
        help="'hashing' needs no model download; 'sentence-transformers' is the real bi-encoder.",
    )
    chunk_size = st.number_input("Chunk size (0 = off)", min_value=0, max_value=1024, value=256, step=64)
    system = st.selectbox("System", ["all", "bm25", "dense", "hybrid", "hybrid+rerank"])
    top_k = st.slider("Results", min_value=3, max_value=25, value=10)

settings, dataset, chunks, systems = load_engine(dataset_id, dense_backend, int(chunk_size), system)
st.sidebar.success(f"{len(dataset.documents):,} docs -> {len(chunks):,} chunks")

example = next(iter(dataset.queries.values()), "reciprocal rank fusion")
query = st.text_input("Query", value=example)

if query:
    columns = st.columns(len(systems))
    for column, (name, retriever) in zip(columns, systems.items()):
        with column:
            st.subheader(name)
            hits, took_ms = timed_search(retriever, query, top_k=max(top_k * 3, top_k))
            hits = dedupe_by_document(hits, top_k)
            st.metric("latency", f"{took_ms:.0f} ms")
            for hit in hits:
                st.markdown(f"**{hit.rank}. `{hit.doc_id}`** - score {hit.score:.4f}")
                st.markdown(
                    make_snippet(hit.text, query, width=240, markup=True),
                    unsafe_allow_html=True,
                )
                st.divider()