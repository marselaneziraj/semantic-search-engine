"""Command line interface.

    semsearch info    --dataset toy
    semsearch search  --dataset toy --system hybrid -q "why does keyword search fail"
    semsearch eval    --dataset beir/scifact
    semsearch ablate  --kind chunk --dataset beir/scifact
    semsearch serve   --dataset beir/scifact --system hybrid
    semsearch ui      --dataset beir/scifact

Every flag overrides the YAML experiment file, which in turn overrides the
defaults in semsearch/config.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semsearch import __version__
from semsearch.config import DEFAULT_CONFIG, Settings


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the YAML experiment file.")
    parser.add_argument("--dataset", default=None, help="ir_datasets id, jsonl:<folder>, or 'toy'.")
    parser.add_argument("--max-docs", type=int, default=None, help="Truncate the corpus.")
    parser.add_argument("--max-queries", type=int, default=None, help="Truncate the query set.")
    parser.add_argument("--top-k", type=int, default=None, help="Results returned / nDCG cutoff.")
    parser.add_argument("--eval-depth", type=int, default=None, help="Ranked-list depth (Recall@k cutoff).")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk length in tokens; 0 disables.")
    parser.add_argument("--dense-backend", choices=["sentence-transformers", "hashing"], default=None,
                        help="'hashing' runs fully offline with no model download.")
    parser.add_argument("--dense-model", default=None, help="sentence-transformers model id.")
    parser.add_argument("--rerank-backend", choices=["cross-encoder", "identity"], default=None)
    parser.add_argument("--rerank-candidates", type=int, default=None,
                        help="How many fused hits the cross-encoder rescores.")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_yaml(args.config)
    if args.dataset:
        settings.dataset = args.dataset
    if args.max_docs is not None:
        settings.max_docs = args.max_docs
    if args.max_queries is not None:
        settings.max_queries = args.max_queries
    if args.top_k is not None:
        settings.top_k = args.top_k
    if args.eval_depth is not None:
        settings.eval_depth = args.eval_depth
    if args.chunk_size is not None:
        settings.chunking.enabled = args.chunk_size > 0
        if args.chunk_size > 0:
            settings.chunking.size = args.chunk_size
            settings.chunking.overlap = min(settings.chunking.overlap, args.chunk_size // 4)
    if args.dense_backend:
        settings.dense.backend = args.dense_backend
    if args.dense_model:
        settings.dense.model_name = args.dense_model
    if args.rerank_backend:
        settings.rerank.backend = args.rerank_backend
    if args.rerank_candidates is not None:
        settings.rerank.candidates = args.rerank_candidates
    return settings


# --------------------------------------------------------------------- info
def cmd_info(args: argparse.Namespace) -> int:
    from semsearch.data.loaders import load_dataset

    settings = _settings_from_args(args)
    print(f"semantic-search-engine {__version__}")
    print(f"config: {settings.describe()}")
    dataset = load_dataset(settings.dataset, max_docs=settings.max_docs,
                           max_queries=settings.max_queries, split=settings.split_queries)
    print(dataset.summary())
    if dataset.documents:
        sample = dataset.documents[0]
        print(f"first document [{sample.doc_id}]: {sample.display_text[:160]}...")
    return 0


# ------------------------------------------------------------------- search
def cmd_search(args: argparse.Namespace) -> int:
    from semsearch.data.chunking import chunk_corpus
    from semsearch.data.loaders import load_dataset
    from semsearch.evaluation.harness import build_systems
    from semsearch.retrieval.base import dedupe_by_document, timed_search
    from semsearch.snippets import make_snippet

    settings = _settings_from_args(args)
    dataset = load_dataset(settings.dataset, max_docs=settings.max_docs,
                           max_queries=settings.max_queries, split=settings.split_queries)
    chunks = chunk_corpus(
        dataset.documents,
        size=settings.chunking.size,
        overlap=settings.chunking.overlap,
        min_chars=settings.chunking.min_chars,
        enabled=settings.chunking.enabled,
    )
    systems, _ = build_systems(settings, chunks, include=[args.system])
    retriever = systems[args.system]

    queries = args.query or []
    if not queries:
        print("No --query given; falling back to the first 3 dataset queries.")
        queries = list(dataset.queries.values())[:3]

    for query in queries:
        hits, took_ms = timed_search(retriever, query, top_k=max(settings.top_k * 3, settings.top_k))
        hits = dedupe_by_document(hits, settings.top_k)
        print(f"\n=== {args.system} | {query!r} | {took_ms:.1f} ms | {len(hits)} hits")
        for hit in hits:
            snippet = make_snippet(hit.text, query, width=180, markup=False)
            print(f"  {hit.rank:>2}. [{hit.doc_id}] score={hit.score:.4f}  {snippet}")
    return 0


# --------------------------------------------------------------------- eval
def cmd_eval(args: argparse.Namespace) -> int:
    from semsearch.data.loaders import load_dataset
    from semsearch.evaluation.harness import evaluate_systems, results_to_frame, save_results
    from semsearch.evaluation.report import write_report

    settings = _settings_from_args(args)
    dataset = load_dataset(settings.dataset, max_docs=settings.max_docs,
                           max_queries=settings.max_queries, split=settings.split_queries)
    print(dataset.summary())
    systems = args.systems.split(",") if args.systems else None
    results = evaluate_systems(dataset, settings, include=systems)

    frame = results_to_frame(results)
    slug = "".join(ch if ch.isalnum() else "-" for ch in dataset.name).strip("-")
    csv_path = save_results(results, Path(settings.results_dir) / f"{slug}_results.csv")
    report = write_report(frame, settings.results_dir, dataset.name, config_line=settings.describe())

    print("\n" + report["table"] + "\n")
    print(f"results csv : {csv_path}")
    print(f"results md  : {report['markdown']}")
    print(f"figure      : {report['figure']}")
    return 0


# ------------------------------------------------------------------ ablate
def cmd_ablate(args: argparse.Namespace) -> int:
    import pandas as pd

    from semsearch import ablations
    from semsearch.data.loaders import load_dataset

    settings = _settings_from_args(args)
    dataset = load_dataset(settings.dataset, max_docs=settings.max_docs,
                           max_queries=settings.max_queries, split=settings.split_queries)
    out_dir = Path(settings.results_dir) / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    kinds = ["chunk", "topk", "model", "index"] if args.kind == "all" else [args.kind]
    frames: list[pd.DataFrame] = []
    for kind in kinds:
        print(f"\n### ablation: {kind}")
        if kind == "chunk":
            frame = ablations.ablate_chunk_size(dataset, settings)
        elif kind == "topk":
            frame = ablations.ablate_rerank_depth(dataset, settings)
        elif kind == "model":
            frame = ablations.ablate_model_size(dataset, settings)
        elif kind == "index":
            frame = ablations.ablate_index_type(dataset, settings)
        else:
            raise ValueError(kind)
        frame.to_csv(out_dir / f"{kind}.csv", index=False)
        metric = "nDCG@" + str(settings.top_k)
        if metric in frame.columns:
            figure = ablations.plot_ablation(frame, out_dir / f"{kind}.png", metric=metric,
                                             title=f"Ablation: {kind} ({dataset.name})")
            print(f"figure: {figure}")
        print(frame.to_string(index=False))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "all_ablations.csv", index=False)
    print(f"\ncombined: {out_dir / 'all_ablations.csv'}")
    return 0


# -------------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    settings = _settings_from_args(args)
    # uvicorn imports the app in its own process, so the resolved settings are
    # handed over through the environment (which outranks the YAML file).
    os.environ["SEMSEARCH_CONFIG"] = args.config
    os.environ["SEMSEARCH_DATASET"] = settings.dataset
    os.environ["SEMSEARCH_SYSTEM"] = args.system
    os.environ["SEMSEARCH_DENSE__BACKEND"] = settings.dense.backend
    os.environ["SEMSEARCH_RERANK__BACKEND"] = settings.rerank.backend
    if settings.max_docs:
        os.environ["SEMSEARCH_MAX_DOCS"] = str(settings.max_docs)
    print(f"http://{args.host}:{args.port}/docs  (system={args.system}, dataset={settings.dataset})")
    uvicorn.run("semsearch.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# ----------------------------------------------------------------------- ui
def cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    app_path = Path(__file__).parent / "ui" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print("launching:", " ".join(command))
    return subprocess.call(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semsearch", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"semsearch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show the resolved config and dataset statistics.")
    _add_common(p_info)
    p_info.set_defaults(func=cmd_info)

    p_search = sub.add_parser("search", help="Run ad-hoc queries against one system.")
    _add_common(p_search)
    p_search.add_argument("-q", "--query", action="append", help="Repeatable.")
    p_search.add_argument("--system", default="hybrid",
                          choices=["bm25", "dense", "hybrid", "hybrid+rerank"])
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="Evaluate all four systems and write results + figure.")
    _add_common(p_eval)
    p_eval.add_argument("--systems", default=None,
                        help="Comma-separated subset, e.g. bm25,dense. Default: all four.")
    p_eval.set_defaults(func=cmd_eval)

    p_ablate = sub.add_parser("ablate", help="Run an ablation sweep and plot it.")
    _add_common(p_ablate)
    p_ablate.add_argument("--kind", default="chunk", choices=["chunk", "topk", "model", "index", "all"])
    p_ablate.set_defaults(func=cmd_ablate)

    p_serve = sub.add_parser("serve", help="Start the FastAPI service.")
    _add_common(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    # 8080 rather than uvicorn's usual 8000: on a developer machine 8000 is very
    # often already taken (another local app, a tunnel, a second service), and a
    # failed bind is a confusing first experience. Override with --port.
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--system", default="hybrid",
                         choices=["bm25", "dense", "hybrid", "hybrid+rerank", "all"])
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_ui = sub.add_parser("ui", help="Start the Streamlit UI.")
    _add_common(p_ui)
    p_ui.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())