import pandas as pd

from semsearch.evaluation.harness import evaluate_systems, results_to_frame, save_results
from semsearch.evaluation.report import plot_results, render_markdown_table, write_report


def test_end_to_end_evaluation_of_all_four_systems(toy_dataset, offline_settings):
    results = evaluate_systems(toy_dataset, offline_settings, verbose=False)
    assert [r.system for r in results] == ["bm25", "dense", "hybrid", "hybrid+rerank"]
    for result in results:
        assert result.n_queries == 5
        assert result.n_documents == 12
        assert 0.0 <= result.metrics["nDCG@10"] <= 1.0
        assert 0.0 <= result.metrics["Recall@100"] <= 1.0
        assert result.mean_latency_ms >= 0.0
        assert result.p95_latency_ms >= result.mean_latency_ms - 1e-9


def test_bm25_beats_hashed_embeddings_on_the_toy_corpus(toy_dataset, offline_settings):
    """Sanity anchor: a hashed bag-of-words dense model is a weak retriever, so
    BM25 should not lose to it. If this flips, something is wired wrong."""
    results = {r.system: r for r in evaluate_systems(toy_dataset, offline_settings,
                                                     include=["bm25", "dense"], verbose=False)}
    assert results["bm25"].metrics["nDCG@10"] >= results["dense"].metrics["nDCG@10"]


def test_run_files_are_written_in_trec_format(toy_dataset, offline_settings):
    evaluate_systems(toy_dataset, offline_settings, include=["bm25"], verbose=False)
    runs = list((offline_settings.results_dir / "runs").glob("*.run"))
    assert runs
    first_line = runs[0].read_text(encoding="utf-8").splitlines()[0].split()
    assert len(first_line) == 6
    assert first_line[1] == "Q0"


def test_results_csv_and_report_are_written(toy_dataset, offline_settings):
    results = evaluate_systems(toy_dataset, offline_settings, include=["bm25", "dense"], verbose=False)
    frame = results_to_frame(results)
    csv_path = save_results(results, offline_settings.results_dir / "toy_results.csv")
    assert csv_path.exists()
    assert set(["system", "nDCG@10", "mean_latency_ms"]).issubset(pd.read_csv(csv_path).columns)

    table = render_markdown_table(frame)
    assert table.startswith("| system |")
    assert "bm25" in table

    figure = plot_results(frame, offline_settings.results_dir / "toy.png")
    assert figure.exists() and figure.stat().st_size > 1000

    report = write_report(frame, offline_settings.results_dir, "toy")
    assert report["markdown"].exists()


def test_unjudged_corpus_is_rejected_with_a_clear_message(offline_settings):
    import pytest

    from semsearch.data.loaders import Dataset, Document

    dataset = Dataset(name="unjudged", documents=[Document("d1", "text")], queries={"q1": "text"})
    with pytest.raises(ValueError, match="no qrels"):
        evaluate_systems(dataset, offline_settings, verbose=False)
