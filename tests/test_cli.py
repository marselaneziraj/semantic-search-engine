from semsearch.cli import build_parser, main


def test_parser_requires_a_subcommand():
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_info_command_runs_on_the_toy_dataset(capsys):
    assert main(["info", "--dataset", "toy"]) == 0
    out = capsys.readouterr().out
    assert "12 documents" in out


def test_search_command_prints_ranked_results(capsys):
    exit_code = main([
        "search", "--dataset", "toy", "--system", "bm25",
        "--dense-backend", "hashing", "--rerank-backend", "identity",
        "-q", "reciprocal rank fusion",
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[d1]" in out
    assert "ms" in out


def test_chunk_size_zero_disables_chunking():
    parser = build_parser()
    args = parser.parse_args(["info", "--chunk-size", "0"])
    assert args.chunk_size == 0
