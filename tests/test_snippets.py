from semsearch.snippets import best_window, highlight, make_snippet


def test_best_window_centres_on_the_query_terms():
    text = " ".join(["filler"] * 60 + ["reciprocal", "rank", "fusion"] + ["filler"] * 60)
    window = best_window(text, "reciprocal rank fusion", width=120)
    assert "reciprocal rank fusion" in window
    assert window.startswith("...")


def test_short_text_is_returned_unchanged():
    assert best_window("short text", "text", width=220) == "short text"


def test_highlight_wraps_terms_case_insensitively():
    marked = highlight("Reciprocal Rank Fusion works", "reciprocal fusion")
    assert "<mark>Reciprocal</mark>" in marked
    assert "<mark>Fusion</mark>" in marked
    assert "Rank" in marked  # not a query term, left alone


def test_highlight_covers_the_whole_word_for_a_prefix_match():
    assert highlight("the ranking of results", "rank") == "the <mark>ranking</mark> of results"


def test_highlight_ignores_short_and_stopword_terms():
    assert "<mark>" not in highlight("the cat sat", "the of a")


def test_make_snippet_without_markup_has_no_tags():
    snippet = make_snippet("alpha beta gamma", "beta", markup=False)
    assert "<mark>" not in snippet
