"""Query-biased snippet extraction with term highlighting.

Showing the first 200 characters of a document is lazy and often useless. A
query-biased snippet picks the window of the text where the query terms are
densest, which is what every real search UI does.
"""

from __future__ import annotations

import re
from typing import Sequence

from semsearch.retrieval.bm25 import tokenize


def best_window(text: str, query: str, width: int = 220) -> str:
    """Return the ~width-character window containing the most query terms."""
    if not text:
        return ""
    terms = set(tokenize(query))
    if not terms or len(text) <= width:
        return text[:width]

    words = text.split()
    window_words = max(8, width // 6)
    best_start, best_score = 0, -1
    for start in range(0, max(1, len(words) - window_words + 1)):
        window = words[start : start + window_words]
        score = sum(1 for word in window if re.sub(r"[^a-z0-9]", "", word.lower()) in terms)
        if score > best_score:
            best_start, best_score = start, score
    snippet = " ".join(words[best_start : best_start + window_words])
    prefix = "... " if best_start > 0 else ""
    suffix = " ..." if best_start + window_words < len(words) else ""
    return f"{prefix}{snippet}{suffix}"


def highlight(text: str, query: str, open_tag: str = "<mark>", close_tag: str = "</mark>") -> str:
    """Wrap every query term occurrence in the given tags (case-insensitive)."""
    terms = sorted({t for t in tokenize(query) if len(t) > 2}, key=len, reverse=True)
    if not terms:
        return text
    # \w* after the term so that a query for "rank" highlights the whole of
    # "ranking" rather than leaving a ragged "<mark>rank</mark>ing" in the UI.
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\w*", re.IGNORECASE)
    return pattern.sub(lambda m: f"{open_tag}{m.group(0)}{close_tag}", text)


def make_snippet(text: str, query: str, width: int = 220, markup: bool = True) -> str:
    window = best_window(text, query, width=width)
    return highlight(window, query) if markup else window
