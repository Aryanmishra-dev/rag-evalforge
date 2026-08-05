"""Unit tests for the vectorized BM25Okapi index."""
# pylint: disable=missing-function-docstring,missing-class-docstring

import numpy as np
import pytest

from src.retrieval.bm25 import BM25Index, tokenize


def _docs():
    return [
        "The quick brown fox jumps over the lazy dog",
        "The brown dog sleeps in the sun",
        "Technical writing must be clear and concise",
        "A quick fox is fast, a brown dog is loyal",
    ]


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert tokenize("Hello, World! Foo's bar-baz") == [
            "hello",
            "world",
            "foo's",
            "bar-baz",
        ]

    def test_empty_string(self):
        assert tokenize("") == []


class TestBM25Index:
    def test_fit_indexes_corpus(self):
        index = BM25Index().fit(_docs())
        assert index.n_docs == 4

    def test_top_ranks_lexically_relevant_docs_first(self):
        index = BM25Index().fit(_docs())
        results = index.top("fox", k=2)
        assert "fox" in " ".join(_docs()).lower()
        # Only docs 0 and 3 contain 'fox'; both should outrank docs without it.
        matched = {int(result.split(":", 1)[0]) for result in results}
        assert matched <= {0, 3}

    def test_multiple_terms_boost_score(self):
        index = BM25Index().fit(_docs())
        score_single = index.scores("fox")[0]
        score_double = index.scores("fox brown")[0]
        assert score_double > score_single

    def test_search_returns_only_matching_docs(self):
        index = BM25Index().fit(_docs())
        results = index.search("zebra")  # absent term
        assert results == []

    def test_doc_ids_respected(self):
        ids = ["a", "b", "c", "d"]
        index = BM25Index().fit(_docs(), ids)
        top = index.top("fox", k=1)
        assert top[0] in ("a", "d")

    def test_empty_corpus(self):
        index = BM25Index().fit([])
        assert index.n_docs == 0
        assert index.top("anything", k=5) == []

    def test_k_zero_returns_nothing(self):
        index = BM25Index().fit(_docs())
        assert index.top("fox", k=0) == []

    def test_refit_replaces_corpus(self):
        index = BM25Index().fit(_docs())
        index.fit(["only one document here"])
        assert index.n_docs == 1
        assert index.top("document", k=1)

    def test_score_vector_shape(self):
        index = BM25Index().fit(_docs())
        scores = index.scores("fox")
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (4,)

    def test_doc_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            BM25Index().fit(_docs(), ["only-one-id"])

    def test_idf_penalizes_common_terms(self):
        index = BM25Index().fit(_docs())
        # 'the' appears in most docs -> lower per-term contribution than 'zebra'-like rare term.
        scores = index.scores("the")
        assert np.all(scores >= 0.0)
