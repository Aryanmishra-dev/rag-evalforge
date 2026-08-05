"""Unit tests for retrieval quality metrics."""
# pylint: disable=missing-function-docstring,missing-class-docstring

from typing import cast

import pytest
from chromadb import QueryResult

from src.evaluation.metrics import extract_pages, hit_rate_at_k, reciprocal_rank


class TestHitRate:
    def test_hit_present(self):
        assert hit_rate_at_k([3, 5, 7], 5) == 1

    def test_hit_absent(self):
        assert hit_rate_at_k([3, 5, 7], 9) == 0

    def test_empty_retrieval(self):
        assert hit_rate_at_k([], 1) == 0


class TestReciprocalRank:
    def test_first_rank(self):
        assert reciprocal_rank([4, 1, 2], 4) == pytest.approx(1.0)

    def test_third_rank(self):
        assert reciprocal_rank([4, 1, 2], 2) == pytest.approx(1.0 / 3.0)

    def test_not_found(self):
        assert reciprocal_rank([4, 1, 2], 9) == 0.0

    def test_duplicate_page_uses_first_occurrence(self):
        assert reciprocal_rank([5, 5, 5], 5) == pytest.approx(1.0)


class TestExtractPages:
    def test_flattens_metadatas(self):
        result = {
            "metadatas": [
                [
                    {"page_number": 2},
                    {"page_number": 9},
                    {"page_number": 2},
                ]
            ]
        }
        assert extract_pages(cast(QueryResult, result)) == [2, 9, 2]

    def test_empty_result(self):
        result = {"metadatas": [[]]}
        assert extract_pages(cast(QueryResult, result)) == []
