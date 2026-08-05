"""Unit tests for Reciprocal Rank Fusion."""
# pylint: disable=missing-function-docstring,missing-class-docstring

from src.retrieval.rrf import reciprocal_rank_fusion


class TestRRF:
    def test_agreement_surfaces_top(self):
        result = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
        assert result[0] == "a"

    def test_top_rank_beats_multiple_low_ranks(self):
        # 'z' is ranked 1 in one list; 'a' appears mid-list in two lists.
        result = reciprocal_rank_fusion([["z", "x", "y"], ["a", "b", "c", "z"]])
        assert result[0] == "z"

    def test_fuses_three_lists(self):
        result = reciprocal_rank_fusion([["a", "b"], ["b", "a"], ["b", "c"]])
        assert result[0] == "b"

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([[], [], []]) == []

    def test_no_duplicates_in_output(self):
        result = reciprocal_rank_fusion([["a", "b"], ["b", "a"], ["a"]])
        assert len(result) == len(set(result))

    def test_k_smoothing_parameter(self):
        # A large k flattens rank differences but preserves relative order.
        small = reciprocal_rank_fusion([["a", "b"], ["a", "b"]], k=1)
        large = reciprocal_rank_fusion([["a", "b"], ["a", "b"]], k=1000)
        assert small == ["a", "b"]
        assert large == ["a", "b"]

    def test_unknown_ids_still_fused(self):
        result = reciprocal_rank_fusion([["a"], ["x", "a"]])
        assert set(result) == {"a", "x"}
