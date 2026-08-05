"""Unit tests for re-ranking."""
# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=unused-argument,import-outside-toplevel
# Fixtures passed only to exercise the (optional) cross-encoder path; the
# CrossEncoderReranker import is intentionally local to the fallback test.

from src.retrieval.rerank import EmbeddingReranker, build_reranker
from src.retrieval.types import RetrievedChunk


def _chunks(texts):
    return [
        RetrievedChunk(chunk_id=f"c{i}", text=text, page_number=i % 5 + 1, score=0.0)
        for i, text in enumerate(texts)
    ]


class TestEmbeddingReranker:
    def test_identical_text_ranks_first(self, fake_embed):
        chunks = _chunks(
            [
                "the quick brown fox jumps",
                "unrelated technical writing content",
                "completely different subject matter here",
            ]
        )
        ranked = EmbeddingReranker().rerank("the quick brown fox jumps", chunks, top_k=2)
        assert len(ranked) == 2
        # The exact-match chunk shares a deterministic embedding with the query.
        assert ranked[0].chunk_id == "c0"

    def test_assigns_scores_in_range(self, fake_embed):
        chunks = _chunks(["hello world", "foo bar"])
        ranked = EmbeddingReranker().rerank("hello", chunks, top_k=2)
        assert all(-1.0 <= c.score <= 1.0 for c in ranked)

    def test_top_k_truncates(self, fake_embed):
        chunks = _chunks(["a", "b", "c", "d"])
        assert len(EmbeddingReranker().rerank("a", chunks, top_k=1)) == 1

    def test_empty_candidates(self, fake_embed):
        assert EmbeddingReranker().rerank("query", [], top_k=3) == []

    def test_top_k_zero(self, fake_embed):
        chunks = _chunks(["a", "b"])
        assert EmbeddingReranker().rerank("query", chunks, top_k=0) == []

    def test_scores_are_consistent_with_cosine(self, fake_embed):
        chunks = _chunks(["same phrase here", "unrelated words"])
        ranked = EmbeddingReranker().rerank("same phrase here", chunks, top_k=2)
        assert ranked[0].score >= ranked[1].score


class TestBuildReranker:
    def test_default_is_embedding_reranker(self):
        assert isinstance(build_reranker(None), EmbeddingReranker)

    def test_cross_encoder_falls_back_when_unavailable(self, monkeypatch, fake_embed):
        monkeypatch.setattr("src.retrieval.rerank._CrossEncoder", None)
        from src.retrieval.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker("some/model")
        chunks = _chunks(["a", "b"])
        result = reranker.rerank("a", chunks, top_k=1)
        assert len(result) == 1
