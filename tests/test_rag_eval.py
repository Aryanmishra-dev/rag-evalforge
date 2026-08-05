"""Unit tests for the RAG evaluation orchestration."""

import json

import pytest

from src.evaluation.rag_eval import (
    dataset_sha256,
    evaluate_rag,
    evaluate_retrieval,
    load_pairs,
    pages_of,
    reranked_retriever,
)
from src.retrieval.types import RetrievedChunk

PAIRS = [
    {
        "id": 1,
        "question": "What is a phrase?",
        "expected_answer": "A linguistic element without subject-predicate structure.",
        "page_number": 3,
    },
    {
        "id": 2,
        "question": "Name the clause types.",
        "expected_answer": "Four non-finite clause types.",
        "page_number": 4,
    },
]


def _retriever(by_question):
    """Build a retriever that returns page hits from a question->pages map."""
    # pylint: disable=missing-function-docstring,missing-class-docstring

    def retrieve(query, k):
        page = by_question[query]
        return [
            RetrievedChunk(
                chunk_id=f"c{i}",
                text=f"chunk for page {page}",
                page_number=page,
            )
            for i in range(min(k, 2))
        ]

    return retrieve


class TestPagesOf:
    def test_projects_page_numbers(self):
        chunks = [RetrievedChunk("a", "t", page_number=7)]
        assert pages_of(chunks) == [7]


class TestEvaluateRetrieval:
    def test_perfect_hits(self):
        retriever = _retriever({p["question"]: p["page_number"] for p in PAIRS})
        metrics = evaluate_retrieval(retriever, PAIRS, k=1)
        assert metrics["avg_hit_rate"] == pytest.approx(1.0)
        assert metrics["avg_mrr"] == pytest.approx(1.0)

    def test_misses(self):
        retriever = _retriever({"What is a phrase?": 99, "Name the clause types.": 98})
        metrics = evaluate_retrieval(retriever, PAIRS, k=1)
        assert metrics["avg_hit_rate"] == pytest.approx(0.0)
        assert metrics["avg_mrr"] == pytest.approx(0.0)

    def test_empty_pairs(self):
        assert evaluate_retrieval(_retriever({}), [], k=1) == {
            "avg_hit_rate": 0.0,
            "avg_mrr": 0.0,
        }


class TestEvaluateRag:
    def test_computes_all_metrics(self, fake_chat):
        retriever = _retriever({p["question"]: p["page_number"] for p in PAIRS})
        metrics = evaluate_rag(retriever, PAIRS, k=1, model="qwen2.5:7b")
        assert set(metrics) == {
            "avg_hit_rate",
            "avg_mrr",
            "avg_faithfulness",
            "avg_answer_correctness",
            "avg_answer_relevancy",
        }
        assert metrics["avg_hit_rate"] == pytest.approx(1.0)
        assert metrics["avg_mrr"] == pytest.approx(1.0)
        # Default judge response is 0.5 for every query.
        for key in ("avg_faithfulness", "avg_answer_correctness", "avg_answer_relevancy"):
            assert metrics[key] == pytest.approx(0.5)

    def test_generation_failure_scores_zero(self, fake_chat, monkeypatch):
        retriever = _retriever({p["question"]: p["page_number"] for p in PAIRS})

        def _boom(*args, **kwargs):
            raise ConnectionError("llm down")

        monkeypatch.setattr("src.evaluation.rag_eval.generate_answer", _boom)
        metrics = evaluate_rag(retriever, PAIRS, k=1)
        assert metrics["avg_faithfulness"] == pytest.approx(0.0)
        assert metrics["avg_answer_correctness"] == pytest.approx(0.0)
        assert metrics["avg_answer_relevancy"] == pytest.approx(0.0)


class TestRerankedRetriever:
    def test_expands_candidates_and_truncates(self, fake_embed, monkeypatch):
        calls = {"n": 0}
        retriever = _retriever({"q": 1})

        def counting(query, k):
            calls["n"] = k
            return retriever(query, k)

        reranked = reranked_retriever(counting, reranker=None, candidate_multiplier=3)

        # Build a minimal stub reranker to avoid pulling in embeddings twice.
        class StubReranker:
            def rerank(self, query, chunks, top_k):
                return chunks[:top_k]

        reranked = reranked_retriever(counting, StubReranker(), candidate_multiplier=3)
        results = reranked("q", k=2)
        assert calls["n"] == 6
        assert len(results) == 2


class TestDatasetUtils:
    def test_sha256_stable(self, tmp_path):
        path = tmp_path / "pairs.json"
        path.write_text("[1]", encoding="utf-8")
        first = dataset_sha256(path)
        assert first == dataset_sha256(path)
        assert len(first) == 12

    def test_load_pairs_validates_schema(self, tmp_path):
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps(PAIRS), encoding="utf-8")
        assert load_pairs(path) == PAIRS

    def test_load_pairs_rejects_missing_fields(self, tmp_path):
        path = tmp_path / "pairs.json"
        path.write_text(json.dumps([{"question": "only question"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="missing fields"):
            load_pairs(path)
