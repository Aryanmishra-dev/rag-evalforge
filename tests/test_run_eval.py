"""Integration-style tests for the benchmark driver, using fake collections."""

# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=redefined-outer-name,unused-argument,protected-access
# Fixtures shadow module-level names; some are injected just to patch
# internals. `_pretty_print` is exercised directly in the capsys test.
import pytest

from src.config import EMBED_MODEL
from src.embeddings.chroma_client import collection_name
from src.evaluation import run_eval
from tests.fakes import FakeCollection


@pytest.fixture
def fake_chroma(monkeypatch):
    """Replace Chroma collections with in-memory fakes for every strategy."""
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Technical writing must be clear and concise.",
        "A fox is fast and clever in the forest.",
        "Concise writing saves the reader time.",
    ]

    def get_collection_for_model(strategy, embed_model):
        return FakeCollection(collection_name(strategy, embed_model), docs)

    monkeypatch.setattr(run_eval, "get_collection_for_model", get_collection_for_model)


class TestRunEval:
    def test_retrieval_only_results_shape(self, fake_chroma):
        results = run_eval.run_eval(k=2, include_hybrid=False, full_rag=False)
        assert set(results) == set(run_eval.STRATEGIES)
        for metrics in results.values():
            assert {"avg_hit_rate", "avg_mrr", "n_chunks"} <= set(metrics)
            assert 0.0 <= metrics["avg_hit_rate"] <= 1.0
            assert 0.0 <= metrics["avg_mrr"] <= 1.0

    def test_hybrid_adds_pipelines(self, fake_chroma, fake_embed):
        results = run_eval.run_eval(k=2, include_hybrid=True, full_rag=False)
        assert "hybrid" in results
        assert "hybrid_rerank" in results
        assert results["hybrid"]["n_chunks"] == 4

    def test_full_rag_adds_generation_metrics(self, fake_chroma, fake_chat):
        results = run_eval.run_eval(k=2, include_hybrid=False, full_rag=True)
        for metrics in results.values():
            assert {
                "avg_faithfulness",
                "avg_answer_correctness",
                "avg_answer_relevancy",
            } <= set(metrics)

    def test_custom_pairs_override(self, fake_chroma):
        pairs = [
            {
                "question": "fox",
                "expected_answer": "a fast animal",
                "page_number": 1,
            }
        ]
        results = run_eval.run_eval(k=2, include_hybrid=False, full_rag=False, pairs=pairs)
        assert results["fixed"]["avg_hit_rate"] == pytest.approx(1.0)

    def test_multi_embed_models_namespace_keys(self, fake_chroma):
        models = [EMBED_MODEL, "other-embed:1"]
        results = run_eval.run_eval(k=2, include_hybrid=False, full_rag=False, embed_models=models)
        for strategy in run_eval.STRATEGIES:
            assert strategy in results
            assert f"{strategy}@other_embed_1" in results
        assert len(results) == 2 * len(run_eval.STRATEGIES)

    def test_hybrid_keyed_by_model(self, fake_chroma, fake_embed):
        models = [EMBED_MODEL, "other-embed:1"]
        results = run_eval.run_eval(k=2, include_hybrid=True, full_rag=False, embed_models=models)
        assert "hybrid" in results
        assert "hybrid@other_embed_1" in results
        assert "hybrid_rerank@other_embed_1" in results


class TestMetricKey:
    def test_default_model_untagged(self):
        assert run_eval.metric_key("fixed", EMBED_MODEL) == "fixed"

    def test_other_model_tagged(self):
        assert run_eval.metric_key("fixed", "mxbai-embed-large:latest") == (
            "fixed@mxbai_embed_large_latest"
        )


def test_pretty_print_renders_all_modes(capsys):
    results = {
        "fixed": {
            "avg_hit_rate": 1.0,
            "avg_mrr": 0.9,
            "avg_faithfulness": 0.8,
            "avg_answer_correctness": 0.7,
            "avg_answer_relevancy": 0.6,
        }
    }
    run_eval._pretty_print(results, full_rag=True)
    run_eval._pretty_print(results, full_rag=False)
    out = capsys.readouterr().out
    assert "faith" in out
