"""Offline tests for the MCP server's tool functions (using fakes)."""

# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=redefined-outer-name,unused-argument
# pylint: disable=too-many-arguments,too-many-positional-arguments
# Fixtures shadow module-level names; some are injected just to patch internals.
# The run_eval stubs replicate the real 6-arg signature (disabled there too).

import pytest

from src import mcp_server
from src.experiment.registry import ExperimentRegistry
from tests.fakes import FakeCollection


@pytest.fixture
def fake_ingest(monkeypatch):
    """Replace the pipeline ingest with a fake that reports parsed output."""

    def fake_ingest(pdf_path, embed_model):
        print(f"Parsed {pdf_path} | embed_model={embed_model}")

    monkeypatch.setattr(mcp_server, "ingest", fake_ingest)
    return fake_ingest


@pytest.fixture
def fake_chroma(monkeypatch):
    """Replace collection lookups with in-memory fakes with known counts."""
    counts = {"fixed": 3, "recursive": 4, "sentence": 5, "semantic": 6}

    def get_collection_for_model(strategy, embed_model=None):
        return FakeCollection(strategy, [f"doc{i}" for i in range(counts[strategy])])

    monkeypatch.setattr(mcp_server, "get_collection_for_model", get_collection_for_model)
    return counts


class TestIngestPdf:
    def test_returns_captured_ingest_output(self, fake_ingest, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        out = mcp_server.ingest_pdf(str(pdf))
        assert "embed_model=nomic-embed-text" in out

    def test_missing_pdf_raises(self, fake_ingest):
        with pytest.raises(ValueError, match="PDF not found"):
            mcp_server.ingest_pdf("/no/such/file.pdf")

    def test_relative_path_resolved_from_cwd(self, fake_ingest, monkeypatch, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.chdir(tmp_path)
        out = mcp_server.ingest_pdf("doc.pdf")
        assert "embed_model=nomic-embed-text" in out


class TestRunBenchmark:
    def test_wraps_pipeline_results(self, monkeypatch):
        captured = {}

        def fake_run_eval(k, include_hybrid, full_rag, model, embed_models, progress_cb):
            captured.update(
                k=k,
                include_hybrid=include_hybrid,
                full_rag=full_rag,
                model=model,
                embed_models=embed_models,
            )
            progress_cb("fixed", 1, 1)
            return {"fixed": {"avg_hit_rate": 1.0, "avg_mrr": 0.9}}

        monkeypatch.setattr(mcp_server, "run_eval", fake_run_eval)
        result = mcp_server.run_benchmark(
            k=3, include_hybrid=True, full_rag=False, model=None, embed_models="a,b"
        )
        assert captured["k"] == 3
        assert captured["include_hybrid"] is True
        assert captured["embed_models"] == ["a", "b"]
        assert result["results"]["fixed"]["avg_hit_rate"] == 1.0

    def test_defaults_passthrough(self, monkeypatch):
        captured = {}

        def fake_run_eval(k, include_hybrid, full_rag, model, embed_models, progress_cb):
            captured.update(
                k=k,
                include_hybrid=include_hybrid,
                full_rag=full_rag,
                model=model,
                embed_models=embed_models,
            )
            return {}

        monkeypatch.setattr(mcp_server, "run_eval", fake_run_eval)
        mcp_server.run_benchmark()
        assert captured == {
            "k": 5,
            "include_hybrid": False,
            "full_rag": False,
            "model": None,
            "embed_models": None,
        }


def test_list_collections_counts(fake_chroma):
    assert mcp_server.list_collections() == {
        "fixed": 3,
        "recursive": 4,
        "sentence": 5,
        "semantic": 6,
    }


class TestRegistryTools:
    @pytest.fixture
    def registry_db(self, monkeypatch, tmp_path):
        db = tmp_path / "experiments.db"
        monkeypatch.setattr(mcp_server, "EXPERIMENT_DB_PATH", str(db))
        return db

    def test_run_lifecycle(self, registry_db):
        assert mcp_server.list_runs() == []
        with ExperimentRegistry(registry_db) as registry:
            run_id = registry.start_run(config={"llm_model": "m"}, params={"k": 5})
            registry.log_metrics(run_id, "fixed", {"avg_hit_rate": 0.9})

        assert [run["run_id"] for run in mcp_server.list_runs()] == [run_id]
        run = mcp_server.get_run(run_id)
        assert run["metrics"]["fixed"]["avg_hit_rate"] == 0.9

        assert mcp_server.delete_run(run_id).startswith("Deleted run")
        assert mcp_server.get_run(run_id) is None

    def test_delete_missing_run_raises(self, registry_db):
        with pytest.raises(ValueError, match="No run found"):
            mcp_server.delete_run("run_does_not_exist")
