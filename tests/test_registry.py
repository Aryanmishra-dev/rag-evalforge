"""Unit tests for the SQLite experiment registry."""
# pylint: disable=missing-function-docstring,missing-class-docstring

import pytest

from src.experiment.registry import ExperimentRegistry


@pytest.fixture
def registry(tmp_path):
    reg = ExperimentRegistry(tmp_path / "experiments.db")
    yield reg
    reg.close()


class TestRegistry:
    def test_start_run_returns_clean_id(self, registry):
        run_id = registry.start_run({"a": 1}, {"k": 5}, notes="hello")
        assert run_id.startswith("run_")

    def test_run_persists_config_and_params(self, registry):
        run_id = registry.start_run({"model": "qwen"}, {"k": 3}, dataset_hash="abc123")
        run = registry.get_run(run_id)
        assert run["config"]["model"] == "qwen"
        assert run["params"]["k"] == 3
        assert run["dataset_hash"] == "abc123"
        assert run["git_commit"] is None or isinstance(run["git_commit"], str)

    def test_log_and_read_metrics(self, registry):
        run_id = registry.start_run({}, {})
        registry.log_metrics(run_id, "recursive", {"avg_mrr": 0.85, "n_chunks": 120})
        run = registry.get_run(run_id)
        assert run["metrics"]["recursive"]["avg_mrr"] == pytest.approx(0.85)
        assert run["metrics"]["recursive"]["n_chunks"] == 120

    def test_log_metrics_is_idempotent(self, registry):
        run_id = registry.start_run({}, {})
        registry.log_metrics(run_id, "fixed", {"avg_hit_rate": 0.9})
        registry.log_metrics(run_id, "fixed", {"avg_hit_rate": 0.95})
        run = registry.get_run(run_id)
        assert run["metrics"]["fixed"]["avg_hit_rate"] == pytest.approx(0.95)

    def test_log_per_query(self, registry):
        run_id = registry.start_run({}, {})
        registry.log_per_query(run_id, "fixed", "q1", {"hit": 1.0})
        registry.log_per_query(run_id, "fixed", "q2", {"hit": 0.0})
        run = registry.get_run(run_id)
        assert run["per_query"]["fixed"]["q1"]["hit"] == pytest.approx(1.0)
        assert run["per_query"]["fixed"]["q2"]["hit"] == pytest.approx(0.0)
        # Rolled-up metrics stay empty when only per-query data is logged.
        assert run["metrics"] == {}

    def test_list_runs_newest_first(self, registry):
        registry.start_run({}, {"order": 1})
        registry.start_run({}, {"order": 2})
        runs = registry.list_runs()
        assert len(runs) == 2
        assert runs[0]["params"]["order"] == 2

    def test_get_run_missing_returns_none(self, registry):
        assert registry.get_run("nope") is None

    def test_delete_run(self, registry):
        run_id = registry.start_run({}, {})
        registry.log_metrics(run_id, "fixed", {"avg_mrr": 1.0})
        assert registry.delete_run(run_id) is True
        assert registry.get_run(run_id) is None
        assert registry.delete_run(run_id) is False

    def test_context_manager_closes(self, tmp_path):
        with ExperimentRegistry(tmp_path / "x.db") as reg:
            reg.start_run({}, {})
        # second open works after clean close
        with ExperimentRegistry(tmp_path / "x.db") as reg:
            assert reg.list_runs()  # non-empty
