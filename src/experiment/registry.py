"""Lightweight SQLite experiment registry for reproducible, comparable runs.

Every evaluation run records its identity, git commit, environment config,
dataset identity, hyper-parameters, and per-strategy/per-query metrics in a
single append-only store. This makes results auditable and queryable without a
heavy external tracking service.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import subprocess  # nosec B404
import uuid
from collections.abc import Iterable
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    git_commit    TEXT,
    config_json   TEXT NOT NULL,
    dataset_hash  TEXT,
    params_json   TEXT NOT NULL,
    notes         TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id    TEXT NOT NULL,
    strategy  TEXT NOT NULL,
    metric    TEXT NOT NULL,
    value     REAL NOT NULL,
    PRIMARY KEY (run_id, strategy, metric),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS per_query (
    run_id    TEXT NOT NULL,
    strategy  TEXT NOT NULL,
    query_id  TEXT NOT NULL,
    metric    TEXT NOT NULL,
    value     REAL NOT NULL,
    PRIMARY KEY (run_id, strategy, query_id, metric),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
"""


def _git_commit() -> str | None:
    """Return the short HEAD hash of the enclosing git repo, if any."""
    try:
        output = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )  # nosec B603, B607
    except (OSError, subprocess.SubprocessError):
        return None
    return output.stdout.strip() if output.returncode == 0 else None


class ExperimentRegistry:
    """SQLite-backed store for experiment runs and their metrics."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.executescript(_SCHEMA)

    def start_run(
        self,
        config: dict[str, object],
        params: dict[str, object],
        notes: str = "",
        dataset_hash: str | None = None,
    ) -> str:
        """Insert a run row and return its generated ``run_id``."""
        run_id = f"run_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._conn.execute(
            "INSERT INTO runs (run_id, created_at, git_commit, config_json, "
            "dataset_hash, params_json, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                _dt.datetime.now().isoformat(timespec="seconds"),
                _git_commit(),
                json.dumps(config, sort_keys=True),
                dataset_hash,
                json.dumps(params, sort_keys=True),
                notes,
            ),
        )
        self._conn.commit()
        return run_id

    def log_metrics(self, run_id: str, strategy: str, metrics: dict[str, float]) -> None:
        """Persist scalar metrics for one strategy within ``run_id``."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO metrics (run_id, strategy, metric, value) VALUES (?, ?, ?, ?)",
            [(run_id, strategy, name, value) for name, value in metrics.items()],
        )
        self._conn.commit()

    def log_per_query(
        self, run_id: str, strategy: str, query_id: str, metrics: dict[str, float]
    ) -> None:
        """Persist per-query scalar metrics, enabling granular failure analysis."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO per_query "
            "(run_id, strategy, query_id, metric, value) VALUES (?, ?, ?, ?, ?)",
            [(run_id, strategy, query_id, name, value) for name, value in metrics.items()],
        )
        self._conn.commit()

    def list_runs(self) -> list[dict]:
        """Return run metadata rows, newest first."""
        rows = self._conn.execute(
            "SELECT run_id, created_at, git_commit, dataset_hash, params_json, notes "
            "FROM runs ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
        return [
            {
                "run_id": run_id,
                "created_at": created_at,
                "git_commit": git_commit,
                "dataset_hash": dataset_hash,
                "params": json.loads(params_json) if params_json else {},
                "notes": notes,
            }
            for run_id, created_at, git_commit, dataset_hash, params_json, notes in rows
        ]

    def get_run(self, run_id: str) -> dict | None:
        """Return one run's metadata plus its strategy -> metric -> value map."""
        row = self._conn.execute(
            "SELECT run_id, created_at, git_commit, config_json, dataset_hash, "
            "params_json, notes FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "created_at": row[1],
            "git_commit": row[2],
            "config": json.loads(row[3]) if row[3] else {},
            "dataset_hash": row[4],
            "params": json.loads(row[5]) if row[5] else {},
            "notes": row[6],
            "metrics": self._metrics_for(run_id),
            "per_query": self._per_query_for(run_id),
        }

    def _metrics_for(self, run_id: str) -> dict[str, dict[str, float]]:
        """Return ``{strategy: {metric: value}}`` for a run."""
        metrics: dict[str, dict[str, float]] = {}
        for strategy, metric, value in self._conn.execute(
            "SELECT strategy, metric, value FROM metrics WHERE run_id = ?",
            (run_id,),
        ):
            metrics.setdefault(strategy, {})[metric] = value
        return metrics

    def _per_query_for(self, run_id: str) -> dict[str, dict[str, dict[str, float]]]:
        """Return ``{strategy: {query_id: {metric: value}}}`` for a run."""
        per_query: dict[str, dict[str, dict[str, float]]] = {}
        for strategy, query_id, metric, value in self._conn.execute(
            "SELECT strategy, query_id, metric, value FROM per_query WHERE run_id = ?",
            (run_id,),
        ):
            per_query.setdefault(strategy, {}).setdefault(query_id, {})[metric] = value
        return per_query

    def delete_run(self, run_id: str) -> bool:
        """Delete ``run_id`` and its metrics; returns True if a row was removed."""
        cursor = self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
        self._conn.execute("DELETE FROM per_query WHERE run_id = ?", (run_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> ExperimentRegistry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _exec(self, *args, **kwargs) -> Iterable:  # pragma: no cover - test hook
        return self._conn.execute(*args, **kwargs)
