"""MCP server exposing the RAG EvalForge pipeline to LLM clients.

This is a thin wrapper only: every tool delegates to the existing pipeline
functions (``src.ingestion.ingest``, ``src.evaluation.run_eval``, and the
``src.experiment.registry`` SQLite store). No logic is reimplemented here.

The server chdirs to the repository root on import so ``.env`` and the
relative data paths in ``src.config`` resolve no matter how the process is
launched (e.g. from an MCP client such as Claude Desktop).

Run on stdio transport::

    python src/mcp_server.py
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from src.config import EMBED_MODEL, EXPERIMENT_DB_PATH
from src.embeddings.chroma_client import get_collection_for_model
from src.evaluation.run_eval import STRATEGIES, run_eval
from src.experiment.registry import ExperimentRegistry
from src.ingestion.ingest import ingest

_LOGGER = logging.getLogger(__name__)

mcp = FastMCP(
    "rag-evalforge",
    instructions=(
        "Tools to ingest PDFs, benchmark chunking strategies, and inspect the "
        "SQLite experiment registry of the RAG EvalForge harness."
    ),
)


@mcp.tool()
def ingest_pdf(pdf_path: str, embed_model: str = EMBED_MODEL) -> str:
    """Parse, chunk, and embed a PDF into all four strategy collections.

    Delegates to ``src.ingestion.ingest.ingest``; the returned string reports
    the number of chunks stored per collection.
    """
    path = Path(pdf_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise ValueError(f"PDF not found: {path}")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ingest(path, embed_model=embed_model)
    return buffer.getvalue().strip()


@mcp.tool()
def run_benchmark(
    k: int = 5,
    include_hybrid: bool = False,
    full_rag: bool = False,
    model: str | None = None,
    embed_models: str | None = None,
) -> dict:
    """Benchmark all chunking strategies and return per-strategy metrics.

    Wraps ``src.evaluation.run_eval.run_eval``. ``embed_models`` is an optional
    comma-separated list of embedding models to benchmark; omit it to use the
    configured ``EMBED_MODEL``. Note that ``full_rag`` (LLM-as-a-judge) can
    take several minutes and requires Ollama.
    """
    models = [m.strip() for m in embed_models.split(",") if m.strip()] if embed_models else None
    results = run_eval(
        k=k,
        include_hybrid=include_hybrid,
        full_rag=full_rag,
        model=model,
        embed_models=models,
        progress_cb=lambda key, done, total: _LOGGER.info(
            "benchmark progress %s/%s: %s", done, total, key
        ),
    )
    return {"k": k, "include_hybrid": include_hybrid, "full_rag": full_rag, "results": results}


@mcp.tool()
def list_collections() -> dict[str, int]:
    """Return the number of indexed chunks per chunking strategy."""
    return {strategy: get_collection_for_model(strategy).count() for strategy in STRATEGIES}


@mcp.tool()
def list_runs() -> list[dict]:
    """Return experiment registry run metadata, newest first."""
    with ExperimentRegistry(EXPERIMENT_DB_PATH) as registry:
        return registry.list_runs()


@mcp.tool()
def get_run(run_id: str) -> dict | None:
    """Return one run's metadata plus its strategy metrics and per-query scores."""
    with ExperimentRegistry(EXPERIMENT_DB_PATH) as registry:
        return registry.get_run(run_id)


@mcp.tool()
def delete_run(run_id: str) -> str:
    """Delete ``run_id`` and its metrics; returns a confirmation message."""
    with ExperimentRegistry(EXPERIMENT_DB_PATH) as registry:
        removed = registry.delete_run(run_id)
    if not removed:
        raise ValueError(f"No run found: {run_id}")
    return f"Deleted run {run_id}"


if __name__ == "__main__":
    mcp.run()
