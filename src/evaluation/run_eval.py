"""CLI and API to benchmark chunking strategies and advanced retrieval pipelines.

Two evaluation modes:
* **retrieval-only** (default): hit_rate@k and MRR per strategy.
* **full RAG** (``--rag``): additionally generates answers and scores them with
  an LLM-as-a-judge on faithfulness, answer correctness, and answer relevancy.

``--hybrid`` adds BM25+dense fusion (and a re-ranked variant) to the benchmark.
``--embed-models`` benchmarks the same strategies under multiple embedding
models; strategy results are keyed ``strategy@model`` when a non-default model
is used. Every run is persisted to the experiment registry (SQLite) and as a
JSON file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    EMBED_MODEL,
    EXPERIMENT_DB_PATH,
    JUDGE_MODEL,
    LLM_MODEL,
    RERANKER_MODEL,
)
from src.embeddings.chroma_client import (
    get_collection_for_model,
    slugify_model,
)
from src.evaluation.rag_eval import (
    Retriever,
    dataset_sha256,
    evaluate_rag,
    evaluate_retrieval,
    load_pairs,
    reranked_retriever,
)
from src.experiment.registry import ExperimentRegistry
from src.retrieval.dense import dense_retrieve
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import build_reranker

STRATEGIES = ["fixed", "recursive", "sentence", "semantic"]
TEST_PAIRS_PATH = Path("src/evaluation/test_qa_pairs.json")
RESULTS_DIR = Path("data/eval_results")


class _RetrieverSpec(TypedDict):
    """A single retriever plus its corpus size for one strategy/model."""

    retriever: Retriever
    n_chunks: int


def _dense_retriever(collection) -> Retriever:
    """Return a dense retriever bound to ``collection``."""

    def _retrieve(query: str, k: int):
        return dense_retrieve(collection, query, k)

    return _retrieve


def _hybrid_retriever(collection):
    """Build a hybrid (BM25 + dense, RRF-fused) retriever bound to ``collection``."""
    hybrid = HybridRetriever(collection)
    return hybrid.retrieve, hybrid


def metric_key(strategy: str, embed_model: str) -> str:
    """Namespace a strategy's results by model when it differs from the default."""
    if embed_model == EMBED_MODEL:
        return strategy
    return f"{strategy}@{slugify_model(embed_model)}"


def build_retrievers(embed_model: str, include_hybrid: bool) -> dict[str, _RetrieverSpec]:
    """Return ``{key: {"retriever": Retriever, "n_chunks": int}}`` for one model."""
    retrievers: dict[str, _RetrieverSpec] = {}
    for strategy in STRATEGIES:
        collection = get_collection_for_model(strategy, embed_model)
        retrievers[metric_key(strategy, embed_model)] = {
            "retriever": _dense_retriever(collection),
            "n_chunks": collection.count(),
        }

    if include_hybrid:
        collection = get_collection_for_model(STRATEGIES[0], embed_model)
        hybrid_retriever, hybrid = _hybrid_retriever(collection)
        model_tag = "" if embed_model == EMBED_MODEL else f"@{slugify_model(embed_model)}"

        retrievers[f"hybrid{model_tag}"] = {
            "retriever": hybrid_retriever,
            "n_chunks": hybrid.n_docs,
        }

        reranker = build_reranker(RERANKER_MODEL or None)
        retrievers[f"hybrid_rerank{model_tag}"] = {
            "retriever": reranked_retriever(hybrid_retriever, reranker),
            "n_chunks": hybrid.n_docs,
        }
    return retrievers


def run_eval(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    k: int = 5,
    include_hybrid: bool = False,
    full_rag: bool = False,
    model: str | None = None,
    embed_models: list[str] | None = None,
    pairs: list[dict] | None = None,
    progress_cb: Callable[[str, int, int], None] | None = None,
) -> dict[str, dict[str, float]]:
    """Benchmark all retrievers and return ``{key: {metric: value}}``.

    ``embed_models`` defaults to the configured ``EMBED_MODEL``; passing more
    models evaluates each strategy's collections under every model.
    ``progress_cb(message, done, total)`` is invoked after each retriever
    finishes so callers (e.g. the UI) can render live progress.
    """
    dataset = pairs if pairs is not None else load_pairs(TEST_PAIRS_PATH)
    models = embed_models or [EMBED_MODEL]
    total_steps = sum(len(build_retrievers(embed_model, include_hybrid)) for embed_model in models)
    done = 0

    results: dict[str, dict[str, float]] = {}
    for embed_model in models:
        retrievers = build_retrievers(embed_model, include_hybrid)
        for key, spec in retrievers.items():
            retriever = spec["retriever"]
            if full_rag:
                metrics = evaluate_rag(retriever, dataset, k, model=model)
            else:
                metrics = evaluate_retrieval(retriever, dataset, k)
            metrics["n_chunks"] = spec["n_chunks"]
            results[key] = metrics
            done += 1
            if progress_cb is not None:
                progress_cb(key, done, total_steps)
    return results


def _pretty_print(results: dict[str, dict[str, float]], full_rag: bool) -> None:
    """Render the benchmark table to stdout."""
    header = f"{'key':<24}{'hit@k':>8}{'MRR':>8}"
    if full_rag:
        header += f"{'faith':>8}{'correct':>9}{'rel':>8}"
    print(f"\n{header}")
    print("-" * len(header))
    for key, metrics in results.items():
        row = f"{key:<24}{metrics['avg_hit_rate']:>8.3f}{metrics['avg_mrr']:>8.3f}"
        if full_rag:
            row += (
                f"{metrics['avg_faithfulness']:>8.3f}"
                f"{metrics['avg_answer_correctness']:>9.3f}"
                f"{metrics['avg_answer_relevancy']:>8.3f}"
            )
        print(row)


def main() -> None:
    """Run the benchmark from the command line and persist the results."""
    parser = argparse.ArgumentParser(
        description="Run RAG chunking-strategy and retrieval-pipeline benchmark."
    )
    parser.add_argument("--k", type=int, default=5, help="Retrieved chunks per query.")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Also benchmark BM25+dense fusion and its re-ranked variant.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Run full RAG evaluation: generate answers and judge them.",
    )
    parser.add_argument(
        "--model", default=None, help="Override the LLM used for generation/judging."
    )
    parser.add_argument(
        "--embed-models",
        default=None,
        help="Comma-separated embedding models to benchmark (default: EMBED_MODEL). "
        "Ingest each model first with `python src/ingestion/ingest.py --embed-model <m>`.",
    )
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be >= 1")

    embed_models = (
        [m.strip() for m in args.embed_models.split(",") if m.strip()]
        if args.embed_models
        else None
    )
    if embed_models is not None and not embed_models:
        parser.error("--embed-models requires at least one model")

    results = run_eval(
        k=args.k,
        include_hybrid=args.hybrid,
        full_rag=args.rag,
        model=args.model,
        embed_models=embed_models,
    )
    _pretty_print(results, full_rag=args.rag)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_results_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")

    params = {
        "k": args.k,
        "hybrid": args.hybrid,
        "full_rag": args.rag,
        "model": args.model,
        "embed_models": embed_models or [EMBED_MODEL],
    }
    with ExperimentRegistry(EXPERIMENT_DB_PATH) as registry:
        run_id = registry.start_run(
            config={
                "llm_model": LLM_MODEL,
                "judge_model": JUDGE_MODEL,
                "reranker_model": RERANKER_MODEL,
                "embed_model": EMBED_MODEL,
            },
            params=params,
            dataset_hash=dataset_sha256(TEST_PAIRS_PATH),
            notes=f"results also saved to {out_path.name}",
        )
        for key, metrics in results.items():
            registry.log_metrics(run_id, key, metrics)
    print(f"Registered run {run_id} in {EXPERIMENT_DB_PATH}")


if __name__ == "__main__":
    main()
