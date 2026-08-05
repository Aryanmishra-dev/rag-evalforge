"""End-to-end RAG evaluation: retrieval scoring plus generation quality judging.

A *retriever* is any callable ``(query, k) -> List[RetrievedChunk]``. Dense,
hybrid, and reranked retrievers all satisfy this contract, so a single driver
evaluates every strategy uniformly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path

from src.evaluation.metrics import hit_rate_at_k, reciprocal_rank
from src.generation.answer import generate_answer
from src.generation.judge import (
    judge_answer_correctness,
    judge_answer_relevancy,
    judge_faithfulness,
)
from src.retrieval.types import RetrievedChunk

Retriever = Callable[[str, int], list[RetrievedChunk]]

_LOGGER = logging.getLogger(__name__)


def reranked_retriever(retriever: Retriever, reranker, candidate_multiplier: int = 3):
    """Wrap ``retriever`` so its output is re-ranked down to ``k``.

    Retrieves a wider candidate pool (``k * candidate_multiplier``) so the
    re-ranker has enough material to promote a well-scoring but low-ranked hit.
    """

    def _retrieve(query: str, k: int) -> list[RetrievedChunk]:
        candidates = retriever(query, k * candidate_multiplier)
        return reranker.rerank(query, candidates, k)

    return _retrieve


def pages_of(chunks: list[RetrievedChunk]) -> list[int]:
    """Project retrieved chunks onto their 1-indexed page numbers."""
    return [chunk.page_number for chunk in chunks]


def evaluate_retrieval(
    retriever: Retriever,
    pairs: list[dict],
    k: int,
) -> dict[str, float]:
    """Return average hit_rate@k and MRR of ``retriever`` over ``pairs``."""
    if not pairs:
        return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}
    hits = []
    ranks = []
    for pair in pairs:
        pages = pages_of(retriever(pair["question"], k))
        hits.append(hit_rate_at_k(pages, pair["page_number"]))
        ranks.append(reciprocal_rank(pages, pair["page_number"]))
    return {
        "avg_hit_rate": sum(hits) / len(hits),
        "avg_mrr": sum(ranks) / len(ranks),
    }


def evaluate_rag(
    retriever: Retriever,
    pairs: list[dict],
    k: int,
    model: str | None = None,
) -> dict[str, float]:
    """Evaluate retrieval + generation + judging over ``pairs``.

    Returns the retrieval metrics together with mean faithfulness, answer
    correctness, and answer relevancy. Generation failures on individual
    queries are scored 0.0 and recorded so runs never abort mid-way.
    """
    totals = {
        "hit_rate": 0.0,
        "mrr": 0.0,
        "faithfulness": 0.0,
        "answer_correctness": 0.0,
        "answer_relevancy": 0.0,
    }
    n = len(pairs)
    for pair in pairs:
        chunks = retriever(pair["question"], k)
        pages = pages_of(chunks)
        totals["hit_rate"] += hit_rate_at_k(pages, pair["page_number"])
        totals["mrr"] += reciprocal_rank(pages, pair["page_number"])
        try:
            answer = generate_answer(pair["question"], chunks, model)
            context = "\n\n".join(f"[page {c.page_number}] {c.text}" for c in chunks)
            totals["faithfulness"] += judge_faithfulness(answer, context, model)
            totals["answer_correctness"] += judge_answer_correctness(
                answer, pair["expected_answer"], pair["question"], model
            )
            totals["answer_relevancy"] += judge_answer_relevancy(pair["question"], answer, model)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Network/LLM flakiness on a single query must not abort the run.
            _LOGGER.warning("generation failed for query %r: %s", pair.get("id"), exc)
    return {
        "avg_hit_rate": totals["hit_rate"] / n,
        "avg_mrr": totals["mrr"] / n,
        "avg_faithfulness": totals["faithfulness"] / n,
        "avg_answer_correctness": totals["answer_correctness"] / n,
        "avg_answer_relevancy": totals["answer_relevancy"] / n,
    }


def dataset_sha256(pairs_path: Path) -> str:
    """Return the first 12 hex chars of the SHA-256 of ``pairs_path`` content."""
    return hashlib.sha256(pairs_path.read_bytes()).hexdigest()[:12]


def load_pairs(pairs_path: Path) -> list[dict]:
    """Load and validate the QA pairs dataset."""
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    required = {"question", "expected_answer", "page_number"}
    for pair in pairs:
        missing = required - pair.keys()
        if missing:
            raise ValueError(f"QA pair missing fields {sorted(missing)}: {pair!r}")
    return pairs
