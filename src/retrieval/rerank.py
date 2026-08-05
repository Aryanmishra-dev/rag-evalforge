"""Re-ranking of retrieval candidates: embedding-similarity and cross-encoder.

A cross-encoder re-ranks by scoring the ``(query, chunk)`` pair jointly, which
is more accurate than dot-product recall but too slow to run over the whole
corpus — hence it is applied to a small candidate pool.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.embeddings.embedder import embed
from src.retrieval.types import RetrievedChunk

try:  # optional heavyweight dependency
    from sentence_transformers import CrossEncoder as _CrossEncoder  # type: ignore
except ImportError:  # pragma: no cover - exercised only when optional dep absent
    _CrossEncoder = None


class Reranker(Protocol):
    """Anything that reorders ``chunks`` for ``query`` and keeps the top ``k``."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Return the top ``top_k`` chunks reordered best-first."""


def _cosine_scores(query: np.ndarray, chunk_vectors: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between one query vector and chunk vectors."""
    norms = np.linalg.norm(chunk_vectors, axis=1)
    norms[norms == 0.0] = 1.0  # avoid division by zero on degenerate vectors
    return chunk_vectors @ query / norms


class EmbeddingReranker:
    """Reranks candidates by cosine similarity of their Ollama embeddings.

    Reuses the existing ``nomic-embed-text`` embeddings so no extra models are
    required. Complexity per call: ``O(n)`` embedding requests for ``n``
    candidates plus ``O(n)`` numpy work.
    """

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Return the top ``top_k`` chunks ranked by embedding cosine similarity."""
        if not chunks or top_k <= 0:
            return chunks[:top_k]
        query_vector = np.asarray(embed(query), dtype=np.float64)
        vectors = np.asarray([embed(chunk.text) for chunk in chunks], dtype=np.float64)
        similarities = _cosine_scores(query_vector, vectors)
        order = np.argsort(-similarities, kind="stable")
        return [
            RetrievedChunk(
                chunk_id=chunks[idx].chunk_id,
                text=chunks[idx].text,
                page_number=chunks[idx].page_number,
                score=float(similarities[idx]),
            )
            for idx in order[:top_k]
        ]


class CrossEncoderReranker:
    """Reranks with a sentence-transformers cross-encoder when available.

    Falls back to :class:`EmbeddingReranker` if ``sentence_transformers`` is
    not installed, keeping the codebase runnable on a minimal environment.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = _CrossEncoder(model_name) if _CrossEncoder is not None else None
        self._fallback = EmbeddingReranker()

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Return the top ``top_k`` chunks re-ranked by the cross-encoder."""
        if self._model is None:  # pragma: no cover
            return self._fallback.rerank(query, chunks, top_k)
        if not chunks or top_k <= 0:
            return chunks[:top_k]
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [
            RetrievedChunk(
                chunk_id=chunks[idx].chunk_id,
                text=chunks[idx].text,
                page_number=chunks[idx].page_number,
                score=float(scores[idx]),
            )
            for idx in order[:top_k]
        ]


def build_reranker(model_name: str | None = None) -> Reranker:
    """Return the configured reranker.

    Defaults to :class:`EmbeddingReranker`; pass ``model_name`` to attempt a
    cross-encoder (falling back to embeddings if it cannot be loaded).
    """
    if model_name:
        return CrossEncoderReranker(model_name)
    return EmbeddingReranker()
