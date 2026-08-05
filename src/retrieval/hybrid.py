"""Hybrid retriever combining dense (Chroma) and BM25 lexical search via RRF."""

from collections import OrderedDict

from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import dense_retrieve
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.types import RetrievedChunk

# Bounded LRU cache of BM25Index keyed by (collection_name, corpus_size, k1, b).
# Rebuilding an index is O(T) in tokens, so we reuse it across queries on the
# same corpus — but cap entries so long-lived processes do not leak memory as
# new corpora are ingested.
_BM25_CACHE: "OrderedDict[tuple[str, int, float, float], tuple[int, BM25Index]]" = OrderedDict()
_BM25_CACHE_MAX_SIZE = 8


def _build_bm25(collection, k1: float, b: float) -> tuple[int, BM25Index]:
    """Build (or fetch from cache) a BM25 index over ``collection``'s corpus."""
    collection_name = getattr(collection, "name", "unnamed")
    corpus_size = collection.count()
    cache_key = (collection_name, corpus_size, k1, b)

    cached = _BM25_CACHE.get(cache_key)
    if cached is not None:
        _BM25_CACHE.move_to_end(cache_key)  # mark as most-recently used
        return cached

    data = collection.get(include=["documents", "metadatas"])
    index = BM25Index(k1=k1, b=b).fit(data["documents"], data["ids"])
    _BM25_CACHE[cache_key] = (corpus_size, index)
    if len(_BM25_CACHE) > _BM25_CACHE_MAX_SIZE:
        _BM25_CACHE.popitem(last=False)  # evict least-recently used
    return corpus_size, index


class HybridRetriever:
    """Fuses dense vector search with BM25 lexical search using Reciprocal Rank Fusion.

    Dense search surfaces semantically similar passages; BM25 guarantees exact
    keyword matching that embeddings can dilute. RRF blends both rank signals
    without requiring score normalization across the two systems.
    """

    def __init__(
        self,
        collection,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
    ) -> None:
        self._collection = collection
        self._rrf_k = rrf_k
        self._corpus_size, self._bm25 = _build_bm25(collection, k1, b)
        data = collection.get(include=["documents", "metadatas"])
        self._chunk_by_id: dict[str, RetrievedChunk] = {
            chunk_id: RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                page_number=metadata["page_number"],
            )
            for chunk_id, text, metadata in zip(
                data["ids"], data["documents"], data["metadatas"], strict=True
            )
        }

    @property
    def n_docs(self) -> int:
        """Number of documents in the fused corpus."""
        return self._corpus_size

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Return the top-``k`` fused chunks for ``query``.

        Each system is queried for ``k * 3`` candidates so fusion still has a
        meaningful pool even when individual systems rank the best hit low.
        """
        if self._corpus_size == 0:
            return []
        candidate_k = min(k * 3, self._corpus_size)
        dense_ids = [
            chunk.chunk_id for chunk in dense_retrieve(self._collection, query, candidate_k)
        ]
        lexical_ids = self._bm25.top(query, candidate_k)
        fused_ids = reciprocal_rank_fusion([dense_ids, lexical_ids], k=self._rrf_k)[:k]
        return [self._chunk_by_id[doc_id] for doc_id in fused_ids]
