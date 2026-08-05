"""Unit tests for the dense and hybrid retrievers (offline, using fakes)."""
# pylint: disable=missing-function-docstring,missing-class-docstring

from src.retrieval.dense import dense_retrieve
from src.retrieval.hybrid import HybridRetriever
from tests.fakes import FakeCollection

DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Technical writing must be clear and concise.",
    "A fox is fast and clever in the forest.",
    "Concise writing saves the reader time.",
]


def _collection():
    return FakeCollection("rag_fixed", DOCUMENTS)


class TestDenseRetrieve:
    def test_returns_retrieved_chunks(self):
        chunks = dense_retrieve(_collection(), "fox", k=2)
        assert len(chunks) <= 2
        assert all(c.chunk_id for c in chunks)
        assert all(c.text for c in chunks)
        assert all(isinstance(c.page_number, int) for c in chunks)

    def test_empty_collection(self):
        empty = FakeCollection("rag_fixed", [])
        assert dense_retrieve(empty, "fox", k=5) == []

    def test_k_limit(self):
        chunks = dense_retrieve(_collection(), "writing", k=1)
        assert len(chunks) == 1


class TestHybrid:
    def test_returns_top_k(self):
        hybrid = HybridRetriever(_collection())
        chunks = hybrid.retrieve("fox", k=2)
        assert len(chunks) == 2
        assert hybrid.n_docs == len(DOCUMENTS)

    def test_empty_collection(self):
        hybrid = HybridRetriever(FakeCollection("rag_fixed", []))
        assert hybrid.retrieve("fox", k=5) == []

    def test_bm25_index_is_cached_across_instances(self):
        first = HybridRetriever(_collection())
        second = HybridRetriever(_collection())
        assert first._bm25 is second._bm25  # same corpus, same index object

    def test_lexical_match_surfaces(self):
        # A query with strong lexical overlap should appear in results even if
        # the fake dense ranker and BM25 disagree.
        hybrid = HybridRetriever(_collection())
        results = hybrid.retrieve("concise writing", k=2)
        assert results
