"""Unit tests for the four chunking strategies."""
# pylint: disable=missing-function-docstring,missing-class-docstring

import itertools

import pytest

from src.ingestion.chunkers import (
    _cosine,
    _split_sentences,
    chunk_fixed,
    chunk_recursive,
    chunk_semantic,
    chunk_sentence,
)

PAGES = [
    {"page_number": 1, "text": "The quick brown fox jumps over the lazy dog. " * 30},
    {"page_number": 2, "text": "Technical writing is clear and concise. " * 40},
]


def _texts(chunks):
    return [c["text"] for c in chunks]


def _chunk_ids(chunks):
    return [c["chunk_id"] for c in chunks]


class TestFixed:
    def test_splits_into_chunks_of_max_size(self):
        chunks = chunk_fixed(PAGES, doc_id="d", chunk_size=100, overlap=0)
        assert len(chunks) >= 2
        assert all(len(c["text"]) <= 100 for c in chunks)
        assert all(c["strategy"] == "fixed" for c in chunks)

    def test_ids_encode_provenance(self):
        chunks = chunk_fixed(PAGES, doc_id="d", chunk_size=100, overlap=0)
        assert all(c["chunk_id"].startswith("d_fixed_") for c in chunks)

    def test_page_numbers_preserved(self):
        chunks = chunk_fixed(PAGES, doc_id="d", chunk_size=100, overlap=0)
        pages = {c["page_number"] for c in chunks}
        assert pages == {1, 2}

    def test_overlap_carries_tail(self):
        page = {"page_number": 1, "text": "word " * 100}
        chunks = chunk_fixed([page], doc_id="d", chunk_size=100, overlap=20)
        texts = _texts(chunks)
        for prev, curr in itertools.pairwise(texts):
            assert prev[-20:] in curr

    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError):
            chunk_fixed(PAGES, doc_id="d", chunk_size=50, overlap=50)

    def test_short_pages_are_skipped(self):
        pages = [{"page_number": 1, "text": "short"}]
        assert not chunk_fixed(pages, doc_id="d", chunk_size=100)


class TestRecursive:
    def test_chunks_merge_across_separators(self):
        text = ("Paragraph one. " * 20) + "\n\n" + ("Paragraph two. " * 20)
        chunks = chunk_recursive([{"page_number": 1, "text": text}], doc_id="d")
        assert chunks
        assert all(c["strategy"] == "recursive" for c in chunks)

    def test_respects_chunk_size_cap(self):
        chunks = chunk_recursive(PAGES, doc_id="d", chunk_size=120, overlap=0)
        assert all(len(c["text"]) <= 120 for c in chunks)

    def test_ids_unique_within_page(self):
        chunks = chunk_recursive(PAGES, doc_id="d")
        ids = _chunk_ids(chunks)
        assert len(ids) == len(set(ids))


class TestSentence:
    def test_splits_on_sentence_boundaries(self):
        text = "First sentence. Second sentence! Third sentence? " * 10
        chunks = chunk_sentence([{"page_number": 1, "text": text}], doc_id="d", chunk_size=200)
        assert chunks
        assert all(c["strategy"] == "sentence" for c in chunks)
        assert all(c["page_number"] == 1 for c in chunks)

    def test_sentences_are_preserved(self):
        text = "Alpha beta. Gamma delta."
        chunks = chunk_sentence([{"page_number": 1, "text": text}], doc_id="d", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Alpha beta. Gamma delta."


class TestSemantic:
    def test_chunks_pages(self, fake_embed):
        assert callable(fake_embed)
        chunks = chunk_semantic(PAGES, doc_id="d", chunk_size=500, threshold=0.7)
        assert chunks
        assert all(c["strategy"] == "semantic" for c in chunks)

    def test_size_cap_enforced(self, fake_embed):
        assert callable(fake_embed)
        text = "Sentence about topic one. " * 60
        pages = [{"page_number": 1, "text": text}]
        chunks = chunk_semantic(pages, doc_id="d", chunk_size=120, overlap=0, threshold=0.0)
        assert all(len(c["text"]) <= 130 for c in chunks)

    def test_higher_threshold_splits_more(self, fake_embed):
        assert callable(fake_embed)
        text = " ".join(f"Topic A sentence {i}." for i in range(20))
        pages = [{"page_number": 1, "text": text}]
        low = chunk_semantic(pages, doc_id="d", threshold=0.1)
        high = chunk_semantic(pages, doc_id="d", threshold=0.95)
        assert len(high) > len(low)


class TestHelpers:
    def test_split_sentences_handles_abbreviations(self):
        assert _split_sentences("Dr. Smith wrote. He is good.") == [
            "Dr. Smith wrote.",
            "He is good.",
        ]

    def test_cosine_identical_vectors_is_one(self):
        v = [1.0, 0.0, 3.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors_is_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
