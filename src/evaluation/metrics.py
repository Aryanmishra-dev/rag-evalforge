"""Retrieval quality metrics: hit_rate@k, reciprocal rank, and page extraction."""

from typing import cast

from chromadb import QueryResult


def hit_rate_at_k(retrieved_pages: list[int], expected_page: int) -> int:
    """Return 1 if any retrieved chunk's page matches the expected page, else 0."""
    return 1 if expected_page in retrieved_pages else 0


def reciprocal_rank(retrieved_pages: list[int], expected_page: int) -> float:
    """Return 1 / (1-indexed rank of first page match), or 0 if not found in top-k."""
    for rank, page in enumerate(retrieved_pages, start=1):
        if page == expected_page:
            return 1.0 / rank
    return 0.0


def extract_pages(chroma_result: QueryResult) -> list[int]:
    """Flatten a Chroma query result into an ordered list of page_numbers."""
    metadatas = chroma_result["metadatas"]
    if metadatas is None:
        raise ValueError("Chroma query returned no metadata")
    return [cast(int, meta["page_number"]) for meta in metadatas[0]]
