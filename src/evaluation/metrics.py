"""Retrieval quality metrics: hit_rate@k, reciprocal rank, and page extraction."""
from typing import List

def hit_rate_at_k(retrieved_pages: List[int], expected_page: int) -> int:
    """Return 1 if any retrieved chunk's page matches the expected page, else 0."""
    return 1 if expected_page in retrieved_pages else 0


def reciprocal_rank(retrieved_pages: List[int], expected_page: int) -> float:
    """Return 1 / (1-indexed rank of first page match), or 0 if not found in top-k."""
    for rank, page in enumerate(retrieved_pages, start=1):
        if page == expected_page:
            return 1.0 / rank
    return 0.0


def extract_pages(chroma_result: dict) -> List[int]:
    """Flatten a Chroma query result into an ordered list of page_numbers."""
    return [meta["page_number"] for meta in chroma_result["metadatas"][0]]
