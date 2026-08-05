"""Shared data types for the retrieval layer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieval hit with its provenance and relevance score.

    `chunk_id` is the document-level primary key used for fusion across
    retrieval systems; `score` carries the system-specific relevance value.
    """

    chunk_id: str
    text: str
    page_number: int
    score: float = 0.0
