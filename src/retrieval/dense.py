"""Dense retrieval adapter over a Chroma collection."""

from src.embeddings.chroma_client import query_collection
from src.evaluation.metrics import extract_pages
from src.retrieval.types import RetrievedChunk


def dense_retrieve(collection, query: str, k: int) -> list[RetrievedChunk]:
    """Return the top-``k`` chunks for ``query`` from a Chroma collection.

    Reuses the collection's registered embedding function, so behaviour is
    identical to a raw Chroma ``query`` call but surfaced as ``RetrievedChunk``
    objects with their page provenance and distance (kept as ``score``).
    """
    if collection.count() == 0:
        return []
    result = query_collection(collection, query, n_results=k)
    documents = result["documents"]
    distances = result["distances"]
    if documents is None or distances is None:
        raise RuntimeError("Chroma query returned no documents/distances")
    ids = result["ids"][0]
    documents = documents[0]
    pages = extract_pages(result)
    distances = distances[0]
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            page_number=page,
            score=float(distance) if distance is not None else 0.0,
        )
        for chunk_id, text, page, distance in zip(ids, documents, pages, distances, strict=True)
    ]
