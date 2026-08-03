"""ChromaDB client wrappers for collection management and chunk storage."""
from typing import List

import chromadb

from src.config import CHROMA_DB_PATH
from src.embeddings.embedder import OllamaEmbeddingFunction

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_embedding_function = OllamaEmbeddingFunction()


def get_collection(collection_name: str) -> chromadb.Collection:
    """Return the named collection, creating it with the Ollama embedder if needed."""
    return _client.get_or_create_collection(
        name=collection_name,
        embedding_function=_embedding_function,
    )


def reset_collection(collection_name: str) -> chromadb.Collection:
    """Delete and recreate a collection, so re-ingest leaves no orphaned chunks."""
    try:
        _client.delete_collection(collection_name)
    except chromadb.errors.NotFoundError:
        pass
    return _client.get_or_create_collection(
        name=collection_name,
        embedding_function=_embedding_function,
    )


def add_chunks(collection: chromadb.Collection, chunks: List[dict]) -> None:
    """Upsert chunk records into the given collection."""
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "page_number": c["page_number"],
                "strategy": c["strategy"],
            }
            for c in chunks
        ],
    )


def query_collection(
    collection: chromadb.Collection, query_text: str, n_results: int = 10
) -> dict:
    """Query the collection and return the top `n_results` matches for `query_text`."""
    return collection.query(query_texts=[query_text], n_results=n_results)
