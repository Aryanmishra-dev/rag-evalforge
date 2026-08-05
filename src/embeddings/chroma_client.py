"""ChromaDB client wrappers for collection management and chunk storage.

Collections are namespaced by embedding model so multiple embedding models can
be benchmarked side by side without collisions:
* default model  -> ``rag_<strategy>``  (backward-compatible with existing data)
* other models   -> ``rag_<strategy>__<slugified model id>``
"""

import contextlib
import re

import chromadb
from chromadb import QueryResult
from chromadb.errors import NotFoundError

from src.config import CHROMA_DB_PATH, EMBED_MODEL
from src.embeddings.embedder import OllamaEmbeddingFunction

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_default_embedding_function = OllamaEmbeddingFunction(EMBED_MODEL)


def slugify_model(model: str) -> str:
    """Lowercase ``model`` and collapse non-alphanumeric runs into ``_``."""
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def collection_name(strategy: str, embed_model: str = EMBED_MODEL) -> str:
    """Return the Chroma collection name for ``strategy`` under ``embed_model``."""
    if embed_model == EMBED_MODEL:
        return f"rag_{strategy}"
    return f"rag_{strategy}__{slugify_model(embed_model)}"


def _embedding_function_for(model: str) -> OllamaEmbeddingFunction:
    """Return an Ollama embedding function bound to ``model``."""
    return OllamaEmbeddingFunction(model) if model != EMBED_MODEL else _default_embedding_function


def get_collection(
    name: str,
    embedding_function: OllamaEmbeddingFunction | None = None,
) -> chromadb.Collection:
    """Return the named collection, creating it with the given embedder if needed."""
    return _client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function or _default_embedding_function,
    )


def get_collection_for_model(strategy: str, embed_model: str = EMBED_MODEL) -> chromadb.Collection:
    """Return the strategy collection namespaced for ``embed_model``."""
    return get_collection(
        collection_name(strategy, embed_model),
        embedding_function=_embedding_function_for(embed_model),
    )


def reset_collection(
    name: str,
    embedding_function: OllamaEmbeddingFunction | None = None,
) -> chromadb.Collection:
    """Delete and recreate a collection, so re-ingest leaves no orphaned chunks."""
    with contextlib.suppress(NotFoundError):
        _client.delete_collection(name)
    return _client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function or _default_embedding_function,
    )


def reset_collection_for_model(
    strategy: str, embed_model: str = EMBED_MODEL
) -> chromadb.Collection:
    """Reset the strategy collection namespaced for ``embed_model``."""
    return reset_collection(
        collection_name(strategy, embed_model),
        embedding_function=_embedding_function_for(embed_model),
    )


def add_chunks(collection: chromadb.Collection, chunks: list[dict]) -> None:
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
) -> QueryResult:
    """Query the collection and return the top `n_results` matches for `query_text`."""
    return collection.query(query_texts=[query_text], n_results=n_results)
