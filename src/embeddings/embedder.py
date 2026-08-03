"""Ollama-backed embedding function and helper for ChromaDB."""
from typing import List

import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings

from src.config import EMBED_MODEL


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function that delegates to the local Ollama server."""
    # pylint: disable=too-few-public-methods

    def __call__(self, texts: Documents) -> Embeddings:
        """Embed every input document and return the resulting vectors."""
        return [embed(text) for text in texts]


def embed(text: str) -> List[float]:
    """Return the Ollama embedding vector for a single text string."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]
