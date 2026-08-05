"""Ollama-backed embedding function and helper for ChromaDB."""

from typing import cast

import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings

from src.config import EMBED_MODEL


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function that delegates to the local Ollama server."""

    def __init__(self, model: str = EMBED_MODEL) -> None:
        # Deliberately do NOT call super().__init__(): Chroma's base raises a
        # DeprecationWarning for any class that runs its default constructor.
        self.model = model

    def __call__(self, texts: Documents) -> Embeddings:
        """Embed every input document and return the resulting vectors."""
        return cast(Embeddings, [embed(text, model=self.model) for text in texts])


def embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Return the Ollama embedding vector for a single text string."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]
