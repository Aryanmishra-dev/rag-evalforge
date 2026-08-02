import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings
from src.config import EMBED_MODEL


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            embeddings.append(embed(text))
        return embeddings


def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]