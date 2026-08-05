"""Smoke test verifying that Ollama (LLM + embeddings) and ChromaDB are reachable."""

import chromadb
import ollama

from src.config import CHROMA_DB_PATH, EMBED_MODEL, LLM_MODEL
from src.embeddings.embedder import OllamaEmbeddingFunction

# Test 1 — LLM reachable
print("Testing LLM...")
response = ollama.chat(
    model=LLM_MODEL,
    messages=[{"role": "user", "content": "Say hello in one word."}],
)
print("LLM response:", response["message"]["content"])

# Test 2 — Embeddings reachable
print("\nTesting embeddings...")
embed_response = ollama.embeddings(model=EMBED_MODEL, prompt="test sentence")
vector = embed_response["embedding"]
print("Embedding dimension:", len(vector))

# Test 3 — ChromaDB read/write, now using Ollama embeddings explicitly
print("\nTesting ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
ollama_ef = OllamaEmbeddingFunction()

collection = client.get_or_create_collection(
    name="smoke_test",
    embedding_function=ollama_ef,
)
collection.add(documents=["hello world"], ids=["1"])
result = collection.query(query_texts=["hello"], n_results=1)
print("Chroma query result:", result["documents"])

print("\nAll systems reachable, using nomic-embed-text (768-dim) consistently.")
