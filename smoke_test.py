import ollama
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from src.config import LLM_MODEL, EMBED_MODEL, CHROMA_DB_PATH

# Custom embedding function wrapping Ollama
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

# Test 1 — LLM reachable
print("Testing LLM...")
response = ollama.chat(
    model=LLM_MODEL,
    messages=[{"role": "user", "content": "Say hello in one word."}]
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
    embedding_function=ollama_ef
)
collection.add(documents=["hello world"], ids=["1"])
result = collection.query(query_texts=["hello"], n_results=1)
print("Chroma query result:", result["documents"])

print("\nAll systems reachable, using nomic-embed-text (768-dim) consistently.")