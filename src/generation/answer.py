"""Grounded answer generation over retrieved chunks."""

import ollama

from src.config import LLM_MODEL
from src.retrieval.types import RetrievedChunk

_SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the question using only the "
    "provided context. If the context does not contain the answer, say you could "
    "not find it. Do not use outside knowledge."
)


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Join retrieved chunks into a labelled context block for prompting."""
    return "\n\n".join(f"[source: page {chunk.page_number}] {chunk.text}" for chunk in chunks)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    model: str | None = None,
) -> str:
    """Generate an answer to ``question`` grounded in ``chunks`` via Ollama."""
    prompt = f"Context:\n{build_context(chunks)}\n\nQuestion: {question}\n\nAnswer:"
    response = ollama.chat(
        model=model or LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"].strip()


def reachable_llm(model: str | None = None) -> bool:
    """Return True if the configured Ollama LLM responds to a ping."""
    try:
        ollama.chat(
            model=model or LLM_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            options={"num_predict": 1},
        )
        return True
    except (ollama.ResponseError, ConnectionError, OSError):
        return False
