"""Pytest fixtures: hermetic env config, deterministic embeddings, no-network guards."""

import hashlib
import os

os.environ.setdefault("LLM_MODEL", "qwen2.5:7b")
os.environ.setdefault("EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("CHROMA_DB_PATH", "./data/chroma_db")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
os.environ.setdefault("EXPERIMENT_DB_PATH", "./data/experiments.db")

import pytest

from src.retrieval.hybrid import _BM25_CACHE


def _embed_deterministic(text: str) -> list:
    """Deterministic pseudo-embedding: 16 floats derived from the text hash."""
    digest = hashlib.sha256(text.encode()).digest()
    return [(digest[i] - 127.0) / 255.0 for i in range(16)]


@pytest.fixture(autouse=True)
def _clear_bm25_cache():
    """Reset the module-level BM25 cache so tests never share indexes."""
    _BM25_CACHE.clear()
    yield


@pytest.fixture
def fake_embed(monkeypatch):
    """Replace Ollama embeddings with a deterministic local function."""
    monkeypatch.setattr("src.ingestion.chunkers.embed", _embed_deterministic)
    monkeypatch.setattr("src.retrieval.rerank.embed", _embed_deterministic)
    return _embed_deterministic


@pytest.fixture
def fake_chat(monkeypatch):
    """Replace ollama.chat with a scripted responder.

    ``fake_chat.responses`` is a deque; each call pops the next response string.
    When empty, a default JSON judge response is returned.
    """

    from collections import deque

    class FakeOllama:
        def __init__(self):
            self.responses = deque()
            self.calls = []

        def chat(self, model=None, messages=None, options=None, **kwargs):
            self.calls.append(messages)
            content = (
                self.responses.popleft()
                if self.responses
                else '{"score": 0.5, "reason": "default"}'
            )
            return {"message": {"content": content}}

    fake = FakeOllama()
    monkeypatch.setattr("src.generation.answer.ollama.chat", fake.chat)
    monkeypatch.setattr("src.generation.judge.ollama.chat", fake.chat)
    return fake
