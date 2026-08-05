"""Environment configuration loaded from .env at import time."""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Return the env var `key`, raising ValueError if it is unset or empty."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required env var: {key}")
    return value


def _optional(key: str, default: str) -> str:
    """Return the env var `key` or `default` when unset/empty."""
    return os.getenv(key) or default


LLM_MODEL = _require("LLM_MODEL")
EMBED_MODEL = _require("EMBED_MODEL")
CHROMA_DB_PATH = _require("CHROMA_DB_PATH")
OLLAMA_HOST = _require("OLLAMA_HOST")

# Judge model used for LLM-as-a-judge scoring; defaults to the generation model.
JUDGE_MODEL = _optional("JUDGE_MODEL", LLM_MODEL)

# Optional cross-encoder re-ranker. Leave empty to use the embedding re-ranker.
RERANKER_MODEL = _optional("RERANKER_MODEL", "")

# Experiment registry database.
EXPERIMENT_DB_PATH = _optional("EXPERIMENT_DB_PATH", "./data/experiments.db")

# BM25 / RRF hyper-parameters.
BM25_K1 = float(_optional("BM25_K1", "1.5"))
BM25_B = float(_optional("BM25_B", "0.75"))
RRF_K = int(_optional("RRF_K", "60"))
