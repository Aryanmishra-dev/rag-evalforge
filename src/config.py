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

LLM_MODEL = _require("LLM_MODEL")
EMBED_MODEL = _require("EMBED_MODEL")
CHROMA_DB_PATH = _require("CHROMA_DB_PATH")
OLLAMA_HOST = _require("OLLAMA_HOST")
