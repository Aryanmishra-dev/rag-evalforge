"""Unit tests for model-aware collection naming."""

# pylint: disable=missing-function-docstring,missing-class-docstring
from src.config import EMBED_MODEL
from src.embeddings.chroma_client import collection_name, slugify_model


class TestSlugifyModel:
    def test_keeps_simple_names(self):
        assert slugify_model("nomic-embed-text") == "nomic_embed_text"

    def test_collapses_special_characters(self):
        assert slugify_model("mxbai-embed-large:latest") == "mxbai_embed_large_latest"
        assert slugify_model("my model v2!!") == "my_model_v2"

    def test_lowercases(self):
        assert slugify_model("All-MiniLM") == "all_minilm"


class TestCollectionName:
    def test_default_model_untagged(self):
        assert collection_name("fixed") == "rag_fixed"
        assert collection_name("fixed", EMBED_MODEL) == "rag_fixed"

    def test_other_model_tagged(self):
        assert collection_name("fixed", "mxbai-embed-large:latest") == (
            "rag_fixed__mxbai_embed_large_latest"
        )

    def test_strategy_preserved(self):
        assert collection_name("semantic", "other:1").startswith("rag_semantic")
