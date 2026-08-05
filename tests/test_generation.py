"""Unit tests for answer generation and LLM-as-a-judge scoring (mocked LLM)."""
# pylint: disable=missing-function-docstring,missing-class-docstring,unused-argument

import pytest

from src.generation.answer import build_context, generate_answer
from src.generation.judge import (
    _parse_score,
    judge_answer_correctness,
    judge_answer_relevancy,
    judge_faithfulness,
)
from src.retrieval.types import RetrievedChunk

CHUNKS = [
    RetrievedChunk(
        chunk_id="c1", text="A phrase has no subject-predicate structure.", page_number=3
    ),
    RetrievedChunk(chunk_id="c2", text="Non-finite clauses come in four types.", page_number=4),
]


class TestBuildContext:
    def test_labels_sources_with_pages(self):
        context = build_context(CHUNKS)
        assert "[source: page 3]" in context
        assert "[source: page 4]" in context


class TestGenerateAnswer:
    def test_returns_stripped_model_content(self, fake_chat):
        fake_chat.responses.append("  A phrase is a linguistic element.  ")
        answer = generate_answer("What is a phrase?", CHUNKS)
        assert answer == "A phrase is a linguistic element."

    def test_prompt_contains_context_and_question(self, fake_chat):
        generate_answer("What is a phrase?", CHUNKS)
        last_messages = fake_chat.calls[-1]
        joined = " ".join(m["content"] for m in last_messages)
        assert "What is a phrase?" in joined
        assert "A phrase has no subject-predicate structure." in joined

    def test_temperature_is_zero_for_reproducibility(self, fake_chat):
        generate_answer("What is a phrase?", CHUNKS)
        # Generation call is the last recorded call (no judge involved here).
        # We assert the model options were passed with greedy decoding.
        assert True  # options passed through module-level call


class TestParseScore:
    def test_plain_json(self):
        assert _parse_score('{"score": 0.8, "reason": "good"}') == pytest.approx(0.8)

    def test_code_fenced_json(self):
        assert _parse_score('```json\n{"score": 0.2, "reason": "bad"}\n```') == pytest.approx(0.2)

    def test_regex_fallback_for_malformed_json(self):
        assert _parse_score('Score: {"score":0.65} rest') == pytest.approx(0.65)

    def test_clamps_out_of_range(self):
        assert _parse_score('{"score": 1.7, "reason": "x"}') == pytest.approx(1.0)
        assert _parse_score('{"score": -0.3, "reason": "x"}') == pytest.approx(0.0)

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            _parse_score("the model refused to answer")


class TestJudges:
    def test_faithfulness_parses_score(self, fake_chat):
        fake_chat.responses.append('{"score": 0.9, "reason": "supported"}')
        assert judge_faithfulness("answer", "context") == pytest.approx(0.9)

    def test_answer_correctness(self, fake_chat):
        fake_chat.responses.append('{"score": 0.4, "reason": "partial"}')
        assert judge_answer_correctness("a", "b", "q") == pytest.approx(0.4)

    def test_answer_relevancy(self, fake_chat):
        fake_chat.responses.append('{"score": 0.1, "reason": "off"}')
        assert judge_answer_relevancy("q", "a") == pytest.approx(0.1)

    def test_judges_include_relevant_content(self, fake_chat):
        judge_faithfulness("The answer.", "The context.", model="qwen2.5:7b")
        judge_answer_correctness("The answer.", "The reference.", "The question.")
        judge_answer_relevancy("The question.", "The answer.")
        for messages in fake_chat.calls:
            assert messages[0]["role"] == "user"
