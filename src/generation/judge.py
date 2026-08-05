"""LLM-as-a-judge quality metrics for RAG answers.

Implements the core RAGAS-style metrics:
* **faithfulness** — how well the answer's claims are supported by the
  retrieved context (no hallucination).
* **answer_correctness** — factual overlap between the answer and the
  reference/expected answer.
* **answer_relevancy** — how directly the answer addresses the question.

Judges return a JSON object ``{"score": 0.0..1.0, "reason": str}``; parsing is
defensive so a slightly off-spec model response degrades to a best-effort score
instead of crashing an evaluation run.
"""

from __future__ import annotations

import json
import re

import ollama

from src.config import LLM_MODEL

_SCORE_PATTERN = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?|1\.0)', re.IGNORECASE)


def _ask_judge(prompt: str, model: str | None) -> str:
    """Send a single judge prompt and return the raw model text."""
    response = ollama.chat(
        model=model or LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


def _parse_score(raw: str) -> float:
    """Extract a 0..1 score from a judge response, tolerating malformed JSON."""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        payload = json.loads(cleaned)
        score = float(payload["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        match = _SCORE_PATTERN.search(raw)
        if match is None:
            raise ValueError(f"unparseable judge response: {raw[:200]!r}") from exc
        score = float(match.group(1))
    return max(0.0, min(1.0, score))


def _judge(prompt: str, model: str | None) -> float:
    """Run a judge prompt, parse its score, and clamp it to [0, 1]."""
    return _parse_score(_ask_judge(prompt, model))


def judge_faithfulness(
    answer: str,
    context: str,
    model: str | None = None,
) -> float:
    """Score how fully ``answer`` is supported by ``context`` (0..1)."""
    prompt = (
        "You are an impartial faithfulness judge. Below are an answer and the "
        "context it was grounded on. Score 1.0 if EVERY claim in the answer is "
        "supported by the context, 0.5 if most are, and 0.0 if the answer "
        "contradicts or invents facts not in the context.\n\n"
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        'Respond with ONLY JSON: {"score": <0.0-1.0>, "reason": "<one sentence>"}'
    )
    return _judge(prompt, model)


def judge_answer_correctness(
    answer: str,
    reference_answer: str,
    question: str,
    model: str | None = None,
) -> float:
    """Score the factual overlap between ``answer`` and ``reference_answer`` (0..1)."""
    prompt = (
        "You are an impartial answer-correctness judge. Compare the candidate "
        "answer with the reference answer for the given question. Score 1.0 for "
        "identical/full factual overlap, 0.5 for partial overlap, 0.0 for "
        "contradictory or unrelated answers.\n\n"
        f"Question:\n{question}\n\nReference answer:\n{reference_answer}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        'Respond with ONLY JSON: {"score": <0.0-1.0>, "reason": "<one sentence>"}'
    )
    return _judge(prompt, model)


def judge_answer_relevancy(
    question: str,
    answer: str,
    model: str | None = None,
) -> float:
    """Score how directly ``answer`` addresses ``question`` (0..1)."""
    prompt = (
        "You are an impartial relevancy judge. Score how directly the answer "
        "addresses the question, ignoring factual detail. 1.0 = on-topic and "
        "complete, 0.5 = partially relevant, 0.0 = off-topic.\n\n"
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
        'Respond with ONLY JSON: {"score": <0.0-1.0>, "reason": "<one sentence>"}'
    )
    return _judge(prompt, model)
