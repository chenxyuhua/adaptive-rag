"""Standard data schemas for adaptive-rag.

These are the contract between the pipeline (this package) and the rest of the
team's code (adaptive policy, evaluation, analysis). Every run emits one
PredictionRecord per query in the same shape, regardless of strategy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRecord(BaseModel):
    """One question, normalized across datasets."""

    qid: str
    dataset: Literal["nq", "triviaqa", "hotpotqa", "custom"]
    question: str
    gold_answers: list[str]
    meta: dict[str, Any] = Field(default_factory=dict)


class RetrievedDoc(BaseModel):
    """A single retrieved passage."""

    doc_id: str
    score: float
    text: str
    source: str = "wiki"
    meta: dict[str, Any] = Field(default_factory=dict)


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class PredictionRecord(BaseModel):
    """One row of the standard JSONL output. Same shape across all strategies."""

    qid: str
    dataset: str
    question: str
    gold_answers: list[str]

    strategy: str
    strategy_config: dict[str, Any] = Field(default_factory=dict)

    retrieved: list[RetrievedDoc] = Field(default_factory=list)
    retrieval_calls: int = 0
    retrieved_token_count: int = 0

    prompt: str = ""
    raw_answer: str = ""
    parsed_answer: str = ""

    latency_ms: float = 0.0
    usage: LLMUsage = Field(default_factory=LLMUsage)

    model: str = ""
    config_hash: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # Optional: filled in by the adaptive strategy when it runs.
    # Kept on the record so evaluation can analyze decision quality directly.
    decision: dict[str, Any] | None = None
