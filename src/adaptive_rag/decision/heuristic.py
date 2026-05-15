"""Heuristic need-retrieval score from the no-retrieval draft answer."""

from __future__ import annotations

import re

from ..schemas import QueryRecord
from .base import RetrievalDecider

_UNCERTAIN_PAT = re.compile(
    r"\b(i don't know|i do not know|not sure|cannot answer|can't answer|unclear|"
    r"no information|insufficient)\b",
    re.I,
)


class HeuristicRetrievalDecider(RetrievalDecider):
    """Fast baseline: high score (need retrieval) when the draft looks uncertain or empty."""

    name = "heuristic"

    def __init__(self, uncertain_boost: float = 0.85, confident_drop: float = 0.15):
        self.uncertain_boost = uncertain_boost
        self.confident_drop = confident_drop

    def should_retrieve(
        self, query: QueryRecord, initial_answer: str | None
    ) -> tuple[bool, float]:
        text = (initial_answer or "").strip()
        if not text:
            return True, 1.0
        if _UNCERTAIN_PAT.search(text):
            return True, self.uncertain_boost
        # Short factual-ish answers get a lower retrieval urgency.
        if len(text) < 8:
            return True, 0.7
        return False, self.confident_drop
