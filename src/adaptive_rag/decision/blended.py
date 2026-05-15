"""Blend multiple deciders into a single score (research ablations / ensembles)."""

from __future__ import annotations

from ..schemas import QueryRecord, RetrievedDoc
from .base import ReflectOutcome, RetrievalDecider
from .parsing import clamp01


class BlendedRetrievalDecider(RetrievalDecider):
    """Weighted average of child scores; boolean is score >= 0.5 after blend (runner may threshold)."""

    name = "blended"

    def __init__(self, parts: list[tuple[RetrievalDecider, float]]):
        if not parts:
            raise ValueError("BlendedRetrievalDecider needs at least one child decider.")
        self.parts = parts
        wsum = sum(w for _, w in parts)
        self.parts = [(d, w / wsum) for d, w in parts]

    def should_retrieve(
        self, query: QueryRecord, initial_answer: str | None
    ) -> tuple[bool, float]:
        acc = 0.0
        for dec, w in self.parts:
            _, s = dec.should_retrieve(query, initial_answer)
            acc += w * clamp01(float(s))
        score = clamp01(acc)
        return score >= 0.5, score

    def reflect(
        self,
        query: QueryRecord,
        answer: str,
        retrieved: list[RetrievedDoc],
    ) -> ReflectOutcome:
        for dec, _ in self.parts:
            out = dec.reflect(query, answer, retrieved)
            if out.revised or out.request_retrieve:
                return out
        return ReflectOutcome(revised=False)
