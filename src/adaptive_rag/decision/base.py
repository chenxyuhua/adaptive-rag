"""Contracts for retrieve-or-not deciders and optional reflection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..schemas import QueryRecord, RetrievedDoc


@dataclass
class ReflectOutcome:
    """Result of an optional self-reflection / critique step."""

    revised: bool = False
    revised_answer: str | None = None
    """If True and retrieval was skipped earlier, the runner may do one retrieval pass."""
    request_retrieve: bool = False
    meta: dict = field(default_factory=dict)


class RetrievalDecider(ABC):
    """Per-query policy: score whether to retrieve; optionally critique the final answer."""

    name: str = "abstract"

    @abstractmethod
    def should_retrieve(
        self, query: QueryRecord, initial_answer: str | None
    ) -> tuple[bool, float]:
        """Return (decision, score) with score in [0, 1] meaning P(need retrieval) or calibrated need.

        The runner applies `decision = score >= threshold` unless the decider forces a boolean.
        For convenience, the first return value is the decider's recommended boolean; the runner
        may still apply thresholding when `respect_decider_boolean=False` — see AdaptiveStrategy.
        """
        raise NotImplementedError

    def reflect(
        self,
        query: QueryRecord,
        answer: str,
        retrieved: list[RetrievedDoc],
    ) -> ReflectOutcome:
        """Optional critique / revise / request-retrieve. Default: no-op."""
        return ReflectOutcome(revised=False)


class AlwaysRetrieveDecider(RetrievalDecider):
    """Baseline decider: always recommend retrieval (Caroline-era stub behavior)."""

    name = "always"

    def should_retrieve(
        self, query: QueryRecord, initial_answer: str | None
    ) -> tuple[bool, float]:
        return True, 1.0


class NeverRetrieveDecider(RetrievalDecider):
    """Ablations: never retrieve (score 0)."""

    name = "never"

    def should_retrieve(
        self, query: QueryRecord, initial_answer: str | None
    ) -> tuple[bool, float]:
        return False, 0.0
