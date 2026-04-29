"""Strategy abstraction.

All strategies (no-retrieval, fixed-k, adaptive) implement the same `answer`
method so the runner is strategy-agnostic. Strategies are responsible for
deciding what to retrieve (if anything) and prompting the LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..llm.base import LLMClient
from ..retriever.base import Retriever
from ..schemas import PredictionRecord, QueryRecord


class Strategy(ABC):
    """Stateless per-query policy that produces a PredictionRecord."""

    name: str = "abstract"

    def __init__(self, llm: LLMClient, retriever: Retriever | None = None):
        self.llm = llm
        self.retriever = retriever

    @abstractmethod
    def answer(self, query: QueryRecord) -> PredictionRecord:
        """Produce a populated PredictionRecord (retrieval + generation as needed)."""
        raise NotImplementedError

    @property
    def config(self) -> dict[str, Any]:
        """Override to expose strategy-specific config in the prediction log."""
        return {}
