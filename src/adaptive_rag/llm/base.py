"""LLM client abstraction. Any provider must implement LLMClient.generate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..schemas import LLMUsage


@dataclass
class GenerationResult:
    text: str
    usage: LLMUsage
    latency_ms: float
    model: str


class LLMClient(ABC):
    """Provider-agnostic generation interface."""

    model: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Run a single completion. Must populate text, usage, latency_ms, model."""
        raise NotImplementedError
