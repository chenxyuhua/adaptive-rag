"""Retriever abstraction. FAISS implementation will subclass this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import RetrievedDoc


class Retriever(ABC):
    """Dense or sparse retriever — same interface."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        """Return top-k passages for a query."""
        raise NotImplementedError
