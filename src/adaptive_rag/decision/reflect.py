"""Self-reflection: critique the answer; optionally revise or request retrieval."""

from __future__ import annotations

from ..llm.base import LLMClient
from ..schemas import QueryRecord, RetrievedDoc
from .base import ReflectOutcome, RetrievalDecider
from .parsing import extract_json_object


class ReflectionMixin(RetrievalDecider):
    """Attach reflection to an inner decider (composition over inheritance)."""

    def __init__(self, inner: RetrievalDecider, reflection: "LLMReflection"):
        self.inner = inner
        self.reflection = reflection
        self.name = f"{getattr(inner, 'name', type(inner).__name__)}+reflect"

    def should_retrieve(
        self, query: QueryRecord, initial_answer: str | None
    ) -> tuple[bool, float]:
        return self.inner.should_retrieve(query, initial_answer)

    def reflect(
        self,
        query: QueryRecord,
        answer: str,
        retrieved: list[RetrievedDoc],
    ) -> ReflectOutcome:
        return self.reflection.reflect(query, answer, retrieved)


class LLMReflection:
    """Critique + optional one-shot revision or request for retrieval."""

    def __init__(self, llm: LLMClient, prompt_template: str, *, max_output_tokens: int = 256):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_output_tokens = max_output_tokens
        self._last_meta: dict = {}

    def reflect(
        self,
        query: QueryRecord,
        answer: str,
        retrieved: list[RetrievedDoc],
    ) -> ReflectOutcome:
        self._last_meta = {}
        has_docs = bool(retrieved)
        prompt = self.prompt_template.format(
            question=query.question,
            answer=answer.strip(),
            has_documents="yes" if has_docs else "no",
        )
        res = self.llm.generate(prompt, max_output_tokens=self.max_output_tokens)
        self._last_meta = {"raw": res.text, "latency_ms": res.latency_ms, "usage": res.usage}

        obj = extract_json_object(res.text) or {}
        satisfied = bool(obj.get("satisfied", True))
        need_external = bool(obj.get("need_external_evidence") or obj.get("need_retrieval"))
        action = (obj.get("action") or "none").strip().lower()
        revised_answer = obj.get("revised_answer") or obj.get("answer")

        if satisfied:
            return ReflectOutcome(revised=False, meta={"parsed": obj})

        if not has_docs and need_external:
            return ReflectOutcome(
                revised=False,
                request_retrieve=True,
                meta={"parsed": obj, "raw": res.text},
            )

        if action == "revise" and isinstance(revised_answer, str) and revised_answer.strip():
            return ReflectOutcome(
                revised=True,
                revised_answer=revised_answer.strip(),
                meta={"parsed": obj},
            )

        return ReflectOutcome(revised=False, meta={"parsed": obj, "raw": res.text})
