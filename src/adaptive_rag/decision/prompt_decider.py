"""LLM prompt: estimate whether external retrieval is needed before answering."""

from __future__ import annotations

from ..llm.base import LLMClient
from ..schemas import QueryRecord
from .base import RetrievalDecider
from .parsing import clamp01, extract_json_object


class PromptRetrievalDecider(RetrievalDecider):
    """Prompt-based P(need retrieval) with optional boolean hint in JSON."""

    name = "prompt"

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str,
        *,
        max_output_tokens: int = 128,
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_output_tokens = max_output_tokens
        self._last_meta: dict = {}

    def should_retrieve(
        self, query: QueryRecord, initial_answer: str | None
    ) -> tuple[bool, float]:
        self._last_meta = {}
        prompt = self.prompt_template.format(
            question=query.question,
            initial_answer=(initial_answer or "").strip() or "(no draft yet)",
        )
        res = self.llm.generate(prompt, max_output_tokens=self.max_output_tokens)
        self._last_meta = {"raw": res.text, "latency_ms": res.latency_ms, "usage": res.usage}

        obj = extract_json_object(res.text) or {}
        # Accept several key spellings from messy model output.
        need = obj.get("need_retrieval")
        if need is None:
            need = obj.get("retrieve")
        if need is None:
            need = obj.get("need_retrieve")
        conf = obj.get("confidence")
        if conf is None:
            conf = obj.get("probability")
        if conf is None:
            conf = obj.get("score")

        score = 0.5
        if conf is not None:
            try:
                score = clamp01(float(conf))
            except (TypeError, ValueError):
                score = 0.5
        if need is True:
            score = max(score, 0.75)
        elif need is False:
            score = min(score, 0.35)
        hint = bool(need) if isinstance(need, bool) else score >= 0.5
        self._last_meta["parsed"] = {"need_retrieval_hint": hint, "confidence": score}
        return hint, score
