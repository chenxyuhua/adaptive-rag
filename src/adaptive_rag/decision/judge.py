"""Counterfactual judge: was retrieval useful vs the no-retrieval draft?"""

from __future__ import annotations

from ..llm.base import LLMClient
from .parsing import clamp01, extract_json_object


def heuristic_retrieval_useful(no_ret: str | None, with_ret: str | None) -> tuple[float, str]:
    """Cheap baseline label in [0,1] + discrete bucket when no LLM judge is used."""
    a = (no_ret or "").strip().lower()
    b = (with_ret or "").strip().lower()
    if not b:
        return 0.0, "unknown"
    uncertain = ("don't know" in a) or ("do not know" in a) or (len(a) < 4)
    improved = uncertain and b and ("don't know" not in b) and ("do not know" not in b)
    if improved:
        return 0.85, "useful"
    if a == b:
        return 0.2, "not_useful"
    return 0.55, "uncertain"


class LLMRetrievalUsefulnessJudge:
    """LLM-as-judge for the classification target: retrieval_useful vs not."""

    def __init__(self, llm: LLMClient, prompt_template: str, *, max_output_tokens: int = 128):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_output_tokens = max_output_tokens
        self._last_meta: dict = {}

    def score(self, question: str, no_retrieval_answer: str, with_retrieval_answer: str) -> dict:
        self._last_meta = {}
        prompt = self.prompt_template.format(
            question=question,
            no_retrieval_answer=no_retrieval_answer.strip(),
            with_retrieval_answer=with_retrieval_answer.strip(),
        )
        res = self.llm.generate(prompt, max_output_tokens=self.max_output_tokens)
        self._last_meta = {"raw": res.text, "latency_ms": res.latency_ms, "usage": res.usage}
        obj = extract_json_object(res.text) or {}
        p = obj.get("retrieval_useful_prob")
        if p is None:
            p = obj.get("probability")
        prob = 0.5
        if p is not None:
            try:
                prob = clamp01(float(p))
            except (TypeError, ValueError):
                prob = 0.5
        label = obj.get("label")
        if label not in {"useful", "not_useful", "uncertain"}:
            label = "useful" if prob >= 0.5 else "not_useful"
        return {
            "retrieval_useful_prob": prob,
            "retrieval_useful_label": label,
            "parsed": obj,
            "raw": res.text,
        }
