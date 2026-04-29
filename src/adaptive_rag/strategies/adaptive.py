"""Adaptive (selective) retrieval strategy.

This file is the integration point for the adaptive retrieve-or-not policy.
The infrastructure (calling the decision function, optionally retrieving,
optionally reflecting, populating the PredictionRecord) is handled here.
The decision logic itself lives behind `RetrievalDecider` below — currently a
TODO stub. The policy owner replaces `should_retrieve` (and optionally
`reflect`) with the real implementation.

Contract for the decider:
    - `should_retrieve(query, initial_answer)` returns (decide_to_retrieve, score)
      where score is a probability/confidence in [0, 1] usable for calibration.
    - It may inspect the initial no-retrieval answer to make its decision, or
      ignore it (pure prompted-confidence variant).
"""

from __future__ import annotations

from typing import Any

from ..schemas import LLMUsage, PredictionRecord, QueryRecord
from .base import Strategy
from .fixed_k import format_documents


class RetrievalDecider:
    """TODO: replace with real adaptive retrieval policy.

    The default stub always returns (True, 1.0), which makes the adaptive
    strategy behave like fixed-k until the real policy is wired in. This
    keeps the pipeline runnable end-to-end without blocking on the policy.
    """

    def should_retrieve(
        self, query: QueryRecord, initial_answer: str | None
    ) -> tuple[bool, float]:
        # TODO: implement retrieve-or-not decision (prompted confidence,
        # learned scorer, or heuristic). Return (decision, score in [0, 1]).
        return True, 1.0

    def reflect(
        self, query: QueryRecord, answer: str, retrieved
    ) -> tuple[bool, str | None]:
        # TODO: optional self-reflection / revise step. Return
        # (needs_revision, revised_answer_or_None).
        return False, None


class AdaptiveStrategy(Strategy):
    name = "adaptive"

    def __init__(
        self,
        llm,
        retriever,
        prompt_template_no_retrieval: str,
        prompt_template_with_retrieval: str,
        decider: RetrievalDecider | None = None,
        k: int = 5,
        produce_initial_answer: bool = True,
    ):
        super().__init__(llm=llm, retriever=retriever)
        self.prompt_no_retrieval = prompt_template_no_retrieval
        self.prompt_with_retrieval = prompt_template_with_retrieval
        self.decider = decider or RetrievalDecider()
        self.k = k
        self.produce_initial_answer = produce_initial_answer

    @property
    def config(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "produce_initial_answer": self.produce_initial_answer,
            "decider": type(self.decider).__name__,
        }

    def answer(self, query: QueryRecord) -> PredictionRecord:
        total_latency = 0.0
        usage_acc = LLMUsage()
        retrieval_calls = 0
        retrieved_token_count = 0

        # Step 1: optional initial no-retrieval answer (some deciders use it).
        initial_answer: str | None = None
        initial_prompt = ""
        if self.produce_initial_answer:
            initial_prompt = self.prompt_no_retrieval.format(question=query.question)
            initial = self.llm.generate(initial_prompt)
            initial_answer = initial.text.strip()
            total_latency += initial.latency_ms
            _accumulate(usage_acc, initial.usage)

        # Step 2: ask the decider whether to retrieve.
        decide, score = self.decider.should_retrieve(query, initial_answer)

        retrieved = []
        prompt = initial_prompt
        final_text = initial_answer or ""
        model = ""

        if decide:
            if self.retriever is None:
                raise RuntimeError("AdaptiveStrategy requires a retriever to retrieve.")
            retrieved = self.retriever.retrieve(query.question, k=self.k)
            retrieval_calls = 1
            retrieved_token_count = sum(max(1, len(d.text) // 4) for d in retrieved)

            prompt = self.prompt_with_retrieval.format(
                question=query.question,
                documents=format_documents(retrieved),
            )
            second = self.llm.generate(prompt)
            final_text = second.text.strip()
            total_latency += second.latency_ms
            model = second.model
            _accumulate(usage_acc, second.usage)
        else:
            # No retrieval; final answer is the initial answer.
            if not self.produce_initial_answer:
                # Edge case: decider said no but we never produced an initial answer.
                prompt = self.prompt_no_retrieval.format(question=query.question)
                fallback = self.llm.generate(prompt)
                final_text = fallback.text.strip()
                total_latency += fallback.latency_ms
                model = fallback.model
                _accumulate(usage_acc, fallback.usage)

        # Step 3: optional reflection.
        revised, revised_text = self.decider.reflect(query, final_text, retrieved)
        if revised and revised_text is not None:
            final_text = revised_text

        return PredictionRecord(
            qid=query.qid,
            dataset=query.dataset,
            question=query.question,
            gold_answers=query.gold_answers,
            strategy=self.name,
            strategy_config=self.config,
            retrieved=retrieved,
            retrieval_calls=retrieval_calls,
            retrieved_token_count=retrieved_token_count,
            prompt=prompt,
            raw_answer=final_text,
            parsed_answer=final_text,
            latency_ms=total_latency,
            usage=usage_acc,
            model=model,
            decision={
                "decide_retrieve": decide,
                "score": score,
                "initial_answer": initial_answer,
                "revised": revised,
            },
        )


def _accumulate(acc: LLMUsage, add: LLMUsage | None) -> None:
    if add is None:
        return
    acc.prompt_tokens += add.prompt_tokens
    acc.completion_tokens += add.completion_tokens
    acc.total_tokens += add.total_tokens
