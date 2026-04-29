"""Baseline: answer with no external retrieval."""

from __future__ import annotations

from ..schemas import LLMUsage, PredictionRecord, QueryRecord
from .base import Strategy


class NoRetrievalStrategy(Strategy):
    name = "no_retrieval"

    def __init__(self, llm, prompt_template: str):
        super().__init__(llm=llm, retriever=None)
        self.prompt_template = prompt_template

    def answer(self, query: QueryRecord) -> PredictionRecord:
        prompt = self.prompt_template.format(question=query.question)
        result = self.llm.generate(prompt)
        return PredictionRecord(
            qid=query.qid,
            dataset=query.dataset,
            question=query.question,
            gold_answers=query.gold_answers,
            strategy=self.name,
            strategy_config=self.config,
            retrieved=[],
            retrieval_calls=0,
            retrieved_token_count=0,
            prompt=prompt,
            raw_answer=result.text,
            parsed_answer=result.text.strip(),
            latency_ms=result.latency_ms,
            usage=result.usage or LLMUsage(),
            model=result.model,
        )
