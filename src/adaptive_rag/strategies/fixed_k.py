"""Baseline: top-k retrieval for every query."""

from __future__ import annotations

from ..schemas import LLMUsage, PredictionRecord, QueryRecord
from .base import Strategy


def format_documents(docs) -> str:
    """Format retrieved docs into a single block for the prompt."""
    if not docs:
        return "(no documents)"
    lines = []
    for i, d in enumerate(docs, start=1):
        lines.append(f"[{i}] {d.text.strip()}")
    return "\n\n".join(lines)


class FixedKRetrievalStrategy(Strategy):
    name = "fixed_k"

    def __init__(self, llm, retriever, prompt_template: str, k: int = 5):
        super().__init__(llm=llm, retriever=retriever)
        self.prompt_template = prompt_template
        self.k = k

    @property
    def config(self):
        return {"k": self.k}

    def answer(self, query: QueryRecord) -> PredictionRecord:
        if self.retriever is None:
            raise RuntimeError("FixedKRetrievalStrategy requires a retriever.")

        docs = self.retriever.retrieve(query.question, k=self.k)
        documents_block = format_documents(docs)
        prompt = self.prompt_template.format(
            question=query.question, documents=documents_block
        )
        result = self.llm.generate(prompt)
        # Approximate retrieved token count by char/4 if no tokenizer is plugged in.
        retrieved_token_count = sum(max(1, len(d.text) // 4) for d in docs)

        return PredictionRecord(
            qid=query.qid,
            dataset=query.dataset,
            question=query.question,
            gold_answers=query.gold_answers,
            strategy=self.name,
            strategy_config=self.config,
            retrieved=docs,
            retrieval_calls=1,
            retrieved_token_count=retrieved_token_count,
            prompt=prompt,
            raw_answer=result.text,
            parsed_answer=result.text.strip(),
            latency_ms=result.latency_ms,
            usage=result.usage or LLMUsage(),
            model=result.model,
        )
