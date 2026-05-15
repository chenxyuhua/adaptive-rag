from __future__ import annotations

from adaptive_rag.decision import HeuristicRetrievalDecider, PromptRetrievalDecider
from adaptive_rag.llm.base import GenerationResult, LLMClient
from adaptive_rag.schemas import LLMUsage, QueryRecord, RetrievedDoc
from adaptive_rag.strategies.adaptive import AdaptiveStrategy


class FakeLLM(LLMClient):
    model = "fake"

    def __init__(self, replies: list[str]):
        self._replies = list(replies)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        if not self._replies:
            raise RuntimeError("FakeLLM out of scripted replies")
        text = self._replies.pop(0)
        return GenerationResult(
            text=text,
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_ms=1.0,
            model=self.model,
        )


class FakeRetriever:
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        return [
            RetrievedDoc(doc_id="d1", score=1.0, text="Capital of France is Paris.", source="wiki")
        ]


def _q() -> QueryRecord:
    return QueryRecord(
        qid="1",
        dataset="nq",
        question="What is the capital of France?",
        gold_answers=["Paris"],
    )


def test_adaptive_skips_retrieval_when_prompt_score_low():
    llm = FakeLLM(
        [
            "Paris",
            '{"need_retrieval": false, "confidence": 0.1}',
        ]
    )
    decider = PromptRetrievalDecider(llm, "{question}\n{initial_answer}")
    strat = AdaptiveStrategy(
        llm=llm,
        retriever=FakeRetriever(),
        prompt_template_no_retrieval="{question}",
        prompt_template_with_retrieval="Docs:\n{documents}\nQ:{question}",
        decider=decider,
        k=1,
        decision_threshold=0.5,
        counterfactual_judge="none",
    )
    pred = strat.answer(_q())
    assert pred.retrieval_calls == 0
    assert pred.parsed_answer == "Paris"
    assert pred.decision is not None
    assert pred.decision["decide_retrieve"] is False
    assert pred.decision["score"] == 0.1


def test_adaptive_retrieves_on_heuristic_uncertainty():
    llm = FakeLLM(
        [
            "I don't know.",
            "Paris",
        ]
    )
    strat = AdaptiveStrategy(
        llm=llm,
        retriever=FakeRetriever(),
        prompt_template_no_retrieval="{question}",
        prompt_template_with_retrieval="Docs:\n{documents}\nQ:{question}",
        decider=HeuristicRetrievalDecider(),
        k=1,
        decision_threshold=0.5,
        counterfactual_judge="heuristic",
    )
    pred = strat.answer(_q())
    assert pred.retrieval_calls >= 1
    assert "Paris" in pred.parsed_answer
    cf = pred.decision["counterfactual"]
    assert cf["no_retrieval_answer"] == "I don't know."
    assert cf.get("retrieval_useful_label") in {"useful", "not_useful", "uncertain"}


def test_linear_classifier_weights(tmp_path):
    import json

    from adaptive_rag.decision import LinearClassifierRetrievalDecider

    w = tmp_path / "w.json"
    # 6-dim feature vector; large positive weight on length -> retrieve for long answers
    w.write_text(
        json.dumps({"coef": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "intercept": 5.0}),
        encoding="utf-8",
    )
    dec = LinearClassifierRetrievalDecider(w)
    q = _q()
    need, score = dec.should_retrieve(q, "hi")
    assert need is True
    assert score > 0.9
