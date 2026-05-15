"""Adaptive (selective) retrieval strategy + integration for retrieval deciders."""

from __future__ import annotations

from typing import Any

from ..decision import ReflectOutcome, RetrievalDecider
from ..decision.judge import LLMRetrievalUsefulnessJudge, heuristic_retrieval_useful
from ..schemas import LLMUsage, PredictionRecord, QueryRecord, RetrievedDoc
from .base import Strategy
from .fixed_k import format_documents


def _accumulate(acc: LLMUsage, add: LLMUsage | None) -> None:
    if add is None:
        return
    acc.prompt_tokens += add.prompt_tokens
    acc.completion_tokens += add.completion_tokens
    acc.total_tokens += add.total_tokens


def _usage_from_sidecar(obj: Any) -> LLMUsage | None:
    meta = getattr(obj, "_last_meta", None) or {}
    u = meta.get("usage")
    return u if isinstance(u, LLMUsage) else None


def _merge_sidecar(dst: dict, obj: Any, prefix: str) -> None:
    meta = getattr(obj, "_last_meta", None)
    if isinstance(meta, dict) and meta:
        dst[prefix] = {k: v for k, v in meta.items() if k != "usage"}


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
        *,
        decision_threshold: float = 0.5,
        apply_decision_threshold: bool = True,
        shadow_retrieval_for_log: bool = False,
        counterfactual_judge: str | None = "heuristic",
        counterfactual_prompt_path: str | None = None,
        repo_root: Any | None = None,
    ):
        super().__init__(llm=llm, retriever=retriever)
        self.prompt_no_retrieval = prompt_template_no_retrieval
        self.prompt_with_retrieval = prompt_template_with_retrieval
        self.decider = decider or _default_always_decider()
        self.k = k
        self.produce_initial_answer = produce_initial_answer
        self.decision_threshold = float(decision_threshold)
        self.apply_decision_threshold = bool(apply_decision_threshold)
        self.shadow_retrieval_for_log = bool(shadow_retrieval_for_log)
        self.counterfactual_judge = counterfactual_judge
        self.counterfactual_prompt_path = counterfactual_prompt_path
        self.repo_root = repo_root

    @property
    def config(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "produce_initial_answer": self.produce_initial_answer,
            "decider": getattr(self.decider, "name", type(self.decider).__name__),
            "decision_threshold": self.decision_threshold,
            "apply_decision_threshold": self.apply_decision_threshold,
            "shadow_retrieval_for_log": self.shadow_retrieval_for_log,
            "counterfactual_judge": self.counterfactual_judge,
        }

    def answer(self, query: QueryRecord) -> PredictionRecord:
        total_latency = 0.0
        usage_acc = LLMUsage()
        retrieval_calls = 0
        retrieved_token_count = 0

        decision_trace: dict[str, Any] = {}

        # Step 1: optional no-retrieval draft
        initial_answer: str | None = None
        initial_prompt = ""
        model = ""
        if self.produce_initial_answer:
            initial_prompt = self.prompt_no_retrieval.format(question=query.question)
            initial = self.llm.generate(initial_prompt)
            initial_answer = initial.text.strip()
            total_latency += initial.latency_ms
            model = initial.model
            _accumulate(usage_acc, initial.usage)

        # Step 2: retrieve-or-not
        hint, score = self.decider.should_retrieve(query, initial_answer)
        _merge_sidecar(decision_trace, self.decider, "decider_raw")
        inner = getattr(self.decider, "inner", None)
        if inner is not None:
            _merge_sidecar(decision_trace, inner, "decider_inner")
        _accumulate(usage_acc, _usage_from_sidecar(self.decider))
        _accumulate(usage_acc, _usage_from_sidecar(inner))

        if self.apply_decision_threshold:
            decide = float(score) >= self.decision_threshold
        else:
            decide = bool(hint)

        retrieved: list[RetrievedDoc] = []
        prompt = initial_prompt
        final_text = initial_answer or ""
        with_retrieval_answer: str | None = None
        shadow_docs: list[RetrievedDoc] = []
        shadow_with_answer: str | None = None

        def do_retrieve_generate(
            question: str,
        ) -> tuple[list[RetrievedDoc], str, str, float, LLMUsage | None, str]:
            if self.retriever is None:
                raise RuntimeError("AdaptiveStrategy requires a retriever to retrieve.")
            docs = self.retriever.retrieve(question, k=self.k)
            p2 = self.prompt_with_retrieval.format(
                question=question,
                documents=format_documents(docs),
            )
            second = self.llm.generate(p2)
            return docs, second.text.strip(), p2, second.latency_ms, second.usage, second.model

        if decide:
            docs, ans, p2, lat, use, m2 = do_retrieve_generate(query.question)
            retrieved = docs
            with_retrieval_answer = ans
            prompt = p2
            final_text = ans
            total_latency += lat
            model = m2
            _accumulate(usage_acc, use)
            retrieval_calls = 1
            retrieved_token_count = sum(max(1, len(d.text) // 4) for d in retrieved)
        else:
            if not self.produce_initial_answer:
                prompt = self.prompt_no_retrieval.format(question=query.question)
                fallback = self.llm.generate(prompt)
                final_text = fallback.text.strip()
                total_latency += fallback.latency_ms
                model = fallback.model
                _accumulate(usage_acc, fallback.usage)

            if self.shadow_retrieval_for_log and self.retriever is not None:
                sdocs, sans, _, lat, use, _m = do_retrieve_generate(query.question)
                shadow_docs = sdocs
                shadow_with_answer = sans
                total_latency += lat
                _accumulate(usage_acc, use)
                retrieval_calls += 1
                retrieved_token_count += sum(max(1, len(d.text) // 4) for d in sdocs)

        # Step 3: reflection (revise, or request one retrieval pass if we skipped RAG)
        outcome = self.decider.reflect(query, final_text, retrieved)
        ref_mod = getattr(self.decider, "reflection", None)
        if ref_mod is not None:
            _merge_sidecar(decision_trace, ref_mod, "reflect")

        if outcome.request_retrieve and not decide and self.retriever is not None:
            docs, ans, p2, lat, use, m2 = do_retrieve_generate(query.question)
            retrieved = docs
            with_retrieval_answer = ans
            prompt = p2
            final_text = ans
            decide = True
            total_latency += lat
            model = m2
            _accumulate(usage_acc, use)
            retrieval_calls += 1
            retrieved_token_count += sum(max(1, len(d.text) // 4) for d in retrieved)
            outcome = ReflectOutcome(revised=False)

        if outcome.revised and outcome.revised_answer is not None:
            final_text = outcome.revised_answer

        _accumulate(usage_acc, _usage_from_sidecar(getattr(self.decider, "reflection", None)))

        # Step 4: counterfactual logging (classification target + analysis)
        counterfactual: dict[str, Any] = {
            "no_retrieval_answer": initial_answer,
            "with_retrieval_answer": with_retrieval_answer,
            "shadow_with_retrieval_answer": shadow_with_answer,
            "shadow_retrieved_doc_ids": [d.doc_id for d in shadow_docs],
        }

        no_ans = initial_answer
        with_ans = with_retrieval_answer or shadow_with_answer
        judge_blob: dict[str, Any] = {}
        judge_mode = (self.counterfactual_judge or "none").strip().lower()
        if no_ans is not None and with_ans and judge_mode not in {"", "none"}:
            if judge_mode == "llm" and self.counterfactual_prompt_path and self.repo_root:
                text = (self.repo_root / self.counterfactual_prompt_path).read_text(encoding="utf-8")
                judge = LLMRetrievalUsefulnessJudge(self.llm, text)
                judge_blob = judge.score(query.question, no_ans, with_ans)
                total_latency += float(judge._last_meta.get("latency_ms", 0) or 0)
                _accumulate(usage_acc, _usage_from_sidecar(judge))
                counterfactual.update(
                    {
                        "retrieval_useful_prob": judge_blob.get("retrieval_useful_prob"),
                        "retrieval_useful_label": judge_blob.get("retrieval_useful_label"),
                        "judge": "llm",
                    }
                )
            elif judge_mode == "heuristic":
                prob, label = heuristic_retrieval_useful(no_ans, with_ans)
                counterfactual.update(
                    {
                        "retrieval_useful_prob": prob,
                        "retrieval_useful_label": label,
                        "judge": "heuristic",
                    }
                )

        decision_trace.update(
            {
                "decide_retrieve": decide,
                "score": float(score),
                "hint": bool(hint),
                "threshold": self.decision_threshold,
                "apply_threshold": self.apply_decision_threshold,
                "initial_answer": initial_answer,
                "reflection": {
                    "revised": outcome.revised,
                    "request_retrieve": outcome.request_retrieve,
                    "meta": outcome.meta,
                },
                "counterfactual": counterfactual,
            }
        )

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
            decision=decision_trace,
        )


def _default_always_decider() -> RetrievalDecider:
    from ..decision import AlwaysRetrieveDecider

    return AlwaysRetrieveDecider()
