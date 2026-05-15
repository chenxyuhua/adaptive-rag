#!/usr/bin/env python3
"""Run a baseline strategy over a dataset sample.

Examples
--------
# No retrieval, 50 NQ questions
python scripts/run_baseline.py --dataset nq --strategy no_retrieval --n 50

# Fixed-k=5 retrieval over TriviaQA
python scripts/run_baseline.py --dataset triviaqa --strategy fixed_k --k 5 --n 50

# Adaptive (prompt decider + threshold; add --reflection for critique loop)
python scripts/run_baseline.py --dataset nq --strategy adaptive --n 50 --decision-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when run as a script.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from adaptive_rag import data as data_module  # noqa: E402
from adaptive_rag.eval.metrics import aggregate  # noqa: E402
from adaptive_rag.eval.loader import iter_predictions  # noqa: E402
from adaptive_rag.llm import build_llm  # noqa: E402
from adaptive_rag.pipeline import run as run_pipeline  # noqa: E402
from adaptive_rag.retriever import build_retriever  # noqa: E402
from adaptive_rag.decision import build_decider, maybe_wrap_reflection  # noqa: E402
from adaptive_rag.strategies import (  # noqa: E402
    AdaptiveStrategy,
    FixedKRetrievalStrategy,
    NoRetrievalStrategy,
)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_prompts(cfg: dict) -> tuple[str, str]:
    no_path = _REPO / cfg["prompts"]["no_retrieval"]
    with_path = _REPO / cfg["prompts"]["with_retrieval"]
    return no_path.read_text(encoding="utf-8"), with_path.read_text(encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(_REPO / "configs" / "default.yaml"))
    p.add_argument("--dataset", required=True, choices=["nq", "triviaqa", "hotpotqa"])
    p.add_argument(
        "--strategy",
        required=True,
        choices=["no_retrieval", "fixed_k", "adaptive"],
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n", type=int, default=None, help="Sample size; default = config dev_sample_size")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-root", default=None)
    # Adaptive-only overrides (see configs/default.yaml adaptive.*)
    p.add_argument(
        "--adaptive-decider",
        default=None,
        help="Decider kind: always, never, heuristic, prompt, blended, linear_classifier",
    )
    p.add_argument("--decision-threshold", type=float, default=None)
    p.add_argument(
        "--use-decider-hint-only",
        action="store_true",
        help="Ignore continuous threshold; use the model's boolean need_retrieval hint only.",
    )
    p.add_argument("--reflection", action="store_true", help="Enable self-reflection / critique step.")
    p.add_argument(
        "--shadow-retrieval-for-log",
        action="store_true",
        help="When skipping retrieval, still run a shadow RAG pass for counterfactual logging only.",
    )
    p.add_argument(
        "--counterfactual-judge",
        default=None,
        choices=["heuristic", "llm", "none"],
        help="How to label retrieval usefulness when both answers exist.",
    )
    args = p.parse_args()

    load_dotenv(_REPO / ".env")
    cfg = load_config(args.config)
    no_prompt, with_prompt = load_prompts(cfg)

    n = args.n or cfg["run"]["dev_sample_size"]
    seed = args.seed or cfg["run"]["seed"]
    output_root = args.output_root or cfg["run"]["output_root"]

    # Dataset
    print(f"[run] Loading dataset {args.dataset} (n={n})")
    queries = data_module.load(args.dataset, n=n, seed=seed)
    print(f"[run] Got {len(queries)} queries")

    # LLM
    llm_cfg = cfg["llm"]
    llm = build_llm(
        provider=llm_cfg["provider"],
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        max_output_tokens=llm_cfg["max_output_tokens"],
        requests_per_minute=llm_cfg["requests_per_minute"],
    )

    # Retriever (only if needed)
    needs_retriever = args.strategy in {"fixed_k", "adaptive"}
    retriever = None
    if needs_retriever:
        retriever = build_retriever(
            kind="faiss",
            index_path=str(_REPO / cfg["retriever"]["index_path"]),
            passages_path=str(_REPO / cfg["retriever"]["passages_path"]),
            embedder_model=cfg["embedder"]["model"],
        )

    # Strategy
    if args.strategy == "no_retrieval":
        strategy = NoRetrievalStrategy(llm=llm, prompt_template=no_prompt)
    elif args.strategy == "fixed_k":
        strategy = FixedKRetrievalStrategy(
            llm=llm, retriever=retriever, prompt_template=with_prompt, k=args.k
        )
    else:
        ad_cfg = cfg.get("adaptive") or {}
        dec_kind = args.adaptive_decider or ad_cfg.get("decider_kind", "prompt")
        dec_sub = dict(ad_cfg.get("decider") or {})
        if "prompt_path" not in dec_sub and dec_kind == "prompt":
            dec_sub["prompt_path"] = (cfg.get("prompts") or {}).get(
                "retrieve_decision", "prompts/retrieve_decision.txt"
            )
        decider = build_decider(dec_kind, llm, repo_root=_REPO, cfg=dec_sub)
        reflect_on = args.reflection or bool(ad_cfg.get("enable_reflection"))
        refl_sub = dict(ad_cfg.get("reflection") or {})
        if "prompt_path" not in refl_sub:
            refl_sub["prompt_path"] = (cfg.get("prompts") or {}).get(
                "reflect_critique", "prompts/reflect_critique.txt"
            )
        decider = maybe_wrap_reflection(
            decider,
            llm,
            repo_root=_REPO,
            enabled=reflect_on,
            cfg=refl_sub,
        )
        thr = args.decision_threshold
        if thr is None:
            thr = float(ad_cfg.get("decision_threshold", 0.5))
        if args.use_decider_hint_only:
            apply_thr = False
        else:
            apply_thr = bool(ad_cfg.get("apply_decision_threshold", True))

        shadow = args.shadow_retrieval_for_log or bool(ad_cfg.get("shadow_retrieval_for_log"))
        cf_judge = args.counterfactual_judge or ad_cfg.get("counterfactual_judge", "heuristic")
        cf_prompt = ad_cfg.get("counterfactual_prompt_path") or (cfg.get("prompts") or {}).get(
            "retrieval_useful_judge", "prompts/retrieval_useful_judge.txt"
        )

        strategy = AdaptiveStrategy(
            llm=llm,
            retriever=retriever,
            prompt_template_no_retrieval=no_prompt,
            prompt_template_with_retrieval=with_prompt,
            decider=decider,
            k=args.k,
            decision_threshold=thr,
            apply_decision_threshold=apply_thr,
            shadow_retrieval_for_log=shadow,
            counterfactual_judge=cf_judge,
            counterfactual_prompt_path=cf_prompt,
            repo_root=_REPO,
        )

    # Run
    run_dir = run_pipeline(
        queries=queries,
        strategy=strategy,
        output_root=output_root,
        extra_meta={"dataset": args.dataset, "n": n, "seed": seed},
    )

    # Quick sanity metrics
    metrics = aggregate(iter_predictions(run_dir))
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[run] Done. Run dir: {run_dir}")
    print(f"[run] Sanity metrics: {metrics}")


if __name__ == "__main__":
    main()
