#!/usr/bin/env python3
"""Walk runs/ and produce a single summary JSON across all baseline runs.

Output schema (one row per run dir):
  {
    "dataset": "nq", "strategy": "fixed_k", "k": 5, "n": 50,
    "em": 0.32, "f1": 0.41,
    "avg_latency_ms": 412.3, "avg_prompt_tokens": 980,
    "avg_retrieved_tokens": 950, "retrieval_call_rate": 1.0,
    "run_dir": "runs/nq/fixed_k/20260501T..."
  }

Writes:
  - runs/summary.json   (machine-readable, list of rows above)
  - runs/summary.md     (a small Markdown table for the README/handoff)

Usage:
  python scripts/summarize_baselines.py [--root runs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from adaptive_rag.eval import aggregate, iter_predictions  # noqa: E402


def find_run_dirs(root: Path) -> list[Path]:
    """A run dir contains predictions.jsonl."""
    return sorted(p.parent for p in root.rglob("predictions.jsonl"))


def _rel(p: Path) -> str:
    """Best-effort relative path against the repo root, fall back to str."""
    try:
        return str(p.resolve().relative_to(_REPO))
    except ValueError:
        return str(p)


def summarize_run(run_dir: Path) -> dict:
    preds = list(iter_predictions(run_dir))
    if not preds:
        return {"run_dir": str(run_dir), "n": 0, "skipped": True}

    metrics = aggregate(preds)
    latencies = [p.latency_ms for p in preds]
    prompt_toks = [p.usage.prompt_tokens for p in preds]
    retr_toks = [p.retrieved_token_count for p in preds]
    retrieval_rate = sum(1 for p in preds if p.retrieval_calls > 0) / len(preds)

    first = preds[0]
    return {
        "dataset": first.dataset,
        "strategy": first.strategy,
        "strategy_config": first.strategy_config,
        "model": first.model,
        "n": metrics["n"],
        "em": round(metrics["em"], 4),
        "f1": round(metrics["f1"], 4),
        "avg_latency_ms": round(mean(latencies), 1),
        "avg_prompt_tokens": round(mean(prompt_toks), 1),
        "avg_retrieved_tokens": round(mean(retr_toks), 1),
        "retrieval_call_rate": round(retrieval_rate, 3),
        "run_dir": _rel(run_dir),
    }


def to_markdown_table(rows: list[dict]) -> str:
    headers = [
        "dataset",
        "strategy",
        "n",
        "EM",
        "F1",
        "retrieval rate",
        "avg latency (ms)",
        "avg prompt tok",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        if r.get("skipped"):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("dataset", "")),
                    str(r.get("strategy", "")),
                    str(r.get("n", "")),
                    f"{r.get('em', 0):.3f}",
                    f"{r.get('f1', 0):.3f}",
                    f"{r.get('retrieval_call_rate', 0):.2f}",
                    f"{r.get('avg_latency_ms', 0):.0f}",
                    f"{r.get('avg_prompt_tokens', 0):.0f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"No runs dir at {root}")
        return

    run_dirs = find_run_dirs(root)
    print(f"[summary] Found {len(run_dirs)} runs under {root}")

    rows = [summarize_run(rd) for rd in run_dirs]
    rows = [r for r in rows if not r.get("skipped")]
    rows.sort(key=lambda r: (r["dataset"], r["strategy"], r["run_dir"]))

    out_json = root / "summary.json"
    out_md = root / "summary.md"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    out_md.write_text(
        "# Baseline summary\n\n"
        f"Aggregated over {len(rows)} runs.\n\n"
        + to_markdown_table(rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"[summary] Wrote {out_json}")
    print(f"[summary] Wrote {out_md}")
    print()
    print(to_markdown_table(rows))


if __name__ == "__main__":
    main()
