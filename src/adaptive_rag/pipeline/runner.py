"""Run a strategy over a list of queries and stream predictions to disk."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..logging import JsonlLogger
from ..schemas import PredictionRecord, QueryRecord
from ..strategies import Strategy


def config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def make_run_dir(output_root: str | Path, dataset: str, strategy_name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / dataset / strategy_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run(
    queries: list[QueryRecord],
    strategy: Strategy,
    output_root: str | Path = "runs",
    extra_meta: dict | None = None,
    progress: bool = True,
) -> Path:
    """Execute the strategy on each query and write JSONL + metadata."""
    if not queries:
        raise ValueError("No queries supplied.")

    dataset = queries[0].dataset
    run_dir = make_run_dir(output_root, dataset, strategy.name)
    pred_path = run_dir / "predictions.jsonl"
    meta_path = run_dir / "config.yaml"  # written as JSON for simplicity

    cfg = {
        "strategy": strategy.name,
        "strategy_config": strategy.config,
        "model": getattr(strategy.llm, "model", ""),
        "n": len(queries),
        "dataset": dataset,
        "extra_meta": extra_meta or {},
    }
    cfg_hash = config_hash(cfg)
    cfg["config_hash"] = cfg_hash

    iterator: Any = tqdm(queries, desc=f"{dataset}:{strategy.name}") if progress else queries

    t0 = time.monotonic()
    n_ok = 0
    n_err = 0
    err_path = run_dir / "errors.jsonl"

    with JsonlLogger(pred_path) as logger, err_path.open("a", encoding="utf-8") as err_fh:
        for q in iterator:
            try:
                pred: PredictionRecord = strategy.answer(q)
                pred.config_hash = cfg_hash
                logger.write(pred)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                n_err += 1
                err_fh.write(
                    json.dumps({"qid": q.qid, "error": repr(e)}) + "\n"
                )
        elapsed = time.monotonic() - t0
        logger.write_meta(
            str(meta_path),
            {
                **cfg,
                "elapsed_seconds": elapsed,
                "n_ok": n_ok,
                "n_err": n_err,
            },
        )

    return run_dir
