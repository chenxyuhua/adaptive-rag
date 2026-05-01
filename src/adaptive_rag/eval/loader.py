"""Load prediction JSONL files into typed PredictionRecord objects.

This is the front door for downstream evaluation, calibration, and analysis.
Read predictions from any run directory and operate on Pydantic objects rather
than re-parsing JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from ..schemas import PredictionRecord


def load_predictions(path: str | Path) -> list[PredictionRecord]:
    """Eager load — fine for ≤100k predictions."""
    return list(iter_predictions(path))


def iter_predictions(path: str | Path) -> Iterator[PredictionRecord]:
    """Streaming load — use for large run dirs."""
    p = Path(path)
    if p.is_dir():
        p = p / "predictions.jsonl"
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield PredictionRecord.model_validate_json(line)


def join_runs(*paths: str | Path) -> dict[str, dict[str, PredictionRecord]]:
    """Index predictions by qid across multiple runs (e.g., one per strategy).

    Returns: {qid: {strategy_name: PredictionRecord}}. Useful for the
    counterfactual evaluation: for each query, compare outputs under
    no-retrieval / fixed-k / adaptive side by side.
    """
    out: dict[str, dict[str, PredictionRecord]] = {}
    for p in paths:
        for rec in iter_predictions(p):
            out.setdefault(rec.qid, {})[rec.strategy] = rec
    return out


def filter_by(records: Iterable[PredictionRecord], **kv) -> list[PredictionRecord]:
    """Tiny filter helper. Example: filter_by(recs, dataset='nq')."""
    out = []
    for r in records:
        if all(getattr(r, k, None) == v for k, v in kv.items()):
            out.append(r)
    return out
