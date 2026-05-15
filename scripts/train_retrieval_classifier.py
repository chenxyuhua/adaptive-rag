#!/usr/bin/env python3
"""Fit a tiny logistic decider on tabular features and export JSON for LinearClassifierRetrievalDecider.

Expects a CSV with columns: question, initial_answer, label (0/1).
Label 1 means "retrieval was useful" for training the router (e.g. from oracle EM comparisons).

Example
-------
python scripts/train_retrieval_classifier.py --csv data/decision_labels.csv --out configs/decider_weights.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from adaptive_rag.decision.classifier import build_feature_vector  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install dev extras: pip install -e '.[dev]' (needs scikit-learn).") from e

    xs = []
    ys = []
    with args.csv.open(newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            q = row.get("question") or ""
            a = row.get("initial_answer")
            y = int(row.get("label", "0"))
            xs.append(build_feature_vector(q, a))
            ys.append(y)
    if not xs:
        raise SystemExit("No rows in CSV.")
    X = np.asarray(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.int64)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)
    coef = clf.coef_.reshape(-1).tolist()
    intercept = float(clf.intercept_[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"coef": coef, "intercept": intercept}, indent=2), encoding="utf-8")
    print(f"[train] wrote {args.out} (n={len(ys)}, dim={len(coef)})")


if __name__ == "__main__":
    main()
