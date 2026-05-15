"""Optional linear classifier over simple features (pure Python inference; no numpy import)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..schemas import QueryRecord
from .base import RetrievalDecider
from .parsing import clamp01


def build_feature_vector(question: str, initial_answer: str | None) -> list[float]:
    q = question or ""
    a = (initial_answer or "").strip()
    lower_a = a.lower()
    return [
        float(len(q)),
        float(len(a)),
        float(len(q.split())),
        float(len(a.split())),
        float("know" in lower_a or "sure" in lower_a),
        float(any(c.isdigit() for c in a)),
    ]


class LinearClassifierRetrievalDecider(RetrievalDecider):
    """Load weights+bias from JSON (train offline with scikit-learn — see scripts)."""

    name = "linear_classifier"

    def __init__(self, weights_path: str | Path):
        path = Path(weights_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        coef = raw["coef"]
        if not isinstance(coef, list):
            raise TypeError("weights JSON 'coef' must be a list of floats")
        self.coef: list[float] = [float(c) for c in coef]
        self.intercept = float(raw.get("intercept", 0.0))
        if len(build_feature_vector("", None)) != len(self.coef):
            raise ValueError(
                f"coef length {len(self.coef)} does not match built-in feature dim "
                f"{len(build_feature_vector('', None))}"
            )

    def should_retrieve(
        self, query: QueryRecord, initial_answer: str | None
    ) -> tuple[bool, float]:
        x = build_feature_vector(query.question, initial_answer)
        logit = sum(xi * ci for xi, ci in zip(x, self.coef, strict=True)) + self.intercept
        if logit >= 0:
            z = math.exp(-logit)
            prob = 1.0 / (1.0 + z)
        else:
            z = math.exp(logit)
            prob = z / (1.0 + z)
        score = clamp01(float(prob))
        return score >= 0.5, score
