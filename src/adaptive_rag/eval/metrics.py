"""Stub answer-accuracy metrics.

Provides Exact Match and token-F1 in the SQuAD/NQ-Open style. This is the
*sanity check* for baselines, not the team's full evaluation. The eval owner
extends this module with grounding, calibration, and decision-quality metrics
on top of the same `PredictionRecord` schema.
"""

from __future__ import annotations

import re
import string
from collections import Counter

# TODO: extend with grounding, hallucination detection, calibration (ECE),
# retrieval-decision precision/recall/F1, and failure-mode analysis.


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_TBL = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles, strip punctuation, collapse whitespace."""
    if text is None:
        return ""
    s = text.lower()
    s = s.translate(_PUNCT_TBL)
    s = _ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    p = normalize(prediction)
    return float(any(p == normalize(g) for g in golds))


def f1(prediction: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    p_toks = normalize(prediction).split()
    best = 0.0
    for g in golds:
        g_toks = normalize(g).split()
        if not p_toks or not g_toks:
            best = max(best, 1.0 if p_toks == g_toks else 0.0)
            continue
        common = Counter(p_toks) & Counter(g_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(p_toks)
        recall = num_same / len(g_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def aggregate(records) -> dict[str, float]:
    """Compute mean EM/F1 over an iterable of PredictionRecord."""
    n = 0
    em_sum = 0.0
    f1_sum = 0.0
    for r in records:
        n += 1
        em_sum += exact_match(r.parsed_answer, r.gold_answers)
        f1_sum += f1(r.parsed_answer, r.gold_answers)
    if n == 0:
        return {"n": 0, "em": 0.0, "f1": 0.0}
    return {"n": n, "em": em_sum / n, "f1": f1_sum / n}
