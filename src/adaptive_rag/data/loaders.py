"""Dataset loaders that normalize NQ, TriviaQA, HotpotQA into QueryRecord.

All loaders return a list[QueryRecord]. Use load_dataset(name, n=...) for a
sized dev sample (deterministic via seed). HuggingFace `datasets` is the only
dependency — no per-dataset URL plumbing.
"""

from __future__ import annotations

import random
from typing import Callable

from ..schemas import QueryRecord


def _safe_str(x) -> str:
    return "" if x is None else str(x)


def load_nq(split: str = "validation", n: int | None = None, seed: int = 42) -> list[QueryRecord]:
    """Natural Questions Open."""
    from datasets import load_dataset

    ds = load_dataset("nq_open", split=split)
    return _materialize(ds, n=n, seed=seed, mapper=_map_nq, dataset_name="nq")


def load_triviaqa(
    split: str = "validation", n: int | None = None, seed: int = 42
) -> list[QueryRecord]:
    """TriviaQA — `rc.nocontext` config (no in-dataset evidence; we retrieve)."""
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)
    return _materialize(ds, n=n, seed=seed, mapper=_map_triviaqa, dataset_name="triviaqa")


def load_hotpotqa(
    split: str = "validation", n: int | None = None, seed: int = 42
) -> list[QueryRecord]:
    """HotpotQA — distractor split, multi-hop. Gold answer is the short string."""
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    return _materialize(ds, n=n, seed=seed, mapper=_map_hotpotqa, dataset_name="hotpotqa")


def _materialize(
    ds,
    n: int | None,
    seed: int,
    mapper: Callable[[dict, int], QueryRecord | None],
    dataset_name: str,
) -> list[QueryRecord]:
    indices = list(range(len(ds)))
    if n is not None and n < len(ds):
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:n]

    out: list[QueryRecord] = []
    for i in indices:
        rec = mapper(ds[i], i)
        if rec is not None:
            out.append(rec)
    # Stable order so re-runs are reproducible.
    out.sort(key=lambda r: r.qid)
    return out


def _map_nq(row: dict, idx: int) -> QueryRecord | None:
    answers = row.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    return QueryRecord(
        qid=f"nq-{idx}",
        dataset="nq",
        question=_safe_str(row.get("question", "")).strip(),
        gold_answers=[_safe_str(a) for a in answers if _safe_str(a)],
    )


def _map_triviaqa(row: dict, idx: int) -> QueryRecord | None:
    ans = row.get("answer") or {}
    aliases = ans.get("aliases") or []
    value = ans.get("value")
    gold = []
    if value:
        gold.append(_safe_str(value))
    gold.extend(_safe_str(a) for a in aliases if _safe_str(a))
    qid = _safe_str(row.get("question_id")) or f"triviaqa-{idx}"
    return QueryRecord(
        qid=qid,
        dataset="triviaqa",
        question=_safe_str(row.get("question", "")).strip(),
        gold_answers=list(dict.fromkeys(gold)),
    )


def _map_hotpotqa(row: dict, idx: int) -> QueryRecord | None:
    answer = _safe_str(row.get("answer", "")).strip()
    qid = _safe_str(row.get("id")) or f"hotpotqa-{idx}"
    meta = {"type": _safe_str(row.get("type")), "level": _safe_str(row.get("level"))}
    return QueryRecord(
        qid=qid,
        dataset="hotpotqa",
        question=_safe_str(row.get("question", "")).strip(),
        gold_answers=[answer] if answer else [],
        meta=meta,
    )


LOADERS = {
    "nq": load_nq,
    "triviaqa": load_triviaqa,
    "hotpotqa": load_hotpotqa,
}


def load(name: str, **kwargs) -> list[QueryRecord]:
    """Top-level entry point. `name` is one of: nq, triviaqa, hotpotqa."""
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from {sorted(LOADERS)}.")
    return LOADERS[name](**kwargs)
