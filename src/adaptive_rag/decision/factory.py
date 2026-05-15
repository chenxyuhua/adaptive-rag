"""Build retrieval deciders from config dicts (YAML-friendly)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..llm.base import LLMClient
from .base import AlwaysRetrieveDecider, NeverRetrieveDecider, RetrievalDecider
from .blended import BlendedRetrievalDecider
from .classifier import LinearClassifierRetrievalDecider
from .heuristic import HeuristicRetrievalDecider
from .prompt_decider import PromptRetrievalDecider
from .reflect import LLMReflection, ReflectionMixin


def build_decider(
    kind: str,
    llm: LLMClient,
    *,
    repo_root: Path,
    cfg: dict[str, Any],
) -> RetrievalDecider:
    """Factory for adaptive retrieval deciders.

    Parameters
    ----------
    kind:
        one of always, never, heuristic, prompt, blended, linear_classifier
    cfg:
        kind-specific keys, e.g. prompt paths, weights_path, blend weights.
    """
    kind = (kind or "always").strip().lower()
    if kind == "always":
        return AlwaysRetrieveDecider()
    if kind == "never":
        return NeverRetrieveDecider()
    if kind == "heuristic":
        return HeuristicRetrievalDecider(
            uncertain_boost=float(cfg.get("uncertain_boost", 0.85)),
            confident_drop=float(cfg.get("confident_drop", 0.15)),
        )
    if kind == "prompt":
        rel = cfg.get("prompt_path", "prompts/retrieve_decision.txt")
        path = Path(rel)
        if not path.is_absolute():
            path = repo_root / path
        text = path.read_text(encoding="utf-8")
        return PromptRetrievalDecider(llm, text, max_output_tokens=int(cfg.get("max_output_tokens", 128)))
    if kind == "linear_classifier":
        wp = cfg.get("weights_path")
        if not wp:
            raise ValueError("linear_classifier requires adaptive.decider.weights_path")
        path = Path(wp)
        if not path.is_absolute():
            path = repo_root / path
        return LinearClassifierRetrievalDecider(path)
    if kind == "blended":
        parts_cfg = cfg.get("parts") or []
        parts: list[tuple[RetrievalDecider, float]] = []
        for block in parts_cfg:
            k = block.get("kind")
            w = float(block.get("weight", 1.0))
            sub = {k2: v2 for k2, v2 in block.items() if k2 not in {"kind", "weight"}}
            parts.append((build_decider(k, llm, repo_root=repo_root, cfg=sub), w))
        return BlendedRetrievalDecider(parts)
    raise ValueError(f"Unknown adaptive decider kind: {kind!r}")


def maybe_wrap_reflection(
    decider: RetrievalDecider,
    llm: LLMClient,
    *,
    repo_root: Path,
    enabled: bool,
    cfg: dict[str, Any],
) -> RetrievalDecider:
    if not enabled:
        return decider
    rel = cfg.get("prompt_path", "prompts/reflect_critique.txt")
    path = Path(rel)
    if not path.is_absolute():
        path = repo_root / path
    text = path.read_text(encoding="utf-8")
    reflection = LLMReflection(llm, text, max_output_tokens=int(cfg.get("max_output_tokens", 256)))
    return ReflectionMixin(decider, reflection)
