#!/usr/bin/env python3
"""Thin CLI wrapper around adaptive_rag.corpus.build_index.

Use the Colab notebook (`scripts/build_index_colab.ipynb`) for the GPU build;
this CLI is for local CPU runs or when running inside Colab via shell.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from adaptive_rag.corpus.build_index import build  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build FAISS index over Wikipedia subset.")
    p.add_argument("--out-dir", default="data/index")
    p.add_argument("--num-passages", type=int, default=200_000)
    p.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=os.environ.get("EMBED_DEVICE", "cpu"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    info = build(
        out_dir=args.out_dir,
        num_passages=args.num_passages,
        embedder_model=args.embedder,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(info, indent=2))
