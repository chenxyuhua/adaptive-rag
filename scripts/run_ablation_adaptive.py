#!/usr/bin/env python3
"""Sweep adaptive ablations (threshold, reflection) by invoking run_baseline.py.

Example
-------
python scripts/run_ablation_adaptive.py --dataset nq --n 20 \\
  --thresholds 0.3 0.5 0.7 --reflection-modes off on
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nq", choices=["nq", "triviaqa", "hotpotqa"])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--config", default=str(_REPO / "configs" / "default.yaml"))
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument(
        "--reflection-modes",
        nargs="+",
        default=["off", "on"],
        choices=["off", "on"],
        help="off = no critique step; on = --reflection",
    )
    args = p.parse_args()

    for ref in args.reflection_modes:
        for thr in args.thresholds:
            cmd = [
                sys.executable,
                str(_REPO / "scripts" / "run_baseline.py"),
                "--config",
                args.config,
                "--dataset",
                args.dataset,
                "--strategy",
                "adaptive",
                "--n",
                str(args.n),
                "--decision-threshold",
                str(thr),
            ]
            if ref == "on":
                cmd.append("--reflection")
            print("[ablation] Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=str(_REPO))


if __name__ == "__main__":
    main()
