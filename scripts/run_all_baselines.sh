#!/usr/bin/env bash
# Run every baseline (3 strategies × 3 datasets) at a given sample size,
# then write the consolidated summary. Designed to be safe to re-run:
# new run dirs are created with timestamps, so nothing is overwritten.
#
# Usage: scripts/run_all_baselines.sh [N]
#   N — per-dataset sample size (default 50)
#
# Prereqs:
#   - .env contains GOOGLE_API_KEY
#   - data/index/wiki_subset.{faiss,jsonl} exist (Colab build)
set -euo pipefail

N=${1:-50}
PY=.venv/bin/python

echo "== Baselines: n=$N per dataset, 3 strategies, gemini-2.5-flash"

for DATASET in nq triviaqa hotpotqa; do
  echo "--- $DATASET / no_retrieval"
  $PY scripts/run_baseline.py --dataset "$DATASET" --strategy no_retrieval --n "$N"

  echo "--- $DATASET / fixed_k k=5"
  $PY scripts/run_baseline.py --dataset "$DATASET" --strategy fixed_k --k 5 --n "$N"

  echo "--- $DATASET / adaptive"
  $PY scripts/run_baseline.py --dataset "$DATASET" --strategy adaptive --n "$N"
done

echo "== Writing summary"
$PY scripts/summarize_baselines.py
echo "== Done. See runs/summary.md"
