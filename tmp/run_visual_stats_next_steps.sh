#!/usr/bin/env bash
set -euo pipefail

cd /home/trishita/svgpatchlab-semantic
export PYTHONPATH=.deps
PYTHON=/home/trishita/miniconda3/envs/trishita/bin/python

"$PYTHON" -u scripts/run_spatial_grounding_holdout.py \
  --model-config configs/models/qwen2.5-7b-ollama.json \
  --output-root runs/qwen2.5-7b-spatial-grounding-holdout-v1-server \
  --seed 20260725

"$PYTHON" -u -m svgpatchlab.cli matrix \
  --config configs/experiments/skeleton_patch.json \
  --model-config configs/models/qwen2.5-7b-ollama.json \
  --architectures skeleton_patch visual_stats_patch \
  --render \
  --output-root runs/qwen2.5-7b-visual-stats-ab-full-v1-server

"$PYTHON" -u scripts/analyze_visual_stats_ab.py \
  --root runs/qwen2.5-7b-visual-stats-ab-full-v1-server \
  --dataset-root SVGEditBench
