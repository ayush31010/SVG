#!/bin/sh
set -eu
cd /home/trishita/svgpatchlab-semantic
export PYTHONPATH=.:.deps
exec /home/trishita/miniconda3/envs/trishita/bin/python -u \
  scripts/run_occlusion_deletion_benchmark.py \
  --model-config configs/models/qwen2.5vl-32b-ollama-constrained.json \
  --output-root runs/qwen2.5vl-32b-occlusion-deletion-v6
