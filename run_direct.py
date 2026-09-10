#!/usr/bin/env python3
"""
Standalone evaluation runner for H100 (or any GPU).
No Modal required — just run: python run_direct.py

Requirements:
  pip install vllm cairosvg Pillow torch torch-geometric sentence-transformers pydantic openai

Then start this script. It will:
  1. Boot Qwen3.5-4B vLLM server on port 8000
  2. Run BOTH skeleton_patch_vision and gnn_patch evaluations sequentially
  3. Print final scores when done
"""

import subprocess
import sys
import os
import time
import urllib.request
import urllib.error

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def boot_vllm(port: int = 8000, tensor_parallel: int = 1):
    """Start the vLLM server and wait until it's ready."""
    os.chdir(REPO_ROOT)
    print(f"\n{'='*60}")
    print(f" BOOTING Qwen3.5-4B vLLM on port {port}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3.5-4B",
        "--port", str(port),
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tensor_parallel),
    ]
    server = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    print(f"--> Waiting for server on port {port}...")
    for _ in range(120):  # wait up to 20 minutes
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2) as r:
                if r.status == 200:
                    print(f"--> Server ready on port {port}!")
                    return server
        except urllib.error.URLError:
            pass
        if server.poll() is not None:
            raise RuntimeError("vLLM crashed during startup! Check logs above.")
        time.sleep(10)

    server.terminate()
    raise TimeoutError("vLLM server took too long to start.")


def run_eval(config: str, limit_per_task: int = 0):
    """Run one evaluation config and return the result."""
    print(f"\n{'='*60}")
    print(f" EVALUATING: {config}")
    print(f" Limit per task: {'ALL' if limit_per_task == 0 else limit_per_task}")
    print(f"{'='*60}")

    cmd = [sys.executable, "-m", "svgpatchlab.cli", "evaluate", "--config", config]
    if limit_per_task > 0:
        cmd += ["--limit-per-task", str(limit_per_task)]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    os.chdir(REPO_ROOT)

    # Install local package
    print("--> Installing svgpatchlab package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

    # ── CONFIG ───────────────────────────────────────────────────────────────
    # Set to 0 for full 500-case benchmark, or e.g. 20 for quick 100-case run
    LIMIT_PER_TASK = 0  # Full benchmark — H100 can handle it in ~1-2 hrs

    # If your H100 has multiple GPUs, increase this (must divide 64):
    # 1 GPU → tensor_parallel=1
    # 2 GPUs → tensor_parallel=2
    # 4 GPUs → tensor_parallel=4
    TENSOR_PARALLEL = 1
    # ─────────────────────────────────────────────────────────────────────────

    server = boot_vllm(port=8000, tensor_parallel=TENSOR_PARALLEL)

    try:
        # Run 1: skeleton_patch + vision (core thesis experiment)
        run_eval("configs/experiments/skeleton_patch_vision.json", LIMIT_PER_TASK)

        # Run 2: GNN + vision (experimental)
        run_eval("configs/experiments/gnn_patch.json", LIMIT_PER_TASK)

    finally:
        print("\n--> Shutting down vLLM server...")
        server.terminate()

    print("\n" + "="*60)
    print(" ALL DONE!")
    print(" Results: runs/skeleton_patch_vision/summary.json")
    print("          runs/gnn_patch/summary.json")
    print("="*60)
