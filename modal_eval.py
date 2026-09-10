import modal
import subprocess
import time
import sys
import os

app = modal.App("svgpatchlab-dual-eval")

# Define the environment
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libcairo2-dev", "pkg-config", "python3-dev")
    .pip_install(
        "vllm",
        "cairosvg>=2.7",
        "Pillow>=9",
        "torch",
        "torch-geometric",
        "sentence-transformers",
        "pydantic",
        "openai"
    )
    .add_local_file("patch_flashinfer.py", "/tmp/patch_flashinfer.py", copy=True)
    .run_commands("python3 /tmp/patch_flashinfer.py")
    .add_local_dir(".", remote_path="/root/EditSVG-patch-lab")
)


def _boot_vllm_and_run(config: str, limit_per_task: int = 0):
    """Shared helper: boots a single-GPU vLLM server and runs one eval config."""
    os.chdir("/root/EditSVG-patch-lab")

    print(f"--> Installing local svgpatchlab package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

    print(f"--> Booting Qwen3.5-4B vLLM server on port 8000 for config: {config}")
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3.5-4B",
        "--port", "8000",
        "--enforce-eager",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.95"
    ]
    server = subprocess.Popen(vllm_cmd, stdout=sys.stdout, stderr=sys.stderr)

    import urllib.request
    import urllib.error
    print("--> Waiting for vLLM server to start...")
    for i in range(60):
        try:
            with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2) as r:
                if r.status == 200:
                    print("--> Server is up and ready!")
                    break
        except urllib.error.URLError:
            pass
        if server.poll() is not None:
            raise RuntimeError("vLLM server crashed during startup!")
        time.sleep(10)
    else:
        server.terminate()
        raise TimeoutError("vLLM server took too long to start.")

    eval_cmd = [
        sys.executable, "-m", "svgpatchlab.cli", "evaluate",
        "--config", config,
    ]
    if limit_per_task > 0:
        eval_cmd += ["--limit-per-task", str(limit_per_task)]

    print(f"--> Running evaluation: {config} (limit_per_task={limit_per_task or 'ALL'})")
    try:
        subprocess.run(eval_cmd, check=True)
    finally:
        server.terminate()

    print(f"--> DONE: {config}")


# ── Run 1: skeleton_patch + vision (the official thesis experiment) ──────────
@app.function(image=image, gpu="L4:1", timeout=7200)
def run_skeleton_vision(limit_per_task: int = 0):
    _boot_vllm_and_run("configs/experiments/skeleton_patch_vision.json", limit_per_task)


# ── Run 2: gnn_patch + vision (the experimental GNN approach) ───────────────
@app.function(image=image, gpu="L4:1", timeout=7200)
def run_gnn_vision(limit_per_task: int = 0):
    _boot_vllm_and_run("configs/experiments/gnn_patch.json", limit_per_task)


# ── Local entrypoint: launches BOTH in parallel ──────────────────────────────
@app.local_entrypoint()
def main(limit_per_task: int = 0, quick: bool = False):
    """
    Run both evaluations in parallel on separate L4 GPUs.

    Full run (~5 hrs each, simultaneous):
        modal run --detach modal_eval.py

    Quick run (125 cases each, ~2.5 hrs, simultaneous):
        modal run --detach modal_eval.py --quick

    Custom limit:
        modal run --detach modal_eval.py --limit-per-task 25
    """
    if quick:
        limit_per_task = 25  # 25 cases * 5 tasks = 125 cases each

    print(f"==> Launching BOTH evaluations in parallel (limit_per_task={limit_per_task or 'ALL'})")
    print(f"==> skeleton_patch_vision → runs/skeleton_patch_vision/")
    print(f"==> gnn_patch             → runs/gnn_patch/")

    # Launch both simultaneously — Modal schedules them on separate L4 instances
    skeleton_handle = run_skeleton_vision.spawn(limit_per_task)
    gnn_handle = run_gnn_vision.spawn(limit_per_task)

    print("==> Both jobs launched! Waiting for results...")

    skeleton_handle.get()
    print("==> skeleton_patch_vision COMPLETE!")

    gnn_handle.get()
    print("==> gnn_patch COMPLETE!")

    print("==> ALL EVALUATIONS DONE!")
