import modal
import subprocess
import time
import sys
import os

app = modal.App("svgpatchlab-gnn-eval")

# Define the environment: installing all required packages including PyTorch, PyG, and vLLM
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libcairo2-dev", "pkg-config", "python3-dev")
    .pip_install(
        "vllm",
        "cairosvg>=2.7",
        "Pillow>=9",
        "torch==2.4.0",
        "torch-geometric",
        "sentence-transformers",
        "pydantic",
        "openai"
    )
    # Patch flashinfer's JIT build to remove --compress-mode=size,
    # a CUDA 13.x-only flag that crashes on Modal's CUDA 12.1.1 host nvcc.
    # The patch removes the flag so JIT kernels compile cleanly on CUDA 12.x.
    .add_local_file("patch_flashinfer.py", "/tmp/patch_flashinfer.py")
    .run_commands("python3 /tmp/patch_flashinfer.py")
    .add_local_dir(".", remote_path="/root/EditSVG-patch-lab")
)

# Request an L4 GPU (stronger than T4, usually allowed on free tier)
@app.function(
    image=image, 
    gpu="L4", 
    timeout=7200
)
def run_evaluation():
    os.chdir("/root/EditSVG-patch-lab")
    
    print("--> Installing local svgpatchlab package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    print("--> Booting Qwen3.5-4B vLLM server on port 8000...")
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3.5-4B",
        "--port", "8000",
        "--enforce-eager",  # Prevents CUDA graph memory overallocation
        "--max-model-len", "32768"  # Cap context length to reduce memory pressure
    ]
    server = subprocess.Popen(vllm_cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    # Dynamically wait for the server to actually start listening on port 8000
    print("--> Waiting for vLLM server to start accepting connections on port 8000...")
    import urllib.request
    import urllib.error
    max_retries = 60 # wait up to 10 minutes (60 * 10 seconds)
    for i in range(max_retries):
        try:
            req = urllib.request.Request("http://localhost:8000/v1/models")
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print("--> Server is up and ready!")
                    break
        except urllib.error.URLError:
            pass
        
        # Check if the process crashed
        if server.poll() is not None:
            raise RuntimeError("vLLM server crashed during startup!")
            
        time.sleep(10)
    else:
        server.terminate()
        raise TimeoutError("vLLM server took too long to start.")
    
    print("--> Starting GNN Evaluation...")
    # NOTE: We disable vision context here for the first test so we don't have to boot TWO heavy models at once.
    eval_cmd = [
        sys.executable, "-m", "svgpatchlab.cli", "evaluate",
        "--config", "configs/experiments/gnn_patch.json"
    ]
    
    try:
        subprocess.run(eval_cmd, check=True)
    finally:
        server.terminate()
        
    print("--> Evaluation Complete! Results saved to runs/gnn_patch/")
