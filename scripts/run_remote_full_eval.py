"""Drive the full 153-row node-grounding evaluation on a remote GPU box.

Runs the whole remaining pipeline end to end: upload the bundle, lay out the
repo, verify the adapter survived transfer, evaluate the base model and the
trained adapter over the complete test manifest, build the matched comparison,
and copy the results back.

The remote environment is expected to already carry a venv at
``<root>/venv`` with the pinned stack (torch 2.11.0+cu128, transformers
5.15.0, peft 0.20.0, accelerate 1.14.0).

Credentials come from the environment, never the command line::

    set SVG_REMOTE_PASSWORD=...            # Windows
    export SVG_REMOTE_PASSWORD=...         # POSIX

    python -m scripts.run_remote_full_eval \
        --host 10.71.9.40 --user trishita \
        --bundle path/to/node-grounding-eval-bundle.tar.gz

Re-running is safe: the upload is skipped when the remote copy already matches,
and evaluation is skipped when its metrics file already exists (use --force to
redo it anyway).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
import time
from pathlib import Path

try:
    import paramiko
except ImportError:  # pragma: no cover - dependency is documented in the usage
    raise SystemExit("this driver needs paramiko: python -m pip install paramiko")


ADAPTER_SHA256 = "dc497b620d9cc22d36b541c481e0097941caa12f6cb3aeb5fcbe8c131dfddc49"

REMOTE_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
ROOT="__ROOT__"
GPU="__GPU__"
FORCE="__FORCE__"
cd "$ROOT"

PY="$ROOT/venv/bin/python"
WORK="$ROOT/work"
DATA="$WORK/data/node_grounding-context-v3"
CKPT="$WORK/checkpoints/node-grounding-qwen2.5-vl-7b-context-v3"

echo "=== [1/6] unpack bundle ==="
mkdir -p "$ROOT/unpacked"
tar xzf "$ROOT/eval-bundle.tar.gz" -C "$ROOT/unpacked"

echo "=== [2/6] lay out repo ==="
mkdir -p "$WORK" "$DATA" "$CKPT"
cp -r "$ROOT/unpacked/src/." "$WORK/"
cp "$ROOT/unpacked/dataset/manifest.json" "$DATA/"
cp "$ROOT/unpacked/dataset/"*.jsonl "$DATA/"
if [ ! -d "$DATA/images/test" ]; then
  tar xzf "$ROOT/unpacked/dataset/evidence-images.tar.gz" -C "$DATA"
fi
cp -r "$ROOT/unpacked/adapter/." "$CKPT/"
echo "test images: $(ls -1 "$DATA/images/test" | wc -l)"
echo "test rows:   $(grep -c '' "$DATA/test.jsonl")"

echo "=== [3/6] verify adapter checksum ==="
ACTUAL=$(sha256sum "$CKPT/adapter_model.safetensors" | cut -d' ' -f1)
echo "expected __ADAPTER_SHA__"
echo "actual   $ACTUAL"
if [ "$ACTUAL" != "__ADAPTER_SHA__" ]; then
  echo "ADAPTER CHECKSUM MISMATCH - aborting"; exit 1
fi

cd "$WORK"
export PYTHONPATH="$WORK"
export HF_HOME="$ROOT/hf"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

echo "=== [4/6] base model over the full manifest ==="
if [ "$FORCE" = "1" ] || [ ! -f "$WORK/runs/node-grounding-context-v3-base-test-full/test_metrics.json" ]; then
  "$PY" -m train.node_grounding_sft \
    --config configs/train/node_grounding_context_h200_base_full_eval.json \
    --eval-only > "$ROOT/base_eval.log" 2>&1
  echo "base eval finished"
else
  echo "base metrics already present - skipping"
fi

echo "=== [5/6] trained adapter over the full manifest ==="
if [ "$FORCE" = "1" ] || [ ! -f "$WORK/runs/node-grounding-context-v3-trained-test-full/test_metrics.json" ]; then
  "$PY" -m train.node_grounding_sft \
    --config configs/train/node_grounding_context_h200_full_eval.json \
    --eval-only > "$ROOT/trained_eval.log" 2>&1
  echo "trained eval finished"
else
  echo "trained metrics already present - skipping"
fi

echo "=== [6/6] matched comparison ==="
"$PY" -m scripts.compare_node_grounding_runs \
  --base runs/node-grounding-context-v3-base-test-full \
  --trained runs/node-grounding-context-v3-trained-test-full \
  --data-dir "$DATA" \
  --output "$ROOT/comparison-full.json" > "$ROOT/compare.log" 2>&1
echo "comparison written"

echo "REMOTE_PIPELINE_DONE"
"""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def connect(host: str, user: str) -> paramiko.SSHClient:
    password = os.environ.get("SVG_REMOTE_PASSWORD")
    if not password:
        raise SystemExit("SVG_REMOTE_PASSWORD is required")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host, username=user, password=password, timeout=30, banner_timeout=30, auth_timeout=30
    )
    return client


def run(client: paramiko.SSHClient, command: str, *, check: bool = True) -> str:
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    _, stdout, stderr = client.exec_command(f"echo {encoded} | base64 -d | bash")
    stdout.channel.settimeout(None)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    if check and status != 0:
        raise SystemExit(f"remote command failed ({status}):\n{out}\n{err}")
    return out + err


def upload_bundle(client: paramiko.SSHClient, bundle: Path, remote_path: str) -> None:
    local_size = bundle.stat().st_size
    existing = run(client, f"stat -c %s {remote_path} 2>/dev/null || echo missing").strip()
    if existing.isdigit() and int(existing) == local_size:
        log(f"bundle already present remotely ({local_size / 1e6:.1f} MB) - skipping upload")
        return

    log(f"uploading {local_size / 1e6:.1f} MB ...")
    started = time.time()
    sent = 0
    last = 0.0

    def progress(transferred: int, total: int) -> None:
        nonlocal sent, last
        sent = transferred
        now = time.time()
        if now - last > 5 or transferred == total:
            last = now
            pct = 100.0 * transferred / total if total else 0.0
            rate = transferred / max(now - started, 1e-6) / 1e6
            log(f"  {pct:5.1f}%  {transferred / 1e6:7.1f} MB  {rate:5.1f} MB/s")

    with client.open_sftp() as sftp:
        sftp.put(str(bundle), remote_path, callback=progress)
    log(f"upload complete in {time.time() - started:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="10.71.9.40")
    parser.add_argument("--user", default="trishita")
    parser.add_argument("--root", default="/home/trishita/svg-node-grounding")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES on the remote box")
    parser.add_argument("--force", action="store_true", help="re-run evals even if metrics exist")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--results-dir", type=Path, default=Path("runs/full-manifest-remote"))
    args = parser.parse_args(argv)

    if not args.bundle.exists():
        raise SystemExit(f"bundle not found: {args.bundle}")

    log(f"connecting to {args.user}@{args.host}")
    client = connect(args.host, args.user)
    try:
        run(client, f"mkdir -p {args.root}")
        upload_bundle(client, args.bundle, f"{args.root}/eval-bundle.tar.gz")

        log("checking remote venv")
        venv = run(
            client,
            f'test -x {args.root}/venv/bin/python && {args.root}/venv/bin/python -c '
            f'"import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)" '
            f'|| echo VENV_MISSING',
        ).strip()
        if "VENV_MISSING" in venv:
            raise SystemExit(
                f"no usable venv at {args.root}/venv - install the pinned stack first"
            )
        log(f"remote stack: {venv}")

        script = (
            REMOTE_SCRIPT.replace("__ROOT__", args.root)
            .replace("__GPU__", args.gpu)
            .replace("__FORCE__", "1" if args.force else "0")
            .replace("__ADAPTER_SHA__", ADAPTER_SHA256)
        )
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        run(client, f"echo {encoded} | base64 -d > {args.root}/pipeline.sh && chmod +x {args.root}/pipeline.sh")

        log("launching remote pipeline (detached)")
        # All three streams are redirected: a job inheriting stdout keeps the
        # exec channel open until it exits, which hangs the SSH session.
        run(
            client,
            f"cd {args.root} && nohup ./pipeline.sh > pipeline.log 2>&1 < /dev/null & echo started",
        )

        log("waiting for completion (this includes a ~16 GB model download on first run)")
        while True:
            time.sleep(args.poll_seconds)
            state = run(
                client,
                f"cd {args.root} && "
                f"(grep -q REMOTE_PIPELINE_DONE pipeline.log && echo DONE) || "
                f"(pgrep -f '[p]ipeline.sh' >/dev/null && echo RUNNING) || echo STOPPED; "
                f"tail -n 2 pipeline.log",
            )
            head = state.strip().splitlines()
            marker = head[0].strip() if head else "UNKNOWN"
            tail = " | ".join(line.strip() for line in head[1:])
            log(f"{marker}: {tail}")
            if marker == "DONE":
                break
            if marker == "STOPPED":
                log("pipeline exited without the completion marker; recent output:")
                print(run(client, f"tail -n 40 {args.root}/pipeline.log"))
                return 1

        log("fetching results")
        args.results_dir.mkdir(parents=True, exist_ok=True)
        wanted = [
            ("comparison-full.json", "comparison-full.json"),
            ("pipeline.log", "pipeline.log"),
            ("base_eval.log", "base_eval.log"),
            ("trained_eval.log", "trained_eval.log"),
            ("work/runs/node-grounding-context-v3-base-test-full/test_metrics.json", "base_test_metrics.json"),
            ("work/runs/node-grounding-context-v3-trained-test-full/test_metrics.json", "trained_test_metrics.json"),
            ("work/runs/node-grounding-context-v3-base-test-full/test_predictions.jsonl", "base_test_predictions.jsonl"),
            ("work/runs/node-grounding-context-v3-trained-test-full/test_predictions.jsonl", "trained_test_predictions.jsonl"),
        ]
        with client.open_sftp() as sftp:
            for remote_name, local_name in wanted:
                remote = f"{args.root}/{remote_name}"
                try:
                    sftp.get(remote, str(args.results_dir / local_name))
                    log(f"  got {local_name}")
                except IOError:
                    log(f"  MISSING {remote_name}")

        digest = hashlib.sha256(
            (args.results_dir / "comparison-full.json").read_bytes()
        ).hexdigest()
        log(f"comparison-full.json sha256 {digest}")
        log(f"results in {args.results_dir}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
