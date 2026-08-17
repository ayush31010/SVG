"""Run the identifiability census on a machine that has a working renderer.

The census needs cairo (N+1 rasterizations per document), which the development
laptop does not have.  This ships the source plus the local emoji corpus to a
remote host, runs both corpora there, and brings the reports back.

    set SVG_REMOTE_PASSWORD=...
    python -m scripts.run_remote_census --sample 300
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import tarfile
import time
from pathlib import Path

try:
    import paramiko
except ImportError:  # pragma: no cover
    raise SystemExit("this driver needs paramiko: python -m pip install paramiko")

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "tmp" / "identifiability-census-bundle.tar.gz"
SKIP = {"__pycache__", ".pytest_cache", ".git"}

REMOTE = r"""#!/usr/bin/env bash
set -euo pipefail
ROOT="__ROOT__"
cd "$ROOT"
PY="$ROOT/venv/bin/python"

echo "=== unpack ==="
mkdir -p census
tar xzf "$ROOT/census-bundle.tar.gz" -C census

echo "=== dependencies ==="
"$PY" -m pip install --quiet --no-cache-dir zstandard 2>&1 | tail -2 || true
"$PY" -c "import cairosvg, zstandard; print('cairosvg + zstandard ok')"

cd census
export PYTHONPATH="$PWD"

echo "=== renderer-gated identifiability tests ==="
"$PY" -m unittest tests.test_identifiability -v 2>&1 | tail -6

echo "=== census: svgeditbench (curated emoji) ==="
"$PY" -m scripts.identifiability_census --corpus svgeditbench \
  --root SVGEditBench --sample __SAMPLE__ --size __SIZE__ --max-nodes __MAXNODES__ \
  --output "$ROOT/census-svgeditbench.json"

echo "=== census: openclipart (real-world CC0 clipart) ==="
"$PY" -m scripts.identifiability_census --corpus openclipart \
  --sample __SAMPLE__ --size __SIZE__ --max-nodes __MAXNODES__ \
  --output "$ROOT/census-openclipart.json"

echo "CENSUS_DONE"
"""


def _keep(rel: Path) -> bool:
    return not any(part in SKIP for part in rel.parts)


def build_bundle() -> Path:
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(BUNDLE, "w:gz") as tar:
        for name in ("svgpatchlab", "scripts", "tests", "train", "configs"):
            for path in sorted((REPO / name).rglob("*")):
                rel = path.relative_to(REPO)
                if path.is_file() and _keep(rel):
                    tar.add(path, arcname=rel.as_posix())
                    count += 1
        for name in ("pyproject.toml",):
            if (REPO / name).exists():
                tar.add(REPO / name, arcname=name)
                count += 1
        bench = REPO / "SVGEditBench"
        for path in sorted(bench.rglob("*.svg")):
            tar.add(path, arcname=path.relative_to(REPO).as_posix())
            count += 1
    print(f"bundle: {count} files, {BUNDLE.stat().st_size / 1e6:.1f} MB -> {BUNDLE}")
    return BUNDLE


def run(client: paramiko.SSHClient, command: str, *, check: bool = True) -> str:
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    _, stdout, stderr = client.exec_command(f"echo {encoded} | base64 -d | bash")
    stdout.channel.settimeout(None)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    if check and status != 0:
        raise SystemExit(f"remote failed ({status}):\n{out}\n{err}")
    return out + err


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="10.71.9.40")
    parser.add_argument("--user", default="trishita")
    parser.add_argument("--root", default="/home/trishita/svg-node-grounding")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=400,
        help="per-document rasterization budget; documents above it are skipped and named",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("runs/identifiability-census"))
    args = parser.parse_args(argv)

    password = os.environ.get("SVG_REMOTE_PASSWORD")
    if not password:
        raise SystemExit("SVG_REMOTE_PASSWORD is required")

    bundle = build_bundle()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=30)
    try:
        print(f"uploading to {args.user}@{args.host} ...", flush=True)
        started = time.time()
        with client.open_sftp() as sftp:
            sftp.put(str(bundle), f"{args.root}/census-bundle.tar.gz")
        print(f"uploaded in {time.time() - started:.1f}s", flush=True)

        script = (
            REMOTE.replace("__ROOT__", args.root)
            .replace("__SAMPLE__", str(args.sample))
            .replace("__SIZE__", str(args.size))
            .replace("__MAXNODES__", str(args.max_nodes))
        )
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        run(client, f"echo {encoded} | base64 -d > {args.root}/census.sh && chmod +x {args.root}/census.sh")

        print("running census (this rasterizes N+1 times per document) ...", flush=True)
        output = run(client, f"cd {args.root} && ./census.sh 2>&1", check=False)
        print(output)
        if "CENSUS_DONE" not in output:
            print("census did not reach the completion marker", file=sys.stderr)
            return 1

        args.results_dir.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            for name in (
                "census-svgeditbench.json",
                "census-svgeditbench.per_document.json",
                "census-openclipart.json",
                "census-openclipart.per_document.json",
            ):
                try:
                    sftp.get(f"{args.root}/{name}", str(args.results_dir / name))
                    print(f"  got {name}")
                except IOError:
                    print(f"  MISSING {name}")
        print(f"results in {args.results_dir}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
