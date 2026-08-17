"""Bundle source, adapter, and dataset for the remote node-grounding eval.

The adapter weights and rendered evidence images are deliberately not in git,
so a remote GPU box cannot obtain them by cloning.  This packages them with the
source into one archive for `scripts/run_remote_full_eval.py` to ship.

    python -m scripts.build_eval_bundle
"""
from __future__ import annotations

import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts" / "node-grounding-qwen2.5-vl-7b-context-v3"
OUT = REPO / "tmp" / "node-grounding-eval-bundle.tar.gz"

SOURCE_DIRS = ["train", "svgpatchlab", "configs", "scripts"]
SOURCE_FILES = ["pyproject.toml", "requirements-node-grounding.txt", "requirements-eval.txt"]

SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".git", ".egg-info"}


def _keep(path: Path) -> bool:
    return not any(part in SKIP_DIR_NAMES or part.endswith(".egg-info") for part in path.parts)


def main() -> None:
    added = 0
    with tarfile.open(OUT, "w:gz") as tar:
        for name in SOURCE_DIRS:
            for path in sorted((REPO / name).rglob("*")):
                if path.is_file() and _keep(path.relative_to(REPO)):
                    tar.add(path, arcname=f"src/{path.relative_to(REPO).as_posix()}")
                    added += 1
        for name in SOURCE_FILES:
            path = REPO / name
            if path.exists():
                tar.add(path, arcname=f"src/{name}")
                added += 1

        # Trained adapter, including the tokenizer/processor saved beside it.
        for path in sorted((ART / "adapter").rglob("*")):
            if path.is_file():
                tar.add(path, arcname=f"adapter/{path.relative_to(ART / 'adapter').as_posix()}")
                added += 1

        # Dataset: manifests, split records, and the rendered evidence images.
        for name in ("manifest.json", "train.jsonl", "val.jsonl", "test.jsonl", "evidence-images.tar.gz"):
            path = ART / "dataset" / name
            tar.add(path, arcname=f"dataset/{name}")
            added += 1

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"{added} files -> {OUT}")
    print(f"bundle size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
