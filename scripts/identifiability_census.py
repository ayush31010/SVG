"""Measure how many SVG nodes are identifiable from the rendering alone.

Motivating measurement for render-grounded node selection: if a large share of
real-world nodes produce no distinctive pixels, then *no* amount of visual
evidence can ground a referring expression onto them, and a grounding system
that always commits to an answer is reporting confidence it cannot possess.

Two corpora, deliberately contrasted:

``svgeditbench``
    The curated emoji benchmark this project has been evaluated on. Flat,
    hand-cleaned art. Expected to look easy.

``openclipart``
    178,604 CC0 artist-authored clipart SVGs from Hugging Face
    (``nyuuzyou/openclipart``), distributed as 18 zstd-compressed JSONL shards.
    Real drawing practice, with wrapper groups, stacked fills, and genuine
    occlusion.

Usage::

    python -m scripts.identifiability_census --corpus svgeditbench \
        --root SVGEditBench --sample 300 --output runs/census-emoji.json

    python -m scripts.identifiability_census --corpus openclipart \
        --sample 300 --output runs/census-clipart.json

``openclipart`` needs ``huggingface_hub`` and ``zstandard``; only one shard is
downloaded, not the full 22 GB.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

from svgpatchlab.vision.identifiability import identifiability_census

HF_REPO = "nyuuzyou/openclipart"


def _iter_svgeditbench(root: Path, limit: int, seed: int) -> Iterator[tuple[str, str]]:
    paths = sorted(root.rglob("*.svg"))
    if not paths:
        raise SystemExit(f"no .svg files under {root}")
    rng = random.Random(seed)
    rng.shuffle(paths)
    taken = 0
    for path in paths:
        if taken >= limit:
            return
        try:
            svg = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        taken += 1
        yield str(path.relative_to(root)).replace("\\", "/"), svg


def _iter_openclipart(shard: int, limit: int, seed: int) -> Iterator[tuple[str, str]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("openclipart needs huggingface_hub") from exc
    try:
        import zstandard
    except ImportError as exc:
        raise SystemExit("openclipart needs zstandard") from exc

    filename = f"openclipart_{shard:02d}.jsonl.zst"
    print(f"downloading {filename} from {HF_REPO} ...", flush=True)
    local = hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="dataset")

    # Reservoir-sample the shard so the census is not biased toward whichever
    # documents happen to sit at the front of the file.
    rng = random.Random(seed)
    reservoir: list[tuple[str, str]] = []
    seen = 0
    decompressor = zstandard.ZstdDecompressor()
    with open(local, "rb") as raw, decompressor.stream_reader(raw) as stream:
        text = io.TextIOWrapper(stream, encoding="utf-8")
        for line in text:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            svg = record.get("svg_content")
            if not isinstance(svg, str) or "<svg" not in svg:
                continue
            name = str(record.get("title") or record.get("page_url") or f"row{seen}")
            item = (f"{seen}:{name}"[:120], svg)
            seen += 1
            if len(reservoir) < limit:
                reservoir.append(item)
            else:
                j = rng.randrange(seen)
                if j < limit:
                    reservoir[j] = item
    print(f"sampled {len(reservoir)} of {seen} clipart documents", flush=True)
    yield from reservoir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("svgeditbench", "openclipart"), required=True)
    parser.add_argument("--root", type=Path, default=Path("SVGEditBench"))
    parser.add_argument("--shard", type=int, default=0, help="openclipart shard 0-17")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--size", type=int, default=128, help="raster size in px")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--max-nodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.corpus == "svgeditbench":
        documents = dict(_iter_svgeditbench(args.root, args.sample, args.seed))
    else:
        documents = dict(_iter_openclipart(args.shard, args.sample, args.seed))

    if not documents:
        raise SystemExit("no documents collected")
    print(f"analysing {len(documents)} documents at {args.size}px ...", flush=True)

    census = identifiability_census(
        documents,
        size=args.size,
        threshold=args.threshold,
        max_nodes=args.max_nodes,
    )
    census["corpus"] = args.corpus
    census["sample_requested"] = args.sample
    census["seed"] = args.seed
    if args.corpus == "openclipart":
        census["shard"] = args.shard
        census["source"] = HF_REPO
        census["license"] = "CC0-1.0"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # per_document is verbose; keep it out of the headline file.
    per_document = census.pop("per_document", {})
    args.output.write_text(
        json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    detail = args.output.with_suffix(".per_document.json")
    detail.write_text(
        json.dumps(per_document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    share = census["node_share"]
    print(f"\n=== {args.corpus} ===")
    print(f"documents analysed : {census['documents_analysed']}")
    print(f"documents skipped  : {census['documents_skipped']}")
    print(f"nodes              : {census['node_total']}")
    print(f"  identifiable     : {census['counts']['identifiable']:6d}  {share['identifiable']*100:5.1f}%")
    print(f"  render_equivalent: {census['counts']['render_equivalent']:6d}  {share['render_equivalent']*100:5.1f}%")
    print(f"  invisible        : {census['counts']['invisible']:6d}  {share['invisible']*100:5.1f}%")
    print(
        "documents containing at least one non-identifiable node: "
        f"{census['document_share_with_non_identifiable']*100:.1f}%"
    )
    print(f"\nwrote {args.output} and {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
