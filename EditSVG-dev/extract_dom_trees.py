#!/usr/bin/env python3
"""
extract_dom_trees.py
────────────────────
Extracts DOM trees (skeletons + visual_context annotations from Vision LLM)
for a given number of SVGs from SVGEditBench and saves them as JSON files.

Usage:
    python3 extract_dom_trees.py --limit 70 --config configs/experiments/skeleton_patch_vision.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgpatchlab.config import load_config
from svgpatchlab.core.scene import build_scene
from svgpatchlab.data import SVGEditBench
from svgpatchlab.models import create_model
from svgpatchlab.vision import VisionContextAnnotator


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract annotated DOM trees for benchmark SVGs.")
    parser.add_argument(
        "--config",
        default="configs/experiments/skeleton_patch_vision.json",
        help="Path to experiment config containing vision_context configuration",
    )
    parser.add_argument("--limit", type=int, default=70, help="Number of SVGs to process (default: 70)")
    parser.add_argument(
        "--output-dir",
        default="runs/extracted_dom_trees",
        help="Directory to save output annotated DOM trees (default: runs/extracted_dom_trees)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Extract pure DOM tree skeleton without calling Vision LLM (CPU only, no model server needed)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    arch_config = config["architecture"]
    vision_config = arch_config.get("vision_context", {})

    use_vision = bool(vision_config.get("enabled")) and not args.no_vision
    vision_model = create_model(vision_config.get("model")) if use_vision else None
    annotator = VisionContextAnnotator(vision_config) if use_vision else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = SVGEditBench(config["dataset"]["root"])
    cases = list(benchmark.iter_cases(limit=args.limit))

    print(f"==========================================================")
    print(f" Extracting DOM trees for {len(cases)} SVGs using GPU Vision model")
    print(f" Output directory: {output_dir}")
    print(f"==========================================================")

    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] Processing {case.case_id} ({case.task}/{case.emoji_id})...")
        scene = build_scene(case.source_svg)
        if use_vision and annotator is not None:
            annotated_scene = annotator.annotate(
                svg=case.source_svg,
                scene=scene,
                model=vision_model,
                request_id=case.case_id,
            )
        else:
            annotated_scene = scene

        filename = f"{case.task}_{case.emoji_id}_dom.json"
        out_file = output_dir / filename
        out_file.write_text(json.dumps(annotated_scene, indent=2, sort_keys=True))

    print(f"\n✓ Done! Extracted {len(cases)} DOM trees to '{output_dir}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
