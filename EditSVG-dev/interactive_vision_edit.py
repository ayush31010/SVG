#!/usr/bin/env python3
"""
interactive_vision_edit.py
────────────────────────────
Interactive 4-step Vision Editing Pipeline for SVGs:

  1. Loads N SVGs and runs the Vision LLM to extract context-aware DOM trees.
  2. Displays the annotated DOM tree with visual_context attributes for verification.
  3. Prompts the user for a custom edit instruction.
  4. Generates the grounded patch JSON, executes it on the SVG, and displays the result.

Usage:
    python3 interactive_vision_edit.py --limit 2
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

if sys.platform == "darwin":
    _hb_path = "/opt/homebrew/lib"
    _curr_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _hb_path not in _curr_dyld:
        os.environ["DYLD_LIBRARY_PATH"] = f"{_hb_path}:{_curr_dyld}".rstrip(":")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgpatchlab.config import load_config
from svgpatchlab.core import apply_patch
from svgpatchlab.core.patch import parse_patch, extract_json_object
from svgpatchlab.core.scene import build_scene
from svgpatchlab.data import SVGEditBench
from svgpatchlab.models import create_model
from svgpatchlab.vision import VisionContextAnnotator
from svgpatchlab.architectures.prompts import patch_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Vision-Grounded SVG Editing Pipeline.")
    parser.add_argument(
        "--config",
        default="configs/experiments/skeleton_patch_vision.json",
        help="Path to vision experiment config",
    )
    parser.add_argument("--limit", type=int, default=2, help="Number of SVGs to process (default: 2)")
    parser.add_argument(
        "--output-dir",
        default="runs/interactive_vision_edit",
        help="Output directory for results and SVGs",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    arch_config = config["architecture"]
    vision_config = arch_config.get("vision_context", {})

    if not vision_config.get("enabled"):
        print("Error: vision_context is not enabled in configuration.")
        return 1

    print("==========================================================================")
    print(" 🚀 INTERACTIVE VISION-GROUNDED SVG EDITING PIPELINE")
    print("==========================================================================")

    # Initialize Vision and Patch Models
    print("\n[INIT] Initializing Vision Model and Patch Model adapters...")
    vision_model = create_model(vision_config.get("model"))
    patch_model = create_model(config.get("model"))
    annotator = VisionContextAnnotator(vision_config)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = SVGEditBench(config["dataset"]["root"])
    cases = list(benchmark.iter_cases(limit=args.limit))

    for idx, case in enumerate(cases, 1):
        print(f"\n{'='*74}")
        print(f" SVG #{idx}/{len(cases)} : {case.case_id} ({case.emoji_id})")
        print(f" Default Benchmark Instruction: {case.instruction!r}")
        print(f"{'='*74}")

        # ── STEP 1: Run Vision Model to annotate DOM tree ─────────────────────
        print(f"\n[STEP 1] Running Vision Model to extract contextual representation...")
        scene = build_scene(case.source_svg)
        
        try:
            annotated_scene = annotator.annotate(
                svg=case.source_svg,
                scene=scene,
                model=vision_model,
                request_id=f"interactive:{case.case_id}",
            )
            print(" ✓ Vision model annotation complete!")
        except Exception as exc:
            print(f" ✗ Vision Model Error: {exc}")
            print("   (Ensure your Vision Model server is running on http://localhost:8001/v1 or configured correctly)")
            continue

        # Save annotated DOM JSON
        dom_file = out_dir / f"{case.task}_{case.emoji_id}_annotated_dom.json"
        dom_file.write_text(json.dumps(annotated_scene, indent=2, sort_keys=True))
        print(f" ✓ Saved annotated DOM tree to: {dom_file}")

        # ── STEP 2: Display Annotated DOM Tree with Contextual Attributes ──────
        print(f"\n[STEP 2] Context-Aware DOM Tree Verification for {case.emoji_id}:")
        print("┌─ DOM TREE WITH VISUAL CONTEXT ──────────────────────────────────────")
        for node in annotated_scene["nodes"]:
            nid = node["id"]
            tag = node["tag"]
            fill = node.get("resolved_style", {}).get("fill", "")
            vc = node.get("visual_context")
            
            if vc:
                labels_str = ", ".join(vc.get("labels", []))
                summary = vc.get("summary", "")
                role = vc.get("role", "")
                bbox = vc.get("bbox", [])
                print(f"│  {nid:5s} <{tag:<6}> fill={fill:<8} | role={role:<8} labels=[{labels_str}]")
                print(f"│        └─ summary: \"{summary}\"")
                print(f"│        └─ bbox: {bbox}")
            else:
                print(f"│  {nid:5s} <{tag:<6}> fill={fill:<8} | (no visual_context)")
        print("└─────────────────────────────────────────────────────────────────────")

        # ── STEP 3: Prompt User for Exact Instruction ──────────────────────────
        print(f"\n[STEP 3] Enter your target edit instruction for this SVG.")
        print(f"         Press ENTER to use benchmark default: {case.instruction!r}")
        user_input = input(" ✏️ Your Edit Instruction > ").strip()
        user_instruction = user_input if user_input else case.instruction
        print(f" ▶ Active Instruction: {user_instruction!r}")

        # ── STEP 4: Generate Patch, Execute, & Display Results ───────────────
        print(f"\n[STEP 4] Generating grounded JSON patch and updating SVG...")
        context_json_str = json.dumps(annotated_scene, indent=2, sort_keys=True)
        prompt = patch_prompt(user_instruction, "SVG DOM skeleton with visual context", context_json_str)

        try:
            response = patch_model.generate(
                ModelRequest(
                    prompt=prompt,
                    metadata={"request_id": f"patch:{case.case_id}"},
                )
            )
            patch = parse_patch(response.text)
            output_svg = apply_patch(case.source_svg, patch)
        except Exception as exc:
            print(f" ✗ Patch Generation Error: {exc}")
            continue

        # Save Before & After SVGs and Patch JSON
        case_dir = out_dir / f"case_{idx}_{case.emoji_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        (case_dir / "before.svg").write_text(case.source_svg)
        (case_dir / "after.svg").write_text(output_svg)
        (case_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2))

        print(f"\n ── GENERATED PATCH JSON ──")
        print(json.dumps(patch.to_dict(), indent=2))

        print(f"\n ✓ Output files saved to {case_dir}/:")
        print(f"   - Before SVG : {case_dir}/before.svg")
        print(f"   - After SVG  : {case_dir}/after.svg")
        print(f"   - Patch JSON : {case_dir}/patch.json")

    print("\n==========================================================================")
    print(" ✓ Interactive Pipeline Finished!")
    print("==========================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
