#!/usr/bin/env python3
"""
demo_vision_grounding.py
────────────────────────
Demonstrates the end-to-end visual grounding pipeline for 1 SVG:

  1. Build real DOM skeleton from SVGEditBench (1f199 — blue badge emoji)
  2. Inject realistic mock visual_context (what Qwen2.5-VL would produce)
  3. Construct the full patch prompt → shows the LLM sees "text" label on n2
  4. Simulate the grounded patch: "change the text to purple" → n2.fill = purple
  5. Apply patch, save before/after SVGs

No model server required — the mock annotation simulates GPU vision output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgpatchlab.core import apply_patch
from svgpatchlab.core.patch import Patch, PatchOperation
from svgpatchlab.core.scene import build_scene
from svgpatchlab.data.svgeditbench import SVGEditBench
from svgpatchlab.architectures.prompts import patch_prompt

# ─────────────────────────────────────────────────────────────────────────────
INSTRUCTION = "change the text to purple"
OUT_DIR = Path("runs/demo_grounding")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIVIDER = "─" * 68

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load 1 real SVG case
# ─────────────────────────────────────────────────────────────────────────────
bench = SVGEditBench("SVGEditBench")
case = next(bench.iter_cases(tasks=["change_color"], limit=1))

print(f"\n{DIVIDER}")
print(f"  Case : {case.case_id}  ({case.emoji_id})")
print(f"  Edit : {INSTRUCTION!r}")
print(DIVIDER)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build real DOM skeleton (no vision yet)
# ─────────────────────────────────────────────────────────────────────────────
scene_raw = build_scene(case.source_svg)

print("\n[STEP 1] Raw DOM skeleton (no vision context):")
for node in scene_raw["nodes"]:
    fill = node.get("resolved_style", {}).get("fill", "")
    print(f"  {node['id']:5s} <{node['tag']:<6}> fill={fill or '(none)'}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Inject mock visual_context  (simulates Qwen2.5-VL output)
# ─────────────────────────────────────────────────────────────────────────────
MOCK_VISUAL_CONTEXT = {
    "n1": {
        "summary": "solid blue rounded rectangle forming the badge background",
        "labels": ["background", "badge", "rectangle"],
        "role": "background",
        "confidence": 0.97,
        "bbox": [0, 0, 384, 384],
        "visible_pixels": 146034,
    },
    "n2": {
        "summary": "white UP! text glyph inside the blue badge",
        "labels": ["text", "glyph", "label", "foreground"],
        "role": "object",
        "confidence": 0.94,
        "bbox": [24, 104, 325, 166],
        "visible_pixels": 27650,
    },
}

import copy
scene_annotated = copy.deepcopy(scene_raw)
for node in scene_annotated["nodes"]:
    if node["id"] in MOCK_VISUAL_CONTEXT:
        node["visual_context"] = MOCK_VISUAL_CONTEXT[node["id"]]

scene_annotated["visual_context"] = {
    "format": "svgpatchlab.visual_context.v1",
    "annotated_node_count": len(MOCK_VISUAL_CONTEXT),
    "image_size": 384,
    "prompt_version": 1,
}

(OUT_DIR / "annotated_scene.json").write_text(
    json.dumps(scene_annotated, indent=2, sort_keys=True)
)
print("\n[STEP 2] Annotated scene (mock visual_context injected):")
for node in scene_annotated["nodes"]:
    vc = node.get("visual_context")
    if vc:
        print(f"  {node['id']:5s} <{node['tag']:<6}> labels={vc['labels']}  → \"{vc['summary']}\"")
    else:
        print(f"  {node['id']:5s} <{node['tag']:<6}> (no visual_context)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Show the full patch prompt (what the LLM sees)
# ─────────────────────────────────────────────────────────────────────────────
context_str = json.dumps(scene_annotated, indent=2, sort_keys=True)
prompt = patch_prompt(INSTRUCTION, "SVG DOM skeleton with visual context", context_str)

(OUT_DIR / "patch_prompt.txt").write_text(prompt)
print(f"\n[STEP 3] Full patch prompt written → {OUT_DIR}/patch_prompt.txt")
print(f"         (The LLM sees node n2 has labels=['text','glyph'] and")
print(f"          resolves that 'change the text' → target n2)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Grounded patch — what the LLM would produce given visual_context
# ─────────────────────────────────────────────────────────────────────────────
patch_dict = {
    "version": 1,
    "operations": [
        {
            "op": "set_attributes",
            "targets": ["n2"],     # resolved via visual_context.labels["text"]
            "attributes": {"fill": "purple"},
        }
    ],
}
patch_json_str = json.dumps(patch_dict, indent=2)

print(f"\n[STEP 4] Grounded patch (resolved via visual_context labels):")
print(f"  Instruction : {INSTRUCTION!r}")
print(f"  Matched     : n2 → labels contain 'text'")
print(f"  Patch       :")
for line in patch_json_str.splitlines():
    print(f"    {line}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Apply patch to SVG
# ─────────────────────────────────────────────────────────────────────────────
patch = Patch(
    version=1,
    operations=(
        PatchOperation(
            op="set_attributes",
            targets=("n2",),
            attributes=(("fill", "purple"),),
        ),
    ),
)
output_svg = apply_patch(case.source_svg, patch)

(OUT_DIR / "before.svg").write_text(case.source_svg)
(OUT_DIR / "after.svg").write_text(output_svg)

print(f"\n[STEP 5] SVG patch applied:")
print(f"  Before : {OUT_DIR}/before.svg  (text fill=#FFF)")
print(f"  After  : {OUT_DIR}/after.svg   (text fill=purple)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Verify the change
# ─────────────────────────────────────────────────────────────────────────────
scene_after = build_scene(output_svg)
n2_after = next(n for n in scene_after["nodes"] if n["id"] == "n2")
print(f"\n[VERIFY] n2 fill in output SVG: {n2_after['resolved_style'].get('fill')}")

print(f"\n{'='*68}")
print("  SUMMARY")
print(f"{'='*68}")
print("  Without visual_context: LLM had to guess n2 is 'text' from fill=#FFF")
print("  With visual_context   : LLM reads n2.labels=['text','glyph']")
print("                          → unambiguously targets n2 for fill=purple")
print(f"{'='*68}\n")
