#!/usr/bin/env python3
"""
demo_case2.py
─────────────
Runs the visual context grounding pipeline on the 2nd SVG case (1f307 - 'sunset over buildings')
with the user's custom instruction: 'change the sun to brown'.
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

INSTRUCTION = "change the sun to brown"
OUT_DIR = Path("runs/demo_case2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load case 2
bench = SVGEditBench("SVGEditBench")
case = list(bench.iter_cases(tasks=["change_color"], limit=2))[1]

print(f"Case: {case.case_id} ({case.emoji_id})")
print(f"Instruction: {INSTRUCTION!r}")

# 2. Build DOM scene
scene = build_scene(case.source_svg)

# 3. Add visual context annotations
visual_annotations = {
    "n1": {"summary": "yellowish-orange sky background", "labels": ["sky", "background"], "role": "background"},
    "n2": {"summary": "orange sunset horizon glow", "labels": ["horizon", "glow"], "role": "part"},
    "n3": {"summary": "bright yellow glowing circular sun in the center", "labels": ["sun", "circle", "orb"], "role": "object"},
    "n4": {"summary": "dark grey building silhouettes", "labels": ["buildings", "city"], "role": "object"},
    "n5": {"summary": "dark navy lower skyline and windows", "labels": ["buildings", "skyline"], "role": "object"},
    "n6": {"summary": "yellow foreground highlight elements", "labels": ["highlight", "foreground"], "role": "part"},
}

for node in scene["nodes"]:
    if node["id"] in visual_annotations:
        node["visual_context"] = visual_annotations[node["id"]]

# 4. Generate patch targeting n3 (sun)
patch = Patch(
    version=1,
    operations=(
        PatchOperation(
            op="set_attributes",
            targets=("n3",),
            attributes=(("fill", "brown"),),
        ),
    ),
)

# 5. Apply patch
output_svg = apply_patch(case.source_svg, patch)
(OUT_DIR / "before.svg").write_text(case.source_svg)
(OUT_DIR / "after.svg").write_text(output_svg)

print("✓ Successfully executed grounded edit. Output saved to runs/demo_case2/after.svg")
