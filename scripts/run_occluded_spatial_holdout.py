"""Run the frozen occlusion-isolating spatial-grounding holdout (version 2).

The suite asks whether raster-derived node descriptions add information beyond
geometry parsed from SVG source.  Each occluded case has four non-overlapping
red candidates and one grey occluder:

* the decoy is nominally in the named grid cell but fully covered;
* the target is nominally in another cell, but partial occlusion moves its
  visible contribution into the named cell; and
* two fillers occupy other cells.

The clean controls contain four non-overlapping red candidates.  Their target
is both nominally and visibly in the named cell.

Version 2 is deliberately fail-closed.  It fixes every path coordinate to
three characters, balances target and decoy node IDs within every label,
requires one unique source per case, checks clean non-overlap, verifies the
analytic/rendered disagreement with the real renderer, reserves an immutable
output root, streams complete audit records, and writes a provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svgpatchlab.architectures import create_architecture  # noqa: E402
from svgpatchlab.architectures.prompts import (  # noqa: E402
    PATCH_PROMPT_TEMPLATE_DIR,
    PATCH_PROMPT_VERSION,
)
from svgpatchlab.config import load_model_config  # noqa: E402
from svgpatchlab.core.patch import derive_patch  # noqa: E402
from svgpatchlab.core.scene import build_scene  # noqa: E402
from svgpatchlab.eval.metrics import evaluate_output  # noqa: E402
from svgpatchlab.eval.render import VisualStatsCache, ensure_renderer  # noqa: E402
from svgpatchlab.models import create_model  # noqa: E402
from svgpatchlab.models.base import ModelAdapter  # noqa: E402
from svgpatchlab.types import BenchmarkCase, ModelRequest, ModelResponse  # noqa: E402


SUITE_FORMAT = "svgpatchlab.occluded_spatial_holdout.v2"
DEFAULT_SEED = 20260726
DEFAULT_PER_FAMILY = 24
CANVAS = 100
FILL = "#d32f2f"
OCCLUDER_FILL = "#37474f"
STROKE = "#000000"
STROKE_WIDTH = "2"
VISUAL_STATS_RENDER_SIZE = 64
EVALUATION_RENDER_SIZE = 72
HOLDOUT_PATCH_PROMPT_VERSION = 4
VISUAL_STATS_CACHE_FORMAT = VisualStatsCache.FORMAT

ARMS = (
    "strict_skeleton_patch",
    "strict_analytic_stats_patch",
    "strict_visual_stats_patch",
)

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Layout:
    named_cell: str
    target_box: Box
    target_cover: Box
    decoy_box: Box
    decoy_cover: Box


# The target's nominal centre is top/bottom/center.  Covering one side moves
# its visible contribution into the named left/right cell.  The decoy occupies
# a separate, fully covered rectangle in that named cell, so target and decoy
# paint order can be randomized without changing visibility.
LAYOUTS = (
    Layout("top-right", (10, 8, 80, 10), (8, 6, 58, 14), (74, 22, 14, 8), (72, 20, 18, 12)),
    Layout("top-left", (10, 8, 80, 10), (34, 6, 58, 14), (12, 22, 14, 8), (10, 20, 18, 12)),
    Layout("bottom-right", (10, 82, 80, 10), (8, 80, 58, 14), (74, 70, 14, 8), (72, 68, 18, 12)),
    Layout("bottom-left", (10, 82, 80, 10), (34, 80, 58, 14), (12, 70, 14, 8), (10, 68, 18, 12)),
    Layout("right", (10, 43, 80, 10), (8, 41, 58, 14), (74, 56, 14, 8), (72, 54, 18, 12)),
    Layout("left", (10, 43, 80, 10), (34, 41, 58, 14), (12, 56, 14, 8), (10, 54, 18, 12)),
)

CELL_BOXES: dict[str, Box] = {
    "top-left": (12, 22, 10, 8),
    "top": (45, 22, 10, 8),
    "top-right": (78, 22, 10, 8),
    "left": (12, 46, 10, 8),
    "center": (45, 46, 10, 8),
    "right": (78, 46, 10, 8),
    "bottom-left": (12, 72, 10, 8),
    "bottom": (45, 72, 10, 8),
    "bottom-right": (78, 72, 10, 8),
}


@dataclass(frozen=True)
class OccludedCase:
    case: BenchmarkCase
    family: str
    named_cell: str
    layout_index: int
    replicate: int
    target_node_id: str
    decoy_node_id: str | None
    candidate_roles: tuple[str, ...]
    candidate_boxes: tuple[Box, ...]
    occluder_boxes: tuple[Box, ...]


class AuditRecordingModelAdapter(ModelAdapter):
    """Record reconstructable request and raw response metadata for every call."""

    def __init__(self, wrapped: ModelAdapter):
        self.wrapped = wrapped
        self.records: list[dict[str, Any]] = []

    @property
    def supports_images(self) -> bool:
        return bool(getattr(self.wrapped, "supports_images", False))

    def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        request_record = {
            "request_id": request.metadata.get("request_id"),
            "request_metadata": request.metadata,
            "prompt": request.prompt,
            "prompt_sha256": _sha256_text(request.prompt),
            "prompt_characters": len(request.prompt),
            "image_count": len(request.images),
            "image_sha256": [_sha256_text(image) for image in request.images],
            "response_schema_name": request.response_schema_name,
            "response_schema": request.response_schema,
        }
        try:
            response = self.wrapped.generate(request)
        except Exception as exc:
            self.records.append(
                {
                    **request_record,
                    "latency_seconds": time.perf_counter() - started,
                    "ok": False,
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        self.records.append(
            {
                **request_record,
                "latency_seconds": time.perf_counter() - started,
                "ok": True,
                "response_metadata": response.metadata,
                "response_text_sha256": _sha256_text(response.text),
                "response_text_characters": len(response.text),
            }
        )
        return response


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _case_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _fixed_coordinate(value: int) -> str:
    if not isinstance(value, int) or not 0 <= value <= 999:
        raise ValueError(f"holdout coordinate must be an integer in [0, 999]: {value}")
    return f"{value:03d}"


def _rect_path(box: Box) -> str:
    x, y, width, height = box
    if width <= 0 or height <= 0 or x + width > CANVAS or y + height > CANVAS:
        raise ValueError(f"invalid holdout rectangle: {box}")
    return (
        f"M{_fixed_coordinate(x)} {_fixed_coordinate(y)}"
        f"H{_fixed_coordinate(x + width)}"
        f"V{_fixed_coordinate(y + height)}"
        f"H{_fixed_coordinate(x)}Z"
    )


def _multi_rect_path(boxes: Sequence[Box]) -> str:
    if not boxes:
        raise ValueError("multi-rectangle path requires at least one box")
    return "".join(_rect_path(box) for box in boxes)


def _svg(shapes: Sequence[tuple[str, str]]) -> str:
    body = "".join(
        f'<path fill="{fill}" d="{path}"/>' for fill, path in shapes
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {CANVAS} {CANVAS}">{body}</svg>'
    )


def _answer(shapes: Sequence[tuple[str, str]], target_index: int) -> str:
    body = ""
    for index, (fill, path) in enumerate(shapes):
        if index == target_index:
            body += (
                f'<path fill="{fill}" stroke="{STROKE}" '
                f'stroke-width="{STROKE_WIDTH}" d="{path}"/>'
            )
        else:
            body += f'<path fill="{fill}" d="{path}"/>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {CANVAS} {CANVAS}">{body}</svg>'
    )


def _instruction(named_cell: str) -> str:
    return (
        "The following code is the SVG code for a diagram. Please generate an "
        "SVG code that draws a black line around the red shape in the "
        f"{named_cell} of the image."
    )


def _overlap_area(left: Box, right: Box) -> int:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    width = max(0, min(lx + lw, rx + rw) - max(lx, rx))
    height = max(0, min(ly + lh, ry + rh) - max(ly, ry))
    return width * height


def _position_for_box(box: Box) -> str:
    x, y, width, height = box
    cx = (x + width / 2.0) / CANVAS
    cy = (y + height / 2.0) / CANVAS
    row = "top" if cy < 1 / 3 else "bottom" if cy >= 2 / 3 else ""
    column = "left" if cx < 1 / 3 else "right" if cx >= 2 / 3 else ""
    if row and column:
        return f"{row}-{column}"
    return row or column or "center"


def _visible_box_after_covers(box: Box, covers: Sequence[Box]) -> Box | None:
    """Return exact bounds of uncovered unit cells for integer-aligned boxes."""

    x, y, width, height = box
    visible: list[tuple[int, int]] = []
    for px in range(x, x + width):
        for py in range(y, y + height):
            if not any(
                cx <= px < cx + cw and cy <= py < cy + ch
                for cx, cy, cw, ch in covers
            ):
                visible.append((px, py))
    if not visible:
        return None
    x0 = min(px for px, _ in visible)
    y0 = min(py for _, py in visible)
    x1 = max(px for px, _ in visible) + 1
    y1 = max(py for _, py in visible) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _safe_occluded_fillers(layout: Layout, rng: random.Random) -> list[Box]:
    blocked = (
        layout.target_box,
        layout.decoy_box,
        layout.target_cover,
        layout.decoy_cover,
    )
    candidates = [
        box
        for label, box in CELL_BOXES.items()
        if label != layout.named_cell
        and all(_overlap_area(box, other) == 0 for other in blocked)
    ]
    rng.shuffle(candidates)
    if len(candidates) < 2:
        raise ValueError(f"not enough safe fillers for {layout.named_cell}")
    return candidates[:2]


def _slot_schedule(seed: int, family: str, layout_index: int) -> list[int]:
    slots = list(range(4))
    random.Random(_case_seed(seed, family, layout_index, "target-slots")).shuffle(slots)
    return slots


def _build_occluded(
    seed: int,
    layout_index: int,
    replicate: int,
) -> OccludedCase:
    layout = LAYOUTS[layout_index]
    rng = random.Random(_case_seed(seed, "occluded", layout_index, replicate))
    target_slot = _slot_schedule(seed, "occluded", layout_index)[replicate]
    offset = 1 + (_case_seed(seed, "decoy-offset", layout_index) % 3)
    decoy_slot = (target_slot + offset) % 4
    fillers = _safe_occluded_fillers(layout, rng)
    rng.shuffle(fillers)

    roles: list[str | None] = [None] * 4
    boxes: list[Box | None] = [None] * 4
    roles[target_slot] = "target"
    boxes[target_slot] = layout.target_box
    roles[decoy_slot] = "decoy"
    boxes[decoy_slot] = layout.decoy_box
    for filler_index, slot in enumerate(
        index for index, role in enumerate(roles) if role is None
    ):
        roles[slot] = f"filler-{filler_index + 1}"
        boxes[slot] = fillers[filler_index]

    candidate_roles = tuple(str(role) for role in roles)
    candidate_boxes = tuple(box for box in boxes if box is not None)
    shapes = [(FILL, _rect_path(box)) for box in candidate_boxes]
    occluder_boxes = (layout.target_cover, layout.decoy_cover)
    shapes.append((OCCLUDER_FILL, _multi_rect_path(occluder_boxes)))
    source = _svg(shapes)
    answer = _answer(shapes, target_slot)
    case = BenchmarkCase(
        task="set_contour",
        emoji_id=(
            f"occluded-v2-{layout_index:02d}-{replicate:02d}-"
            f"{layout.named_cell}"
        ),
        instruction=_instruction(layout.named_cell),
        source_svg=source,
        answer_svg=answer,
        query_path=Path("."),
        answer_path=Path("."),
    )
    return OccludedCase(
        case=case,
        family="occluded",
        named_cell=layout.named_cell,
        layout_index=layout_index,
        replicate=replicate,
        target_node_id=f"n{target_slot + 1}",
        decoy_node_id=f"n{decoy_slot + 1}",
        candidate_roles=candidate_roles,
        candidate_boxes=candidate_boxes,
        occluder_boxes=occluder_boxes,
    )


def _build_clean(
    seed: int,
    layout_index: int,
    replicate: int,
) -> OccludedCase:
    named_cell = LAYOUTS[layout_index].named_cell
    rng = random.Random(_case_seed(seed, "clean", layout_index, replicate))
    target_slot = _slot_schedule(seed, "clean", layout_index)[replicate]
    target_box = CELL_BOXES[named_cell]
    fillers = [
        box for label, box in CELL_BOXES.items() if label != named_cell
    ]
    rng.shuffle(fillers)
    fillers = fillers[:3]

    roles: list[str | None] = [None] * 4
    boxes: list[Box | None] = [None] * 4
    roles[target_slot] = "target"
    boxes[target_slot] = target_box
    for filler_index, slot in enumerate(
        index for index, role in enumerate(roles) if role is None
    ):
        roles[slot] = f"filler-{filler_index + 1}"
        boxes[slot] = fillers[filler_index]

    candidate_roles = tuple(str(role) for role in roles)
    candidate_boxes = tuple(box for box in boxes if box is not None)
    shapes = [(FILL, _rect_path(box)) for box in candidate_boxes]
    source = _svg(shapes)
    answer = _answer(shapes, target_slot)
    case = BenchmarkCase(
        task="set_contour",
        emoji_id=f"clean-v2-{layout_index:02d}-{replicate:02d}-{named_cell}",
        instruction=_instruction(named_cell),
        source_svg=source,
        answer_svg=answer,
        query_path=Path("."),
        answer_path=Path("."),
    )
    return OccludedCase(
        case=case,
        family="clean",
        named_cell=named_cell,
        layout_index=layout_index,
        replicate=replicate,
        target_node_id=f"n{target_slot + 1}",
        decoy_node_id=None,
        candidate_roles=candidate_roles,
        candidate_boxes=candidate_boxes,
        occluder_boxes=(),
    )


def build_cases(
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
) -> list[OccludedCase]:
    if per_family != DEFAULT_PER_FAMILY:
        raise ValueError(
            f"the frozen v2 suite requires --per-family {DEFAULT_PER_FAMILY}"
        )
    occluded = [
        _build_occluded(seed, layout_index, replicate)
        for layout_index in range(len(LAYOUTS))
        for replicate in range(4)
    ]
    clean = [
        _build_clean(seed, layout_index, replicate)
        for layout_index in range(len(LAYOUTS))
        for replicate in range(4)
    ]
    cases = [*occluded, *clean]
    random.Random(_case_seed(seed, "case-order")).shuffle(cases)
    if len({item.case.source_svg for item in cases}) != len(cases):
        raise RuntimeError("v2 construction generated duplicate source SVGs")
    return cases


def _patch_targets(patch: Any | None) -> set[str]:
    if patch is None:
        return set()
    operations = (
        patch.operations
        if hasattr(patch, "operations")
        else patch.get("operations", [])
    )
    targets: set[str] = set()
    for operation in operations:
        values = (
            operation.targets
            if hasattr(operation, "targets")
            else operation.get("targets", [])
        )
        targets.update(values)
    return targets


def _candidate_nodes(scene: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in scene["nodes"]
        if node["id"] in {"n1", "n2", "n3", "n4"}
    ]


def _assert_balanced_ids(cases: Sequence[OccludedCase], family: str) -> None:
    rows = [item for item in cases if item.family == family]
    expected = Counter({f"n{index}": len(rows) // 4 for index in range(1, 5)})
    targets = Counter(item.target_node_id for item in rows)
    if targets != expected:
        raise ValueError(f"{family} target node IDs are not balanced: {targets}")
    if family == "occluded":
        decoys = Counter(item.decoy_node_id for item in rows)
        if decoys != expected:
            raise ValueError(f"occluded decoy node IDs are not balanced: {decoys}")

    for named_cell in {item.named_cell for item in rows}:
        label_rows = [item for item in rows if item.named_cell == named_cell]
        if Counter(item.target_node_id for item in label_rows) != Counter(
            {f"n{index}": 1 for index in range(1, 5)}
        ):
            raise ValueError(f"{family}/{named_cell} target IDs are not balanced")
        if family == "occluded" and Counter(
            item.decoy_node_id for item in label_rows
        ) != Counter({f"n{index}": 1 for index in range(1, 5)}):
            raise ValueError(f"occluded/{named_cell} decoy IDs are not balanced")


def validate_static_suite(cases: Sequence[OccludedCase]) -> dict[str, Any]:
    if PATCH_PROMPT_VERSION != HOLDOUT_PATCH_PROMPT_VERSION:
        raise ValueError(
            f"holdout freezes patch prompt v{HOLDOUT_PATCH_PROMPT_VERSION}, "
            f"but active version is v{PATCH_PROMPT_VERSION}"
        )
    if len(cases) != DEFAULT_PER_FAMILY * 2:
        raise ValueError(f"expected 48 cases, got {len(cases)}")
    family_counts = Counter(item.family for item in cases)
    if family_counts != Counter(
        {"occluded": DEFAULT_PER_FAMILY, "clean": DEFAULT_PER_FAMILY}
    ):
        raise ValueError(f"unexpected family counts: {family_counts}")
    source_count = len({item.case.source_svg for item in cases})
    if source_count != len(cases):
        raise ValueError("every v2 case must have a unique source SVG")
    _assert_balanced_ids(cases, "occluded")
    _assert_balanced_ids(cases, "clean")

    red_lengths: set[int] = set()
    occluder_lengths: set[int] = set()
    for item in cases:
        if len(item.candidate_roles) != 4 or len(item.candidate_boxes) != 4:
            raise ValueError(f"{item.case.case_id}: expected four candidates")
        if item.candidate_roles.count("target") != 1:
            raise ValueError(f"{item.case.case_id}: target role is not unique")
        target_index = item.candidate_roles.index("target")
        if item.target_node_id != f"n{target_index + 1}":
            raise ValueError(f"{item.case.case_id}: target ID/role mismatch")

        for left_index, left in enumerate(item.candidate_boxes):
            for right in item.candidate_boxes[left_index + 1 :]:
                if _overlap_area(left, right):
                    raise ValueError(
                        f"{item.case.case_id}: candidate boxes overlap"
                    )

        target_box = item.candidate_boxes[target_index]
        if item.family == "occluded":
            if item.decoy_node_id is None:
                raise ValueError(f"{item.case.case_id}: missing decoy ID")
            decoy_index = item.candidate_roles.index("decoy")
            if item.decoy_node_id != f"n{decoy_index + 1}":
                raise ValueError(f"{item.case.case_id}: decoy ID/role mismatch")
            decoy_box = item.candidate_boxes[decoy_index]
            target_visible = _visible_box_after_covers(
                target_box, item.occluder_boxes
            )
            decoy_visible = _visible_box_after_covers(
                decoy_box, item.occluder_boxes
            )
            if _position_for_box(target_box) == item.named_cell:
                raise ValueError(f"{item.case.case_id}: target is nominally correct")
            if target_visible is None or _position_for_box(target_visible) != item.named_cell:
                raise ValueError(f"{item.case.case_id}: target visibility is wrong")
            if _position_for_box(decoy_box) != item.named_cell:
                raise ValueError(f"{item.case.case_id}: decoy nominal cell is wrong")
            if decoy_visible is not None:
                raise ValueError(f"{item.case.case_id}: decoy is not fully covered")
        else:
            if item.decoy_node_id is not None or item.occluder_boxes:
                raise ValueError(f"{item.case.case_id}: clean case has an occluder")
            if _position_for_box(target_box) != item.named_cell:
                raise ValueError(f"{item.case.case_id}: clean target cell is wrong")

        allowed_named_indices = {target_index}
        if item.family == "occluded":
            allowed_named_indices.add(item.candidate_roles.index("decoy"))
        for index, box in enumerate(item.candidate_boxes):
            if (
                index not in allowed_named_indices
                and _position_for_box(box) == item.named_cell
            ):
                raise ValueError(
                    f"{item.case.case_id}: non-target shares named cell"
                )

        patch = derive_patch(item.case.source_svg, item.case.answer_svg)
        if _patch_targets(patch) != {item.target_node_id}:
            raise ValueError(f"{item.case.case_id}: answer target mismatch")
        if len(patch.operations) != 1:
            raise ValueError(f"{item.case.case_id}: answer is not one operation")
        operation = patch.operations[0]
        if operation.attributes_dict != {
            "stroke": STROKE,
            "stroke-width": STROKE_WIDTH,
        }:
            raise ValueError(f"{item.case.case_id}: answer attributes mismatch")

        scene = build_scene(item.case.source_svg)
        candidates = _candidate_nodes(scene)
        if len(candidates) != 4:
            raise ValueError(f"{item.case.case_id}: scene candidate count changed")
        for node in candidates:
            if node["tag"] != "path":
                raise ValueError(f"{item.case.case_id}: non-path candidate")
            if node.get("attributes") != {"fill": FILL}:
                raise ValueError(f"{item.case.case_id}: candidate attributes leak")
            protected = node.get("protected_geometry", {}).get("d", {})
            if set(protected) != {"sha256", "characters"}:
                raise ValueError(f"{item.case.case_id}: path is not SHA-only")
            red_lengths.add(int(protected["characters"]))
        hashes = {
            node["protected_geometry"]["d"]["sha256"] for node in candidates
        }
        if len(hashes) != len(candidates):
            raise ValueError(f"{item.case.case_id}: duplicate candidate geometry")

        other_paths = [
            node
            for node in scene["nodes"]
            if node["tag"] == "path" and node["id"] not in {n["id"] for n in candidates}
        ]
        if item.family == "occluded":
            if len(other_paths) != 1:
                raise ValueError(f"{item.case.case_id}: expected one occluder")
            occluder_lengths.add(
                int(other_paths[0]["protected_geometry"]["d"]["characters"])
            )
        elif other_paths:
            raise ValueError(f"{item.case.case_id}: clean case has extra paths")

    if len(red_lengths) != 1:
        raise ValueError(f"label-correlated red path lengths remain: {red_lengths}")
    if len(occluder_lengths) != 1:
        raise ValueError(
            f"label-correlated occluder path lengths remain: {occluder_lengths}"
        )

    return {
        "cases": len(cases),
        "families": dict(sorted(family_counts.items())),
        "unique_source_svgs": source_count,
        "one_unique_source_per_case": True,
        "fixed_width_coordinates": True,
        "equal_red_path_character_counts": True,
        "equal_occluder_path_character_counts": True,
        "balanced_target_node_ids": True,
        "balanced_decoy_node_ids": True,
        "balanced_ids_within_each_label": True,
        "candidate_pairwise_non_overlap": True,
        "clean_pairwise_non_overlap": True,
        "clean_has_no_occluder": True,
        "raw_path_coordinates_hidden_from_model": True,
        "protected_path_sha256_only": True,
        "red_path_characters": next(iter(red_lengths)),
        "occluder_path_characters": next(iter(occluder_lengths)),
    }


StatsProvider = Callable[[str], dict[str, dict[str, Any]]]


def verify_construction(
    cases: Sequence[OccludedCase],
    *,
    cache_dir: str | Path = ".cache/visual_stats_occluded_v2",
    analytic_provider: StatsProvider | None = None,
    visual_provider: StatsProvider | None = None,
) -> dict[str, Any]:
    """Verify actual analytic and rendered descriptors for every case."""

    if analytic_provider is None:
        from svgpatchlab.core.geometry import node_analytic_stats

        analytic_provider = node_analytic_stats
    if visual_provider is None:
        cache = VisualStatsCache(str(cache_dir))
        visual_provider = lambda svg: cache.get_or_compute(  # noqa: E731
            svg, size=VISUAL_STATS_RENDER_SIZE
        )

    report: dict[str, Any] = {
        "occluded_verified": 0,
        "clean_verified": 0,
        "problems": [],
    }
    for item in cases:
        analytic = analytic_provider(item.case.source_svg)
        rendered = visual_provider(item.case.source_svg)
        candidate_ids = [f"n{index}" for index in range(1, 5)]
        analytic_named = [
            node_id
            for node_id in candidate_ids
            if (analytic.get(node_id) or {}).get("position") == item.named_cell
        ]
        rendered_named = [
            node_id
            for node_id in candidate_ids
            if (rendered.get(node_id) or {}).get("visible") is not False
            and (rendered.get(node_id) or {}).get("position") == item.named_cell
        ]
        target_visual = rendered.get(item.target_node_id) or {}
        if item.family == "occluded":
            decoy_visual = rendered.get(item.decoy_node_id or "") or {}
            ok = (
                analytic_named == [item.decoy_node_id]
                and rendered_named == [item.target_node_id]
                and target_visual.get("visible") is not False
                and decoy_visual.get("visible") is False
            )
            if ok:
                report["occluded_verified"] += 1
            else:
                report["problems"].append(
                    {
                        "case_id": item.case.case_id,
                        "family": item.family,
                        "named_cell": item.named_cell,
                        "target_node_id": item.target_node_id,
                        "decoy_node_id": item.decoy_node_id,
                        "analytic_named_candidates": analytic_named,
                        "rendered_named_candidates": rendered_named,
                        "target_visual": target_visual,
                        "decoy_visual": decoy_visual,
                    }
                )
        else:
            all_visible = all(
                (rendered.get(node_id) or {}).get("visible") is not False
                for node_id in candidate_ids
            )
            ok = (
                analytic_named == [item.target_node_id]
                and rendered_named == [item.target_node_id]
                and all_visible
            )
            if ok:
                report["clean_verified"] += 1
            else:
                report["problems"].append(
                    {
                        "case_id": item.case.case_id,
                        "family": item.family,
                        "named_cell": item.named_cell,
                        "target_node_id": item.target_node_id,
                        "analytic_named_candidates": analytic_named,
                        "rendered_named_candidates": rendered_named,
                        "all_candidates_visible": all_visible,
                    }
                )
    return report


def _prompt_provenance() -> dict[str, Any]:
    template = (
        PATCH_PROMPT_TEMPLATE_DIR / f"patch_v{HOLDOUT_PATCH_PROMPT_VERSION}.txt"
    )
    return {
        "version": HOLDOUT_PATCH_PROMPT_VERSION,
        "identifier": f"svgpatchlab.patch.v{HOLDOUT_PATCH_PROMPT_VERSION}",
        "template": template.name,
        "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
    }


def _redact_config(value: Any, key: str = "") -> Any:
    sensitive = ("api_key", "apikey", "password", "secret", "token")
    if any(part in key.lower() for part in sensitive):
        return "<redacted>"
    if isinstance(value, dict):
        return {name: _redact_config(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "tracked_worktree_dirty": bool(status) if status is not None else None,
    }


def _model_config_provenance(
    model_config_path: str | Path,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    path = Path(model_config_path).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "resolved_redacted": _redact_config(model_config),
    }


def _case_manifest(item: OccludedCase) -> dict[str, Any]:
    gold_patch = derive_patch(
        item.case.source_svg, item.case.answer_svg
    ).to_dict()
    return {
        "case_id": item.case.case_id,
        "family": item.family,
        "named_cell": item.named_cell,
        "layout_index": item.layout_index,
        "replicate": item.replicate,
        "instruction": item.case.instruction,
        "instruction_sha256": _sha256_text(item.case.instruction),
        "source_svg": item.case.source_svg,
        "source_sha256": _sha256_text(item.case.source_svg),
        "answer_svg": item.case.answer_svg,
        "answer_sha256": _sha256_text(item.case.answer_svg),
        "target_node_id": item.target_node_id,
        "decoy_node_id": item.decoy_node_id,
        "candidate_roles": list(item.candidate_roles),
        "candidate_boxes": [list(box) for box in item.candidate_boxes],
        "occluder_boxes": [list(box) for box in item.occluder_boxes],
        "gold_patch": gold_patch,
    }


def build_manifest(
    cases: Sequence[OccludedCase],
    *,
    seed: int,
    model_config_path: str | Path,
    model_config: dict[str, Any],
    arms: Sequence[str],
    design: dict[str, Any],
    verification: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "format": SUITE_FORMAT,
        "created_at_utc": created_at
        or datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model_config": _model_config_provenance(
            model_config_path, model_config
        ),
        "patch_prompt": _prompt_provenance(),
        "rendering": {
            "visual_stats_render_size": VISUAL_STATS_RENDER_SIZE,
            "evaluation_render_size": EVALUATION_RENDER_SIZE,
            "visual_stats_cache_format": VISUAL_STATS_CACHE_FORMAT,
        },
        "arm_order": list(arms),
        "case_order": [item.case.case_id for item in cases],
        "case_count": len(cases),
        "unique_source_count": len({item.case.source_svg for item in cases}),
        "design": design,
        "verification": verification,
        "code": {
            "script": str(script_path.relative_to(REPO_ROOT)),
            "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
            "git": _git_provenance(),
        },
        "cases": [_case_manifest(item) for item in cases],
    }


def _json_bytes(payload: Any, *, pretty: bool) -> bytes:
    if pretty:
        text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    else:
        text = json.dumps(payload, sort_keys=True, default=str) + "\n"
    return text.encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, _json_bytes(payload, pretty=True))


def _atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write(path, payload.encode("utf-8"))


def _stream_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(_json_bytes(record, pretty=False))
    handle.flush()
    os.fsync(handle.fileno())


def reserve_output_root(output_root: Path) -> None:
    """Reserve a fresh root; existing experiment artifacts are immutable."""

    output_root.mkdir(parents=True, exist_ok=False)


def _architecture_for_arm(name: str, output_root: Path) -> Any:
    if name == "strict_visual_stats_patch":
        return create_architecture(
            name,
            cache_dir=str(output_root / ".visual-stats-cache"),
            render_size=VISUAL_STATS_RENDER_SIZE,
        )
    return create_architecture(name)


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def _metrics_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    calls = [
        call
        for record in records
        for call in record.get("model_call_details", [])
    ]
    return {
        "cases": len(records),
        "target_exact_rate": _mean(
            float(bool(record["target_exact"])) for record in records
        ),
        "gold_patch_exact_rate": _mean(
            float(bool(record["gold_patch_exact"])) for record in records
        ),
        "valid_output_rate": _mean(
            float(bool(record["metrics"].get("valid_output")))
            for record in records
        ),
        "decoy_selected_rate": _mean(
            float(bool(record["picked_decoy"]))
            for record in records
            if record["decoy_node_id"] is not None
        ),
        "mean_failure_aware_mse": _mean(
            float(record["metrics"]["failure_aware_mse"])
            for record in records
            if record["metrics"].get("failure_aware_mse") is not None
        ),
        "mean_wall_seconds": _mean(
            float(record["wall_seconds"]) for record in records
        ),
        "mean_model_latency_seconds": _mean(
            float(call["latency_seconds"]) for call in calls
        ),
        "model_calls": len(calls),
        "prompt_tokens": sum(
            int(
                call.get("response_metadata", {})
                .get("usage", {})
                .get("prompt_tokens", 0)
            )
            for call in calls
        ),
        "completion_tokens": sum(
            int(
                call.get("response_metadata", {})
                .get("usage", {})
                .get("completion_tokens", 0)
            )
            for call in calls
        ),
    }


def run_arm(
    name: str,
    cases: Sequence[OccludedCase],
    model_config: dict[str, Any],
    output_root: Path,
    *,
    architecture_factory: Callable[[str, Path], Any] | None = None,
    model_factory: Callable[[dict[str, Any]], ModelAdapter] | None = None,
    evaluator: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    architecture_factory = architecture_factory or _architecture_for_arm
    model_factory = model_factory or create_model
    evaluator = evaluator or evaluate_output
    architecture = architecture_factory(name, output_root)
    model = AuditRecordingModelAdapter(model_factory(model_config))
    arm_dir = output_root / name
    arm_dir.mkdir(parents=True, exist_ok=False)
    output_dir = arm_dir / "outputs"
    output_dir.mkdir(exist_ok=False)
    results_path = arm_dir / "results.jsonl"
    records: list[dict[str, Any]] = []

    with results_path.open("xb") as handle:
        for item in cases:
            call_offset = len(model.records)
            started = time.perf_counter()
            try:
                result = architecture.run(item.case, model)
            except Exception as exc:
                fatal = {
                    "case_id": item.case.case_id,
                    "architecture": name,
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                    "model_call_details": model.records[call_offset:],
                }
                _atomic_write_json(arm_dir / "fatal.json", fatal)
                raise
            wall_seconds = time.perf_counter() - started
            gold_patch = derive_patch(
                item.case.source_svg, item.case.answer_svg
            )
            predicted_targets = _patch_targets(result.patch)
            gold_targets = _patch_targets(gold_patch)
            metrics = evaluator(
                item.case.source_svg,
                item.case.answer_svg,
                result.output_svg,
                result.patch,
                render=True,
                render_size=EVALUATION_RENDER_SIZE,
            )
            patch_payload = (
                result.patch.to_dict() if result.patch is not None else None
            )
            output_path: str | None = None
            output_sha256: str | None = None
            if result.output_svg is not None:
                relative = Path("outputs") / f"{item.case.emoji_id}.svg"
                _atomic_write_text(arm_dir / relative, result.output_svg)
                output_path = relative.as_posix()
                output_sha256 = _sha256_text(result.output_svg)
            record = {
                "case_id": item.case.case_id,
                "family": item.family,
                "named_cell": item.named_cell,
                "layout_index": item.layout_index,
                "replicate": item.replicate,
                "architecture": name,
                "target_node_id": item.target_node_id,
                "decoy_node_id": item.decoy_node_id,
                "candidate_roles": list(item.candidate_roles),
                "source_sha256": _sha256_text(item.case.source_svg),
                "answer_sha256": _sha256_text(item.case.answer_svg),
                "error": result.error,
                "model_calls": result.model_calls,
                "patch": patch_payload,
                "gold_patch": gold_patch.to_dict(),
                "raw_responses": result.raw_responses,
                "architecture_details": result.details,
                "predicted_targets": sorted(predicted_targets),
                "gold_targets": sorted(gold_targets),
                "target_exact": predicted_targets == gold_targets,
                "gold_patch_exact": metrics.get("gold_patch_exact"),
                "picked_decoy": (
                    item.decoy_node_id in predicted_targets
                    if item.decoy_node_id is not None
                    else False
                ),
                "output_svg": result.output_svg,
                "output_svg_path": output_path,
                "output_svg_sha256": output_sha256,
                "wall_seconds": wall_seconds,
                "model_call_details": model.records[call_offset:],
                "metrics": metrics,
            }
            records.append(record)
            _stream_record(handle, record)

    summary = {
        "format": SUITE_FORMAT,
        "architecture": name,
        "overall": _metrics_summary(records),
        "by_family": {
            family: _metrics_summary(
                [record for record in records if record["family"] == family]
            )
            for family in ("occluded", "clean")
        },
    }
    _atomic_write_json(arm_dir / "summary.json", summary)
    return records, summary


def _sign_test_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2.0**discordant))


def _paired_group(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    left_arm: str,
    right_arm: str,
) -> dict[str, Any]:
    left_by_id = {record["case_id"]: record for record in left}
    right_by_id = {record["case_id"]: record for record in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("paired arms used different cases")
    case_ids = sorted(left_by_id)

    def correctness(field: str) -> dict[str, Any]:
        left_only = right_only = both = neither = 0
        for case_id in case_ids:
            left_ok = bool(left_by_id[case_id][field])
            right_ok = bool(right_by_id[case_id][field])
            if left_ok and not right_ok:
                left_only += 1
            elif right_ok and not left_ok:
                right_only += 1
            elif left_ok:
                both += 1
            else:
                neither += 1
        return {
            "left_only_wins": left_only,
            "right_only_wins": right_only,
            "both_correct": both,
            "neither_correct": neither,
            "discordant": left_only + right_only,
            "two_sided_sign_test_p": _sign_test_p(left_only, right_only),
        }

    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "cases": len(case_ids),
        "target_exact": correctness("target_exact"),
        "gold_patch_exact": correctness("gold_patch_exact"),
    }


def paired_summary(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    left_arm: str,
    right_arm: str,
) -> dict[str, Any]:
    families = {record["family"] for record in left}
    if families != {record["family"] for record in right}:
        raise ValueError("paired arms used different families")
    return {
        "overall": _paired_group(
            left, right, left_arm=left_arm, right_arm=right_arm
        ),
        "by_family": {
            family: _paired_group(
                [record for record in left if record["family"] == family],
                [record for record in right if record["family"] == family],
                left_arm=left_arm,
                right_arm=right_arm,
            )
            for family in sorted(families)
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        default="configs/models/qwen2.5-7b-ollama-constrained.json",
    )
    parser.add_argument(
        "--output-root",
        default="runs/occluded-spatial-holdout-v2",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-family", type=int, default=DEFAULT_PER_FAMILY)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(set(args.arms)) != len(args.arms):
        raise SystemExit("arm names must be unique")
    unknown = [arm for arm in args.arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unsupported holdout arms: {unknown}")

    cases = build_cases(args.seed, args.per_family)
    design = validate_static_suite(cases)
    ensure_renderer()

    if args.verify_only:
        verification = verify_construction(cases)
        print(json.dumps({"design": design, "verification": verification}, indent=2))
        if verification["problems"]:
            raise SystemExit("suite construction failed rendered verification")
        return 0

    model_config = load_model_config(args.model_config)
    output_root = Path(args.output_root)
    reserve_output_root(output_root)
    verification = verify_construction(
        cases,
        cache_dir=output_root / ".visual-stats-cache",
    )
    _atomic_write_json(output_root / "verification.json", verification)
    if verification["problems"]:
        raise SystemExit("suite construction failed rendered verification")

    manifest = build_manifest(
        cases,
        seed=args.seed,
        model_config_path=args.model_config,
        model_config=model_config,
        arms=args.arms,
        design=design,
        verification=verification,
    )
    _atomic_write_json(output_root / "manifest.json", manifest)
    _atomic_write_json(
        output_root / "status.json",
        {
            "format": SUITE_FORMAT,
            "state": "running",
            "started_at_utc": manifest["created_at_utc"],
            "completed_arms": [],
        },
    )

    arm_records: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    completed: list[str] = []
    try:
        for arm in args.arms:
            records, summary = run_arm(
                arm, cases, model_config, output_root
            )
            arm_records[arm] = records
            arm_summaries[arm] = summary
            completed.append(arm)
            _atomic_write_json(
                output_root / "status.json",
                {
                    "format": SUITE_FORMAT,
                    "state": "running",
                    "started_at_utc": manifest["created_at_utc"],
                    "completed_arms": completed,
                },
            )
    except Exception as exc:
        _atomic_write_json(
            output_root / "status.json",
            {
                "format": SUITE_FORMAT,
                "state": "failed",
                "started_at_utc": manifest["created_at_utc"],
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_arms": completed,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise

    paired: dict[str, Any] = {}
    for left_index, left_arm in enumerate(args.arms):
        for right_arm in args.arms[left_index + 1 :]:
            key = f"{left_arm}__vs__{right_arm}"
            paired[key] = paired_summary(
                arm_records[left_arm],
                arm_records[right_arm],
                left_arm=left_arm,
                right_arm=right_arm,
            )

    summary = {
        "format": SUITE_FORMAT,
        "seed": args.seed,
        "manifest_sha256": hashlib.sha256(
            (output_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "design": design,
        "verification": verification,
        "arm_order": list(args.arms),
        "arms": arm_summaries,
        "paired": paired,
    }
    _atomic_write_json(output_root / "summary.json", summary)
    _atomic_write_json(
        output_root / "status.json",
        {
            "format": SUITE_FORMAT,
            "state": "complete",
            "started_at_utc": manifest["created_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_arms": completed,
            "summary_sha256": hashlib.sha256(
                (output_root / "summary.json").read_bytes()
            ).hexdigest(),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
