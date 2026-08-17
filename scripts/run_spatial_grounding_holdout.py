from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svgpatchlab.architectures.patching import (
    SkeletonPatchArchitecture,
    VisualStatsPatchArchitecture,
)
from svgpatchlab.architectures.prompts import (
    PATCH_PROMPT_TEMPLATE_DIR,
    PATCH_PROMPT_VERSION,
)
from svgpatchlab.config import load_model_config
from svgpatchlab.core.patch import derive_patch
from svgpatchlab.core.scene import build_scene
from svgpatchlab.core.xml import parse_svg, serialize_svg
from svgpatchlab.eval.metrics import evaluate_output
from svgpatchlab.eval.render import ensure_renderer
from svgpatchlab.models import RecordingModelAdapter, create_model
from svgpatchlab.types import BenchmarkCase


SUITE_FORMAT = "svgpatchlab.spatial_grounding_holdout.v1"
DEFAULT_SEED = 20260725
FILL = "#ff0000"
STROKE = "#000000"
STROKE_WIDTH = "2"
POSITION_LAYOUTS = 3
SIZE_LAYOUTS = 3
VISUAL_STATS_RENDER_SIZE = 64
EVALUATION_RENDER_SIZE = 72
HOLDOUT_PATCH_PROMPT_VERSION = 4

POSITION_CENTERS = {
    "top-left": (16, 16),
    "top": (50, 16),
    "top-right": (84, 16),
    "left": (16, 50),
    "center": (50, 50),
    "right": (84, 50),
    "bottom-left": (16, 84),
    "bottom": (50, 84),
    "bottom-right": (84, 84),
}
SIZE_LEVELS = {
    "smallest": 6,
    "second-smallest": 12,
    "second-largest": 18,
    "largest": 26,
}
SIZE_CENTERS = (
    (18, 18),
    (82, 18),
    (18, 82),
    (82, 82),
)
POSITION_DIMENSIONS = (
    (8, 10),
    (10, 8),
    (8, 12),
    (12, 8),
    (10, 12),
    (12, 10),
    (10, 10),
)
SIZE_DIMENSIONS = {
    "smallest": ((6, 8), (8, 6), (8, 8)),
    "second-smallest": ((10, 12), (12, 10), (12, 12)),
    "second-largest": ((16, 20), (18, 18), (20, 16)),
    "largest": ((24, 28), (26, 26), (28, 24)),
}
JITTER_OFFSETS = (
    (-2, -2),
    (-2, 1),
    (-1, 2),
    (0, -1),
    (0, 2),
    (1, -2),
    (1, 1),
    (2, -1),
    (2, 2),
)
_RECT_PATH = re.compile(
    r"^M(?P<x0>\d{2}) (?P<y0>\d{2})"
    r"H(?P<x1>\d{2})V(?P<y1>\d{2})H(?P=x0)Z$"
)


@dataclass(frozen=True)
class HoldoutCase:
    case: BenchmarkCase
    family: str
    descriptor: str
    layout_index: int
    target_node_id: str
    dom_order: tuple[str, ...]


def _rect_path(
    center: tuple[int, int],
    width: int,
    height: int,
) -> str:
    cx, cy = center
    if width % 2 or height % 2:
        raise ValueError("holdout rectangle dimensions must be even")
    x0, y0 = cx - width // 2, cy - height // 2
    x1, y1 = cx + width // 2, cy + height // 2
    coordinates = (x0, y0, x1, y1)
    if any(value < 0 or value > 99 for value in coordinates):
        raise ValueError("holdout paths require two-digit coordinates")
    return f"M{x0:02d} {y0:02d}H{x1:02d}V{y1:02d}H{x0:02d}Z"


def _path_box(path: str) -> tuple[int, int, int, int]:
    match = _RECT_PATH.fullmatch(path)
    if match is None:
        raise ValueError(f"unexpected holdout path syntax: {path}")
    x0 = int(match.group("x0"))
    y0 = int(match.group("y0"))
    x1 = int(match.group("x1"))
    y1 = int(match.group("y1"))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid holdout rectangle: {path}")
    return x0, y0, x1 - x0, y1 - y0


def _position_for_box(box: tuple[int, int, int, int]) -> str:
    x, y, width, height = box
    cx = (x + width / 2.0) / 100.0
    cy = (y + height / 2.0) / 100.0
    columns = ("left", "", "right")
    rows = ("top", "", "bottom")
    row = rows[min(int(cy * 3), 2)]
    column = columns[min(int(cx * 3), 2)]
    if row and column:
        return f"{row}-{column}"
    return row or column or "center"


def _case_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _prompt_provenance() -> dict[str, Any]:
    template = (
        PATCH_PROMPT_TEMPLATE_DIR
        / f"patch_v{HOLDOUT_PATCH_PROMPT_VERSION}.txt"
    )
    return {
        "version": HOLDOUT_PATCH_PROMPT_VERSION,
        "identifier": (
            f"svgpatchlab.patch.v{HOLDOUT_PATCH_PROMPT_VERSION}"
        ),
        "template": template.name,
        "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
    }


def _case_dom_order(
    descriptors: Sequence[str],
    descriptor: str,
    target_index: int,
    *,
    seed: int,
    used: set[tuple[str, ...]],
) -> tuple[str, ...]:
    remaining = [item for item in descriptors if item != descriptor]
    rng = random.Random(seed)
    for _ in range(10_000):
        rng.shuffle(remaining)
        order = tuple(
            remaining[:target_index]
            + [descriptor]
            + remaining[target_index:]
        )
        if order not in used:
            used.add(order)
            return order
    raise RuntimeError("could not generate a unique deterministic DOM order")


def _jittered_centers(
    centers: Sequence[tuple[int, int]],
    *,
    seed: int,
) -> list[tuple[int, int]]:
    offsets = list(JITTER_OFFSETS)
    random.Random(seed).shuffle(offsets)
    return [
        (center[0] + offset[0], center[1] + offset[1])
        for center, offset in zip(centers, offsets)
    ]


def _source_svg(
    paths_by_descriptor: dict[str, str],
    dom_order: Sequence[str],
) -> str:
    paths = "".join(
        f'<path d="{paths_by_descriptor[descriptor]}" fill="{FILL}"/>'
        for descriptor in dom_order
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        + paths
        + "</svg>"
    )


def _answer_svg(
    source: str,
    dom_order: Sequence[str],
    descriptor: str,
) -> str:
    root = parse_svg(source)
    target = list(root)[dom_order.index(descriptor)]
    target.attrib["stroke"] = STROKE
    target.attrib["stroke-width"] = STROKE_WIDTH
    return serialize_svg(root)


def _benchmark_case(
    *,
    family: str,
    descriptor: str,
    layout_index: int,
    source: str,
    dom_order: Sequence[str],
    instruction: str,
) -> HoldoutCase:
    target_node_id = f"n{dom_order.index(descriptor) + 1}"
    return HoldoutCase(
        case=BenchmarkCase(
            task="set_contour",
            emoji_id=(
                f"holdout-{family}-layout-{layout_index}-{descriptor}"
            ),
            instruction=instruction,
            source_svg=source,
            answer_svg=_answer_svg(source, dom_order, descriptor),
            query_path=Path("."),
            answer_path=Path("."),
        ),
        family=family,
        descriptor=descriptor,
        layout_index=layout_index,
        target_node_id=target_node_id,
        dom_order=tuple(dom_order),
    )


def _position_cases(seed: int) -> list[HoldoutCase]:
    descriptors = tuple(POSITION_CENTERS)
    used_orders: set[tuple[str, ...]] = set()
    cases: list[HoldoutCase] = []
    variant_index = 0
    for layout_index in range(1, POSITION_LAYOUTS + 1):
        for descriptor_index, descriptor in enumerate(descriptors):
            case_seed = _case_seed(
                seed,
                "position",
                layout_index,
                descriptor,
            )
            target_index = (
                descriptor_index + (layout_index - 1) * 3
            ) % len(descriptors)
            dom_order = _case_dom_order(
                descriptors,
                descriptor,
                target_index,
                seed=case_seed,
                used=used_orders,
            )
            centers = _jittered_centers(
                tuple(POSITION_CENTERS.values()),
                seed=case_seed + 1,
            )
            paths = {
                candidate: _rect_path(
                    center,
                    *POSITION_DIMENSIONS[
                        (candidate_index + variant_index)
                        % len(POSITION_DIMENSIONS)
                    ],
                )
                for candidate_index, (candidate, center) in enumerate(
                    zip(descriptors, centers)
                )
            }
            source = _source_svg(paths, dom_order)
            cases.append(
                _benchmark_case(
                    family="position",
                    descriptor=descriptor,
                    layout_index=layout_index,
                    source=source,
                    dom_order=dom_order,
                    instruction=(
                        f"Add a {STROKE} outline {STROKE_WIDTH} units wide "
                        "around only the red shape at the "
                        f"{descriptor} of the image."
                    ),
                )
            )
            variant_index += 1
    return cases


def _size_cases(seed: int) -> list[HoldoutCase]:
    descriptors = tuple(SIZE_LEVELS)
    used_orders: set[tuple[str, ...]] = set()
    cases: list[HoldoutCase] = []
    variant_index = 0
    for layout_index in range(1, SIZE_LAYOUTS + 1):
        for descriptor_index, descriptor in enumerate(descriptors):
            case_seed = _case_seed(
                seed,
                "size",
                layout_index,
                descriptor,
            )
            target_index = (
                descriptor_index + layout_index - 1
            ) % len(descriptors)
            dom_order = _case_dom_order(
                descriptors,
                descriptor,
                target_index,
                seed=case_seed,
                used=used_orders,
            )
            center_order = list(SIZE_CENTERS)
            random.Random(case_seed + 1).shuffle(center_order)
            centers = _jittered_centers(
                center_order,
                seed=case_seed + 2,
            )
            paths = {
                candidate: _rect_path(
                    center,
                    *SIZE_DIMENSIONS[candidate][
                        (candidate_index + variant_index)
                        % len(SIZE_DIMENSIONS[candidate])
                    ],
                )
                for candidate_index, (candidate, center) in enumerate(
                    zip(descriptors, centers)
                )
            }
            source = _source_svg(paths, dom_order)
            cases.append(
                _benchmark_case(
                    family="size",
                    descriptor=descriptor,
                    layout_index=layout_index,
                    source=source,
                    dom_order=dom_order,
                    instruction=(
                        f"Add a {STROKE} outline {STROKE_WIDTH} units wide "
                        f"around only the {descriptor} red shape."
                    ),
                )
            )
            variant_index += 1
    return cases


def build_cases(seed: int = DEFAULT_SEED) -> list[HoldoutCase]:
    """Build the frozen held-out suite without touching model or renderer state."""
    return _position_cases(seed) + _size_cases(seed)


def _patch_targets(patch: Any | None) -> set[str]:
    if patch is None:
        return set()
    return {
        target
        for operation in patch.operations
        for target in operation.targets
    }


def validate_suite(cases: Sequence[HoldoutCase]) -> dict[str, Any]:
    """Fail closed if a future edit leaks an easy DOM grounding cue."""
    if PATCH_PROMPT_VERSION != HOLDOUT_PATCH_PROMPT_VERSION:
        raise ValueError(
            "held-out suite is frozen to patch prompt "
            f"v{HOLDOUT_PATCH_PROMPT_VERSION}, but active version is "
            f"v{PATCH_PROMPT_VERSION}"
        )
    case_ids = [item.case.case_id for item in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("held-out case IDs must be unique")

    path_character_counts: set[int] = set()
    sources: dict[str, str] = {}
    orders_by_family: dict[str, set[tuple[str, ...]]] = {}
    target_indices_by_family: dict[str, Counter[str]] = {}
    jitter_observed = False
    non_square_observed = False
    for item in cases:
        source_digest = hashlib.sha256(
            item.case.source_svg.encode("utf-8")
        ).hexdigest()
        sources[source_digest] = item.case.source_svg
        orders_by_family.setdefault(item.family, set()).add(item.dom_order)
        target_indices_by_family.setdefault(item.family, Counter())[
            item.target_node_id
        ] += 1

        gold_targets = _patch_targets(
            derive_patch(item.case.source_svg, item.case.answer_svg)
        )
        if gold_targets != {item.target_node_id}:
            raise ValueError(
                f"{item.case.case_id} gold target is {sorted(gold_targets)}, "
                f"expected {item.target_node_id}"
            )

        scene = build_scene(item.case.source_svg)
        nodes = scene["nodes"][1:]
        if not nodes or any(node["tag"] != "path" for node in nodes):
            raise ValueError("every candidate must be a path")
        if len(nodes) != len(item.dom_order):
            raise ValueError("DOM order metadata must cover every candidate")
        if any(node["attributes"] != {"fill": FILL} for node in nodes):
            raise ValueError("candidate DOM attributes must be identical")

        hashes: list[str] = []
        source_paths = [
            element.attrib["d"]
            for element in list(parse_svg(item.case.source_svg))
        ]
        serialized_scene = json.dumps(scene, sort_keys=True)
        for node in nodes:
            if "d" in node["attributes"]:
                raise ValueError("path coordinates leaked into scene attributes")
            protected = node.get("protected_geometry", {})
            if set(protected) != {"d"}:
                raise ValueError("each candidate needs protected d metadata")
            path_info = protected["d"]
            path_character_counts.add(int(path_info["characters"]))
            hashes.append(str(path_info["sha256"]))
        if len(hashes) != len(set(hashes)):
            raise ValueError("path hashes must be unique within each source")
        if any(path in serialized_scene for path in source_paths):
            raise ValueError("raw path coordinates leaked into the model scene")

        boxes = [_path_box(path) for path in source_paths]
        non_square_observed = non_square_observed or any(
            width != height for _, _, width, height in boxes
        )
        target_index = int(item.target_node_id[1:]) - 1
        if item.dom_order[target_index] != item.descriptor:
            raise ValueError("target node does not match semantic DOM order")

        if item.family == "position":
            for candidate, box in zip(item.dom_order, boxes):
                if _position_for_box(box) != candidate:
                    raise ValueError(
                        f"{item.case.case_id} moved {candidate} out of its bin"
                    )
                x, y, width, height = box
                center = (x + width // 2, y + height // 2)
                base = POSITION_CENTERS[candidate]
                if abs(center[0] - base[0]) > 2 or abs(
                    center[1] - base[1]
                ) > 2:
                    raise ValueError("position jitter exceeded two units")
                jitter_observed = jitter_observed or center != base
        elif item.family == "size":
            areas = {
                candidate: width * height
                for candidate, (_, _, width, height) in zip(
                    item.dom_order,
                    boxes,
                )
            }
            if len(set(areas.values())) != len(areas):
                raise ValueError("relative-size candidates must not tie")
            ranked = tuple(sorted(areas, key=areas.get))
            if ranked != tuple(SIZE_LEVELS):
                raise ValueError(
                    f"{item.case.case_id} size rank changed to {ranked}"
                )

            assigned_bases: set[tuple[int, int]] = set()
            for x, y, width, height in boxes:
                center = (x + width // 2, y + height // 2)
                nearest = min(
                    SIZE_CENTERS,
                    key=lambda base: max(
                        abs(center[0] - base[0]),
                        abs(center[1] - base[1]),
                    ),
                )
                if max(
                    abs(center[0] - nearest[0]),
                    abs(center[1] - nearest[1]),
                ) > 2:
                    raise ValueError("size-case jitter exceeded two units")
                assigned_bases.add(nearest)
                jitter_observed = jitter_observed or center != nearest
            if len(assigned_bases) != len(SIZE_CENTERS):
                raise ValueError("size candidates must occupy distinct regions")
        else:
            raise ValueError(f"unknown holdout family: {item.family}")

    if len(sources) != len(cases):
        raise ValueError(
            "every held-out case must use a distinct source SVG; got "
            f"{len(sources)} sources for {len(cases)} cases"
        )
    if len(path_character_counts) != 1:
        raise ValueError(
            "all protected paths must have equal character counts; got "
            f"{sorted(path_character_counts)}"
        )

    by_family = {
        family: sum(item.family == family for item in cases)
        for family in sorted({item.family for item in cases})
    }
    if any(
        len(orders_by_family[family]) != count
        for family, count in by_family.items()
    ):
        raise ValueError("every case must use a distinct family DOM order")
    expected_target_counts = {
        "position": Counter(
            {f"n{index}": POSITION_LAYOUTS for index in range(1, 10)}
        ),
        "size": Counter(
            {f"n{index}": SIZE_LAYOUTS for index in range(1, 5)}
        ),
    }
    if target_indices_by_family != expected_target_counts:
        raise ValueError(
            "target node indices are not balanced: "
            f"{target_indices_by_family}"
        )
    if not jitter_observed or not non_square_observed:
        raise ValueError("holdout requires coordinate and aspect variation")

    return {
        "cases": len(cases),
        "cases_by_family": by_family,
        "unique_source_svgs": len(sources),
        "one_unique_source_per_case": True,
        "position_replicates_per_descriptor": POSITION_LAYOUTS,
        "size_replicates_per_descriptor": SIZE_LAYOUTS,
        "position_descriptors": list(POSITION_CENTERS),
        "size_descriptors": list(SIZE_LEVELS),
        "deterministic_case_specific_dom_order": True,
        "unique_dom_order_within_each_family": True,
        "balanced_target_node_indices": True,
        "coordinate_jitter_units": 2,
        "within_position_bin_jitter_verified": True,
        "relative_size_ranking_verified": True,
        "aspect_variation_present": True,
        "same_tag": "path",
        "same_fill": FILL,
        "raw_path_coordinates_hidden_from_model": True,
        "protected_path_sha256_only": True,
        "equal_path_character_counts": True,
        "path_characters": next(iter(path_character_counts)),
        "unique_path_hashes_within_each_source": True,
    }


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
            float(bool(record["metrics"].get("gold_patch_exact")))
            for record in records
        ),
        "valid_output_rate": _mean(
            float(bool(record["metrics"]["valid_output"]))
            for record in records
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
        "prompt_tokens": sum(
            int(
                call.get("metadata", {})
                .get("usage", {})
                .get("prompt_tokens", 0)
            )
            for call in calls
        ),
        "completion_tokens": sum(
            int(
                call.get("metadata", {})
                .get("usage", {})
                .get("completion_tokens", 0)
            )
            for call in calls
        ),
    }


def _run_arm(
    name: str,
    cases: Sequence[HoldoutCase],
    model_config: dict[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if name == "skeleton_patch":
        architecture = SkeletonPatchArchitecture()
    elif name == "visual_stats_patch":
        architecture = VisualStatsPatchArchitecture(
            cache_dir=str(output_root / ".visual-stats-cache"),
            render_size=VISUAL_STATS_RENDER_SIZE,
        )
    else:  # pragma: no cover
        raise ValueError(name)

    model = RecordingModelAdapter(create_model(model_config))
    records: list[dict[str, Any]] = []
    arm_dir = output_root / name
    arm_dir.mkdir(parents=True, exist_ok=False)
    with (arm_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for item in cases:
            case = item.case
            call_offset = len(model.records)
            started = time.perf_counter()
            result = architecture.run(case, model)
            wall_seconds = time.perf_counter() - started
            call_details = model.records[call_offset:]
            predicted_targets = _patch_targets(result.patch)
            gold_targets = _patch_targets(
                derive_patch(case.source_svg, case.answer_svg)
            )
            metrics = evaluate_output(
                case.source_svg,
                case.answer_svg,
                result.output_svg,
                result.patch,
                render=True,
                render_size=EVALUATION_RENDER_SIZE,
            )
            record = {
                "case_id": case.case_id,
                "family": item.family,
                "descriptor": item.descriptor,
                "layout_index": item.layout_index,
                "architecture": name,
                "expected_target": item.target_node_id,
                "error": result.error,
                "patch": (
                    result.patch.to_dict()
                    if result.patch is not None
                    else None
                ),
                "raw_responses": result.raw_responses,
                "predicted_targets": sorted(predicted_targets),
                "gold_targets": sorted(gold_targets),
                "target_exact": predicted_targets == gold_targets,
                "wall_seconds": wall_seconds,
                "model_call_details": call_details,
                "metrics": metrics,
            }
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    summary = {
        "architecture": name,
        **_metrics_summary(records),
        "by_family": {
            family: _metrics_summary(
                [record for record in records if record["family"] == family]
            )
            for family in sorted({record["family"] for record in records})
        },
    }
    (arm_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return records, summary


def _correctness_outcomes(
    baseline: Sequence[dict[str, Any]],
    visual: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    outcomes = {
        "visual_only_wins": 0,
        "skeleton_only_wins": 0,
        "both_correct": 0,
        "neither_correct": 0,
    }
    for left, right in zip(baseline, visual):
        if field == "target_exact":
            left_correct = bool(left[field])
            right_correct = bool(right[field])
        else:
            left_correct = bool(left["metrics"].get(field))
            right_correct = bool(right["metrics"].get(field))
        if right_correct and not left_correct:
            outcomes["visual_only_wins"] += 1
        elif left_correct and not right_correct:
            outcomes["skeleton_only_wins"] += 1
        elif left_correct and right_correct:
            outcomes["both_correct"] += 1
        else:
            outcomes["neither_correct"] += 1
    return outcomes


def _paired_outcomes(
    baseline: Sequence[dict[str, Any]],
    visual: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {record["case_id"]: record for record in baseline}
    visual_by_id = {record["case_id"]: record for record in visual}
    if set(baseline_by_id) != set(visual_by_id):
        raise ValueError("held-out A/B arms used different cases")

    case_ids = sorted(baseline_by_id)
    left = [baseline_by_id[case_id] for case_id in case_ids]
    right = [visual_by_id[case_id] for case_id in case_ids]
    mse = {
        "visual_lower": 0,
        "skeleton_lower": 0,
        "ties": 0,
        "missing_pairs": 0,
        "mean_visual_minus_skeleton": None,
    }
    deltas: list[float] = []
    for left_record, right_record in zip(left, right):
        left_mse = left_record["metrics"].get("failure_aware_mse")
        right_mse = right_record["metrics"].get("failure_aware_mse")
        if left_mse is None or right_mse is None:
            mse["missing_pairs"] += 1
            continue
        delta = float(right_mse) - float(left_mse)
        deltas.append(delta)
        if abs(delta) <= 1e-12:
            mse["ties"] += 1
        elif delta < 0:
            mse["visual_lower"] += 1
        else:
            mse["skeleton_lower"] += 1
    mse["mean_visual_minus_skeleton"] = _mean(deltas)

    return {
        "cases": len(case_ids),
        "target_exact": _correctness_outcomes(
            left,
            right,
            "target_exact",
        ),
        "gold_patch_exact": _correctness_outcomes(
            left,
            right,
            "gold_patch_exact",
        ),
        "failure_aware_mse": mse,
    }


def paired_summary(
    baseline: Sequence[dict[str, Any]],
    visual: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_families = {record["family"] for record in baseline}
    visual_families = {record["family"] for record in visual}
    if baseline_families != visual_families:
        raise ValueError("held-out A/B arms used different families")
    return {
        "overall": _paired_outcomes(baseline, visual),
        "by_family": {
            family: _paired_outcomes(
                [
                    record
                    for record in baseline
                    if record["family"] == family
                ],
                [
                    record
                    for record in visual
                    if record["family"] == family
                ],
            )
            for family in sorted(baseline_families)
        },
    }


def _manifest(cases: Sequence[HoldoutCase], seed: int) -> dict[str, Any]:
    source_hashes = {
        hashlib.sha256(item.case.source_svg.encode("utf-8")).hexdigest()
        for item in cases
    }
    return {
        "format": SUITE_FORMAT,
        "seed": seed,
        "patch_prompt": _prompt_provenance(),
        "case_count": len(cases),
        "unique_source_count": len(source_hashes),
        "one_unique_source_per_case": len(source_hashes) == len(cases),
        "cases": [
            {
                "case_id": item.case.case_id,
                "family": item.family,
                "descriptor": item.descriptor,
                "layout_index": item.layout_index,
                "expected_target": item.target_node_id,
                "source_sha256": hashlib.sha256(
                    item.case.source_svg.encode("utf-8")
                ).hexdigest(),
                "answer_sha256": hashlib.sha256(
                    item.case.answer_svg.encode("utf-8")
                ).hexdigest(),
                "dom_order_sha256": hashlib.sha256(
                    json.dumps(item.dom_order).encode("utf-8")
                ).hexdigest(),
            }
            for item in cases
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen path-only position-and-size grounding A/B holdout."
        )
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/qwen2.5-7b-ollama.json",
    )
    parser.add_argument(
        "--output-root",
        default="runs/qwen2.5-7b-spatial-grounding-holdout-v1",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def reserve_output_root(output_root: Path) -> None:
    """Reserve a fresh result root so prior experiment artifacts stay immutable."""
    output_root.mkdir(parents=True, exist_ok=False)


def main() -> int:
    args = build_parser().parse_args()
    ensure_renderer()
    model_config = load_model_config(args.model_config)
    cases = build_cases(args.seed)
    design = validate_suite(cases)

    output_root = Path(args.output_root)
    reserve_output_root(output_root)
    (output_root / "manifest.json").write_text(
        json.dumps(_manifest(cases, args.seed), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    baseline, baseline_summary = _run_arm(
        "skeleton_patch",
        cases,
        model_config,
        output_root,
    )
    visual, visual_summary = _run_arm(
        "visual_stats_patch",
        cases,
        model_config,
        output_root,
    )
    summary = {
        "format": SUITE_FORMAT,
        "seed": args.seed,
        "model_config": args.model_config,
        "patch_prompt": _prompt_provenance(),
        "design": {
            **design,
            "visual_stats_render_size": VISUAL_STATS_RENDER_SIZE,
            "evaluation_render_size": EVALUATION_RENDER_SIZE,
            "arm_order": ["skeleton_patch", "visual_stats_patch"],
        },
        "arms": {
            "skeleton_patch": baseline_summary,
            "visual_stats_patch": visual_summary,
        },
        "paired": paired_summary(baseline, visual),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
