from __future__ import annotations

import base64
import io
import json
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from svgpatchlab.core import PatchPolicy, apply_patch, build_scene, parse_patch, validate_patch
from svgpatchlab.core.patch import (
    PATCH_SCHEMA_NAME,
    PatchError,
    extract_json_object,
    patch_json_schema,
)
from svgpatchlab.core.xml import (
    element_fingerprint,
    index_tree,
    local_name,
    parse_svg,
    serialize_svg,
)
from svgpatchlab.eval.render import render_svg_png, render_svg_visual_context
from svgpatchlab.models import ModelAdapter
from svgpatchlab.types import ArchitectureResult, BenchmarkCase, ModelRequest

from .base import Architecture
from .prompts import patch_prompt, qwen_completion_prompt, target_selection_prompt
from .root_tasks import compile_root_task_patch


TARGET_SELECTION_SCHEMA_NAME = "svgpatchlab_target_selection_v1"
TARGET_SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["targets"],
    "properties": {
        "targets": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "pattern": "^n[0-9]+$"},
        }
    },
}

COMPLETION_SCHEMA_NAME = "svgpatchlab_completion_v1"
COMPLETION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["target", "element"],
    "properties": {
        "target": {"type": "string", "pattern": "^n[0-9]+$"},
        "element": {"type": "string", "minLength": 1},
    },
}

DELETE_PATCH_SCHEMA_NAME = "svgpatchlab_delete_patch_v1"
DELETE_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "operations"],
    "properties": {
        "version": {"type": "integer", "enum": [2]},
        "operations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["op", "targets"],
                "properties": {
                    "op": {"type": "string", "enum": ["remove_element"]},
                    "targets": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "string",
                            "pattern": "^n[0-9]+$",
                        },
                    },
                },
            },
        },
    },
}


def _png_data_url(png: bytes) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _selected_targets(
    response_text: str,
    scene: dict[str, Any],
    max_candidates: int,
    *,
    forbid_root: bool,
) -> tuple[str, ...]:
    payload = extract_json_object(response_text)
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise PatchError("target selection requires a nonempty 'targets' list")
    if not all(isinstance(item, str) for item in raw_targets):
        raise PatchError("selected targets must be node ID strings")

    nodes = {node["id"]: node for node in scene["nodes"]}
    unknown = sorted(set(raw_targets) - set(nodes))
    if unknown:
        raise PatchError(f"target selection contains unknown IDs: {', '.join(unknown)}")

    root_id = scene["root_id"]
    if forbid_root and root_id in raw_targets:
        raise PatchError("element removal cannot select the SVG root")

    parents = {node_id: node.get("parent") for node_id, node in nodes.items()}

    def ancestors(node_id: str) -> set[str]:
        result: set[str] = set()
        current = parents.get(node_id)
        while current is not None:
            result.add(current)
            current = parents.get(current)
        return result

    # Preserve model ranking while discarding duplicates and contradictory
    # ancestor/descendant alternatives. The first-ranked target wins.
    selected: list[str] = []
    for target in raw_targets:
        if target in selected:
            continue
        target_ancestors = ancestors(target)
        if any(
            existing in target_ancestors or target in ancestors(existing)
            for existing in selected
        ):
            continue
        selected.append(target)
        if len(selected) >= max_candidates:
            break

    if not selected:
        raise PatchError("target selection did not contain a usable node ID")
    return tuple(selected)


def _hide_node(svg: str, node_id: str) -> str:
    root = parse_svg(svg)
    by_id = {node.node_id: node.element for node in index_tree(root)}
    if node_id not in by_id:
        raise PatchError(f"cannot preview unknown node: {node_id}")
    element = by_id[node_id]
    prior = element.attrib.get("style")
    element.attrib["style"] = (f"{prior};" if prior else "") + "display:none"
    return serialize_svg(root)


def _spatial_label(x_pct: float, y_pct: float) -> str:
    horizontal = "left" if x_pct < 33.333 else "right" if x_pct > 66.667 else "center"
    vertical = "top" if y_pct < 33.333 else "bottom" if y_pct > 66.667 else "center"
    return vertical if horizontal == "center" else (
        horizontal if vertical == "center" else f"{vertical}-{horizontal}"
    )


def _component_summaries(
    changed: Any,
    categories: dict[str, Any],
    *,
    max_components: int = 3,
) -> list[dict[str, Any]]:
    import numpy as np

    height, width = changed.shape
    visited = np.zeros_like(changed, dtype=np.bool_)
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for start_y, start_x in np.argwhere(changed):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue
        visited[y, x] = True
        stack = [(y, x)]
        pixels: list[tuple[int, int]] = []
        while stack:
            current_y, current_x = stack.pop()
            pixels.append((current_y, current_x))
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_y == 0 and delta_x == 0:
                        continue
                    next_y = current_y + delta_y
                    next_x = current_x + delta_x
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and changed[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        components.append((len(pixels), pixels))

    summaries: list[dict[str, Any]] = []
    for area, pixels in sorted(components, key=lambda item: item[0], reverse=True)[
        :max_components
    ]:
        rows = np.asarray([pixel[0] for pixel in pixels], dtype=np.int32)
        columns = np.asarray([pixel[1] for pixel in pixels], dtype=np.int32)
        x_pct = 100.0 * float(columns.mean() + 0.5) / width
        y_pct = 100.0 * float(rows.mean() + 0.5) / height
        category_counts = {
            name: int(mask[rows, columns].sum())
            for name, mask in categories.items()
            if mask[rows, columns].any()
        }
        summaries.append(
            {
                "area_px": area,
                "area_pct_canvas": round(100.0 * area / changed.size, 3),
                "bbox_px": [
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max() - columns.min() + 1),
                    int(rows.max() - rows.min() + 1),
                ],
                "centroid_pct": [round(x_pct, 2), round(y_pct, 2)],
                "position": _spatial_label(x_pct, y_pct),
                "pixel_categories": category_counts,
            }
        )
    return summaries


def _color_name(pixel: Any) -> str:
    red, green, blue, alpha = (int(value) for value in pixel)
    if alpha <= 12:
        return "transparent"
    # Quantization suppresses anti-aliasing noise while retaining useful color
    # transitions for the text-only model.
    channels = tuple(min(255, (value // 32) * 32 + 16) for value in (red, green, blue))
    return f"#{channels[0]:02x}{channels[1]:02x}{channels[2]:02x}@{alpha / 255:.2f}"


def _ownership_metrics(
    changed: Any,
    id_map: Any,
    scene: dict[str, Any] | None,
    target: str | None,
) -> dict[str, Any] | None:
    if id_map is None or scene is None or target is None:
        return None

    parents = {node["id"]: node.get("parent") for node in scene["nodes"]}

    def is_target_or_descendant(node_id: str) -> bool:
        current: str | None = node_id
        while current is not None:
            if current == target:
                return True
            current = parents.get(current)
        return False

    indexes = [
        index
        for index, node_id in enumerate(id_map.leaf_node_ids)
        if is_target_or_descendant(node_id)
    ]
    if not indexes:
        return {
            "visible_target_area_px": 0,
            "changed_pixels_owned_by_target_pct": 0.0,
            "visible_target_pixels_changed_pct": 0.0,
        }

    import numpy as np

    ownership = np.isin(id_map.labels, indexes)
    visible_area = int(ownership.sum())
    intersection = int((ownership & changed).sum())
    changed_area = int(changed.sum())
    return {
        "visible_target_area_px": visible_area,
        "changed_pixels_owned_by_target_pct": round(
            100.0 * intersection / changed_area, 3
        )
        if changed_area
        else 0.0,
        "visible_target_pixels_changed_pct": round(
            100.0 * intersection / visible_area, 3
        )
        if visible_area
        else 0.0,
    }


def _counterfactual_difference(
    original_png: bytes,
    hidden_png: bytes,
    *,
    id_map: Any = None,
    scene: dict[str, Any] | None = None,
    target: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - renderer dependency guard
        raise RuntimeError("counterfactual summaries require numpy and Pillow") from exc

    original_u8 = np.asarray(
        Image.open(io.BytesIO(original_png)).convert("RGBA"), dtype=np.uint8
    )
    hidden_u8 = np.asarray(
        Image.open(io.BytesIO(hidden_png)).convert("RGBA"), dtype=np.uint8
    )
    if original_u8.shape != hidden_u8.shape:
        raise RuntimeError("counterfactual preview size differs from the original render")

    delta = np.abs(original_u8.astype(np.int16) - hidden_u8.astype(np.int16))
    changed = delta.max(axis=2) > 5
    height, width = changed.shape
    original_visible = original_u8[:, :, 3] > 12
    hidden_visible = hidden_u8[:, :, 3] > 12
    became_transparent = changed & original_visible & ~hidden_visible
    became_visible = changed & ~original_visible & hidden_visible
    repainted_or_revealed = changed & original_visible & hidden_visible
    alpha_changed = changed & (delta[:, :, 3] > 5)
    rgb_changed = changed & (delta[:, :, :3].max(axis=2) > 5)
    categories = {
        "became_transparent": became_transparent,
        "became_visible": became_visible,
        "repainted_or_revealed": repainted_or_revealed,
        "alpha_changed": alpha_changed,
        "rgb_changed": rgb_changed,
    }

    # The visualization is derived from the already-rendered pair; it does not
    # require an additional SVG rasterization. Unchanged content remains as a
    # dim grayscale reference. Difference categories use a fixed legend.
    grayscale = np.rint(
        original_u8[:, :, :3].astype(np.float32).mean(axis=2) * 0.22
    ).astype(np.uint8)
    heatmap = np.empty_like(original_u8)
    heatmap[:, :, :3] = grayscale[:, :, None]
    heatmap[:, :, 3] = 255
    heatmap[repainted_or_revealed, :3] = (0, 199, 255)
    heatmap[alpha_changed, :3] = (255, 214, 10)
    heatmap[became_visible, :3] = (52, 199, 89)
    heatmap[became_transparent, :3] = (255, 59, 48)
    heatmap_image = Image.fromarray(heatmap, mode="RGBA")
    heatmap_output = io.BytesIO()
    heatmap_image.save(heatmap_output, format="PNG")

    if not changed.any():
        return (
            {
                "changed": False,
                "changed_area_px": 0,
                "changed_area_pct": 0.0,
                "became_transparent_area_pct": 0.0,
                "became_visible_area_pct": 0.0,
                "repainted_or_revealed_area_pct": 0.0,
                "alpha_changed_area_pct": 0.0,
                "rgb_changed_area_pct": 0.0,
                # Backward-compatible names used by existing prompts/results.
                "transparent_hole_area_pct": 0.0,
                "revealed_or_repainted_area_pct": 0.0,
                "components": [],
                "dominant_color_transitions": [],
                "ownership": _ownership_metrics(
                    changed, id_map, scene, target
                ),
            },
            heatmap_output.getvalue(),
        )

    rows = np.any(changed, axis=1).nonzero()[0]
    columns = np.any(changed, axis=0).nonzero()[0]
    changed_rows, changed_columns = np.nonzero(changed)
    centroid_x_pct = 100.0 * float(changed_columns.mean() + 0.5) / width
    centroid_y_pct = 100.0 * float(changed_rows.mean() + 0.5) / height
    transition_counts = Counter(
        (_color_name(original_u8[y, x]), _color_name(hidden_u8[y, x]))
        for y, x in zip(changed_rows.tolist(), changed_columns.tolist())
    )
    changed_count = int(changed.sum())
    summary = {
        "changed": True,
        "changed_area_px": changed_count,
        "changed_area_pct": round(100.0 * changed_count / changed.size, 3),
        "bbox_px": [
            int(columns[0]),
            int(rows[0]),
            int(columns[-1] - columns[0] + 1),
            int(rows[-1] - rows[0] + 1),
        ],
        "bbox_pct": [
            round(100.0 * int(columns[0]) / width, 2),
            round(100.0 * int(rows[0]) / height, 2),
            round(100.0 * int(columns[-1] - columns[0] + 1) / width, 2),
            round(100.0 * int(rows[-1] - rows[0] + 1) / height, 2),
        ],
        "centroid_pct": [round(centroid_x_pct, 2), round(centroid_y_pct, 2)],
        "position": _spatial_label(centroid_x_pct, centroid_y_pct),
        "render_size": [width, height],
        "became_transparent_area_pct": round(
            100.0 * int(became_transparent.sum()) / changed.size, 3
        ),
        "became_visible_area_pct": round(
            100.0 * int(became_visible.sum()) / changed.size, 3
        ),
        "repainted_or_revealed_area_pct": round(
            100.0 * int(repainted_or_revealed.sum()) / changed.size, 3
        ),
        "alpha_changed_area_pct": round(
            100.0 * int(alpha_changed.sum()) / changed.size, 3
        ),
        "rgb_changed_area_pct": round(
            100.0 * int(rgb_changed.sum()) / changed.size, 3
        ),
        # Backward-compatible names used by existing prompts/results.
        "transparent_hole_area_pct": round(
            100.0 * int(became_transparent.sum()) / changed.size, 3
        ),
        "revealed_or_repainted_area_pct": round(
            100.0
            * int((repainted_or_revealed | became_visible).sum())
            / changed.size,
            3,
        ),
        "mean_rgba_delta": round(float(delta[changed].mean()) / 255.0, 4),
        "max_channel_delta": round(float(delta[changed].max()) / 255.0, 4),
        "components": _component_summaries(changed, categories),
        "dominant_color_transitions": [
            {
                "from": source,
                "to": destination,
                "pixels": count,
                "pct_of_changed": round(100.0 * count / changed_count, 2),
            }
            for (source, destination), count in transition_counts.most_common(3)
        ],
        "ownership": _ownership_metrics(changed, id_map, scene, target),
    }
    return summary, heatmap_output.getvalue()


def _counterfactual_summary(original_png: bytes, hidden_png: bytes) -> dict[str, Any]:
    """Backward-compatible numeric-only wrapper used by external callers."""

    summary, _ = _counterfactual_difference(original_png, hidden_png)
    return summary


def _completion_triggered(
    preview_records: list[dict[str, Any]],
    removed_targets: set[str],
    *,
    min_transparent_area_pct: float,
    max_revealed_fraction: float,
) -> tuple[bool, list[str]]:
    """Conservatively identify deletions that expose no encoded underlayer."""

    triggered_targets: list[str] = []
    for record in preview_records:
        target = record["target"]
        if target not in removed_targets:
            continue
        difference = record["difference"]
        changed_area = float(difference.get("changed_area_pct", 0.0))
        transparent_area = float(
            difference.get("became_transparent_area_pct", 0.0)
        )
        revealed_area = float(
            difference.get("revealed_or_repainted_area_pct", 0.0)
        )
        revealed_fraction = revealed_area / changed_area if changed_area else 0.0
        if (
            bool(difference.get("changed"))
            and transparent_area >= min_transparent_area_pct
            and revealed_fraction <= max_revealed_fraction
        ):
            triggered_targets.append(target)
    return bool(triggered_targets), triggered_targets


def _change_mask(original_png: bytes, edited_png: bytes) -> Any:
    import numpy as np
    from PIL import Image

    original = np.asarray(
        Image.open(io.BytesIO(original_png)).convert("RGBA"), dtype=np.int16
    )
    edited = np.asarray(
        Image.open(io.BytesIO(edited_png)).convert("RGBA"), dtype=np.int16
    )
    if original.shape != edited.shape:
        raise RuntimeError("completion comparison render sizes differ")
    return np.abs(original - edited).max(axis=2) > 5


def _outside_edit_mse(
    reference_png: bytes,
    candidate_png: bytes,
    editable_mask: Any,
) -> float:
    import numpy as np
    from PIL import Image

    reference = np.asarray(
        Image.open(io.BytesIO(reference_png)).convert("RGBA"), dtype=np.float32
    ) / 255.0
    candidate = np.asarray(
        Image.open(io.BytesIO(candidate_png)).convert("RGBA"), dtype=np.float32
    ) / 255.0
    if reference.shape != candidate.shape or reference.shape[:2] != editable_mask.shape:
        raise RuntimeError("completion locality comparison shapes differ")
    outside = ~editable_mask
    if not outside.any():
        return 0.0
    return float(np.square(reference[outside] - candidate[outside]).mean())


def _inside_edit_mse(
    reference_png: bytes,
    candidate_png: bytes,
    editable_mask: Any,
) -> float:
    import numpy as np
    from PIL import Image

    reference = np.asarray(
        Image.open(io.BytesIO(reference_png)).convert("RGBA"), dtype=np.float32
    ) / 255.0
    candidate = np.asarray(
        Image.open(io.BytesIO(candidate_png)).convert("RGBA"), dtype=np.float32
    ) / 255.0
    if reference.shape != candidate.shape or reference.shape[:2] != editable_mask.shape:
        raise RuntimeError("completion foreground comparison shapes differ")
    if not editable_mask.any():
        return 0.0
    return float(
        np.square(reference[editable_mask] - candidate[editable_mask]).mean()
    )


def _validate_removed_elements_absent(
    source_svg: str,
    generated_svg: str,
    removed_targets: set[str],
) -> None:
    source_nodes = {node.node_id: node.element for node in index_tree(parse_svg(source_svg))}
    removed_fingerprints = {
        element_fingerprint(source_nodes[target])
        for target in removed_targets
        if target in source_nodes
    }
    generated_fingerprints = {
        element_fingerprint(node.element)
        for node in index_tree(parse_svg(generated_svg))
    }
    if removed_fingerprints & generated_fingerprints:
        raise PatchError("Qwen completion restored an element that was deleted")


def _completion_reconstruction_candidates(
    source_svg: str,
    selected_targets: tuple[str, ...],
    removed_targets: set[str],
    scene_nodes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    source_nodes = {
        node.node_id: node.element for node in index_tree(parse_svg(source_svg))
    }
    reconstructable_tags = {
        "circle",
        "ellipse",
        "line",
        "path",
        "polygon",
        "polyline",
        "rect",
    }
    removed_boxes = [
        scene_nodes.get(target, {}).get("visual", {}).get("bbox")
        for target in removed_targets
    ]
    removed_boxes = [
        box
        for box in removed_boxes
        if isinstance(box, list) and len(box) == 4
    ]

    def box_gap(target: str) -> float:
        box = scene_nodes.get(target, {}).get("visual", {}).get("bbox")
        if not isinstance(box, list) or len(box) != 4 or not removed_boxes:
            return float("inf")
        x, y, width, height = (float(value) for value in box)
        right, bottom = x + width, y + height
        gaps: list[float] = []
        for removed in removed_boxes:
            rx, ry, rwidth, rheight = (float(value) for value in removed)
            rright, rbottom = rx + rwidth, ry + rheight
            dx = max(rx - right, x - rright, 0.0)
            dy = max(ry - bottom, y - rbottom, 0.0)
            gaps.append((dx * dx + dy * dy) ** 0.5)
        return min(gaps)

    eligible = [
        target
        for target, element in source_nodes.items()
        if target not in removed_targets
        and target != "n0"
        and local_name(element.tag) in reconstructable_tags
        and scene_nodes.get(target, {}).get("visual", {}).get("visible", True)
    ]
    selected_rank = {
        target: index for index, target in enumerate(selected_targets)
    }
    eligible.sort(
        key=lambda target: (
            box_gap(target),
            0 if target in selected_rank else 1,
            selected_rank.get(target, 0),
            int(target[1:]),
        )
    )
    # Stage one may correctly shortlist only the foreground. The nearest
    # surviving drawable nodes supply completion candidates without exposing
    # the full DOM to the generator.
    finite_gaps = [
        box_gap(target)
        for target in eligible
        if box_gap(target) != float("inf")
    ]
    if finite_gaps and removed_boxes:
        closest_gap = min(finite_gaps)
        spatial_margin = max(
            max(float(box[2]), float(box[3])) * 0.05
            for box in removed_boxes
        )
        eligible = [
            target
            for target in eligible
            if box_gap(target) <= closest_gap + spatial_margin
        ]
    else:
        shortlisted = [
            target for target in eligible if target in selected_rank
        ]
        eligible = shortlisted or eligible
    eligible = eligible[:4]
    return [
        {
            "target": target,
            "element": serialize_svg(source_nodes[target]),
            "visual": scene_nodes.get(target, {}).get("visual", {}),
        }
        for target in eligible
    ]


_PATH_TOKEN_RE = re.compile(
    r"[A-Za-z]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def _snap_generated_geometry(existing: Any, replacement: Any) -> None:
    """Fit a generated circle exactly to a surviving semicircular arc."""

    if local_name(existing.tag) != "path" or local_name(replacement.tag) != "circle":
        return
    tokens = _PATH_TOKEN_RE.findall(existing.attrib.get("d", ""))
    if len(tokens) < 11 or tokens[0] != "M" or tokens[3] != "A":
        return
    try:
        start_x, start_y = float(tokens[1]), float(tokens[2])
        radius_x, radius_y = abs(float(tokens[4])), abs(float(tokens[5]))
        end_x, end_y = float(tokens[9]), float(tokens[10])
    except ValueError:
        return
    if not radius_x or abs(radius_x - radius_y) > max(radius_x, radius_y) * 0.02:
        return
    diameter = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
    if abs(diameter - 2.0 * radius_x) > radius_x * 0.05:
        return

    replacement.attrib["cx"] = f"{(start_x + end_x) / 2.0:.12g}"
    replacement.attrib["cy"] = f"{(start_y + end_y) / 2.0:.12g}"
    replacement.attrib["r"] = f"{(radius_x + radius_y) / 2.0:.12g}"


def _apply_completion_replacement(
    source_svg: str,
    localized_svg: str,
    response_text: str,
    allowed_targets: set[str],
) -> str:
    payload = extract_json_object(response_text)
    if set(payload) != {"target", "element"}:
        raise PatchError(
            "Qwen completion must contain exactly 'target' and 'element'"
        )
    target = payload["target"]
    element_xml = payload["element"]
    if not isinstance(target, str) or target not in allowed_targets:
        raise PatchError("Qwen completion selected an invalid reconstruction target")
    if not isinstance(element_xml, str) or not element_xml.strip():
        raise PatchError("Qwen completion requires a nonempty replacement element")

    wrapper = parse_svg(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        + element_xml.strip()
        + "</svg>"
    )
    replacement_children = list(wrapper)
    if len(replacement_children) != 1:
        raise PatchError("Qwen completion replacement must contain one SVG element")
    replacement = replacement_children[0]

    source_nodes = {
        node.node_id: node.element for node in index_tree(parse_svg(source_svg))
    }
    if target not in source_nodes:
        raise PatchError("Qwen completion target is missing from the source SVG")
    source_fingerprint = element_fingerprint(source_nodes[target])

    localized_root = parse_svg(localized_svg)
    matches = [
        node.element
        for node in index_tree(localized_root)
        if element_fingerprint(node.element) == source_fingerprint
    ]
    if len(matches) != 1:
        raise PatchError(
            "Qwen completion target is not uniquely preserved after deletion"
        )
    existing = matches[0]
    _snap_generated_geometry(existing, replacement)

    parent = next(
        (
            candidate
            for candidate in localized_root.iter()
            if existing in list(candidate)
        ),
        None,
    )
    if parent is None:
        raise PatchError("Qwen completion cannot replace the SVG root")
    child_index = list(parent).index(existing)
    replacement.tail = existing.tail
    parent.remove(existing)
    parent.insert(child_index, replacement)
    return serialize_svg(localized_root)


def _validate_generated_completion(source_svg: str, generated_svg: str) -> None:
    source_root = parse_svg(source_svg)
    generated_root = parse_svg(generated_svg)
    canvas_attributes = ("viewBox", "width", "height")
    mismatched = [
        name
        for name in canvas_attributes
        if source_root.attrib.get(name) != generated_root.attrib.get(name)
    ]
    if mismatched:
        raise PatchError(
            "Qwen completion changed canvas attributes: " + ", ".join(mismatched)
        )

    forbidden_tags = {"script", "foreignObject", "iframe", "object", "embed"}
    nodes = index_tree(generated_root)
    if len(nodes) > 512:
        raise PatchError("Qwen completion exceeds the 512-node safety limit")
    for node in nodes:
        tag = local_name(node.element.tag)
        if tag in forbidden_tags:
            raise PatchError(f"Qwen completion contains forbidden element: {tag}")
        for name, value in node.element.attrib.items():
            lowered_name = local_name(name).lower()
            compact_value = value.lower().replace(" ", "")
            if lowered_name.startswith("on"):
                raise PatchError(
                    f"Qwen completion contains event handler: {lowered_name}"
                )
            if lowered_name in {"href", "src"} and not value.startswith("#"):
                raise PatchError(
                    f"Qwen completion contains external reference: {lowered_name}"
                )
            if any(
                token in compact_value
                for token in ("javascript:", "data:", "<", ">")
            ):
                raise PatchError(
                    f"Qwen completion contains unsafe value for {lowered_name}"
                )


class SemanticIdPatchArchitecture(Architecture):
    """Two-stage patching grounded by a rendered element ownership buffer.

    Stage one ranks at most `max_candidates` DOM nodes using the compact scene
    plus a normal/ID render pair when the model supports images. Only those
    candidates are then hidden and rerendered. Stage two sees the previews and
    emits the ordinary validated patch. The final patch may address only the
    shortlisted nodes.
    """

    name = "semantic_id_patch"
    requires_renderer = True

    def __init__(
        self,
        render_size: int = 192,
        max_candidates: int = 3,
        allow_counterfactual_fallback: bool = True,
        qwen_completion: bool = False,
        completion_min_transparent_area_pct: float = 0.25,
        completion_max_revealed_fraction: float = 0.1,
        completion_max_outside_mse: float = 0.01,
        completion_min_inside_change_mse: float = 0.001,
        completion_min_generated_change_mse: float = 0.0005,
        completion_max_attempts: int = 2,
        route_root_tasks: bool = True,
        policy: PatchPolicy | None = None,
    ):
        if render_size < 32:
            raise ValueError("render_size must be at least 32")
        if not 1 <= max_candidates <= 8:
            raise ValueError("max_candidates must be between 1 and 8")
        self.render_size = int(render_size)
        self.max_candidates = int(max_candidates)
        self.allow_counterfactual_fallback = bool(allow_counterfactual_fallback)
        self.qwen_completion = bool(qwen_completion)
        self.completion_min_transparent_area_pct = float(
            completion_min_transparent_area_pct
        )
        self.completion_max_revealed_fraction = float(
            completion_max_revealed_fraction
        )
        self.completion_max_outside_mse = float(completion_max_outside_mse)
        self.completion_min_inside_change_mse = float(
            completion_min_inside_change_mse
        )
        self.completion_min_generated_change_mse = float(
            completion_min_generated_change_mse
        )
        self.completion_max_attempts = int(completion_max_attempts)
        self.route_root_tasks = bool(route_root_tasks)
        if self.completion_min_transparent_area_pct < 0:
            raise ValueError("completion_min_transparent_area_pct cannot be negative")
        if not 0 <= self.completion_max_revealed_fraction <= 1:
            raise ValueError(
                "completion_max_revealed_fraction must be between zero and one"
            )
        if self.completion_max_outside_mse < 0:
            raise ValueError("completion_max_outside_mse cannot be negative")
        if self.completion_min_inside_change_mse < 0:
            raise ValueError("completion_min_inside_change_mse cannot be negative")
        if self.completion_min_generated_change_mse < 0:
            raise ValueError(
                "completion_min_generated_change_mse cannot be negative"
            )
        if not 1 <= self.completion_max_attempts <= 3:
            raise ValueError("completion_max_attempts must be between one and three")
        self.policy = policy or PatchPolicy()

    @lru_cache(maxsize=128)
    def _visual_context(self, svg: str):
        return render_svg_visual_context(
            svg,
            size=self.render_size,
            fallback=self.allow_counterfactual_fallback,
        )

    def run(self, case: BenchmarkCase, model: ModelAdapter) -> ArchitectureResult:
        result = ArchitectureResult()
        try:
            base_scene = build_scene(case.source_svg)
            if self.route_root_tasks:
                routed_patch = compile_root_task_patch(case, base_scene)
                if routed_patch is not None:
                    validate_patch(
                        routed_patch,
                        base_scene,
                        self.policy,
                        task=case.task,
                    )
                    result.patch = routed_patch
                    result.output_svg = apply_patch(case.source_svg, routed_patch)
                    result.details["root_task_router"] = {
                        "routed": True,
                        "task": case.task,
                        "model_bypassed": True,
                    }
                    return result

            visual_context = self._visual_context(case.source_svg)
            scene = build_scene(case.source_svg, visual_stats=visual_context.stats)
            scene_text = json.dumps(scene, indent=2, sort_keys=True)

            can_show_images = bool(model.supports_images)
            can_show_id_map = bool(
                model.supports_images and visual_context.id_map is not None
            )
            selection_images = (
                (
                    visual_context.normal_data_url,
                    visual_context.id_data_url,
                )
                if can_show_id_map
                else (
                    (visual_context.normal_data_url,)
                    if can_show_images
                    else ()
                )
            )
            # The type checker cannot infer that id_data_url is non-None from
            # the id_map test above; filter defensively at the API boundary.
            selection_images = tuple(
                image for image in selection_images if image is not None
            )

            selection_request = ModelRequest(
                target_selection_prompt(
                    case.instruction,
                    scene_text,
                    max_candidates=self.max_candidates,
                    has_images=can_show_images,
                    has_id_map=can_show_id_map,
                ),
                images=selection_images,
                metadata={
                    "request_id": f"{case.case_id}:semantic-select",
                    "visual_context_method": visual_context.method,
                },
                response_schema={
                    **TARGET_SELECTION_SCHEMA,
                    "properties": {
                        **TARGET_SELECTION_SCHEMA["properties"],
                        "targets": {
                            **TARGET_SELECTION_SCHEMA["properties"]["targets"],
                            "maxItems": self.max_candidates,
                        },
                    },
                },
                response_schema_name=TARGET_SELECTION_SCHEMA_NAME,
            )
            result.model_calls += 1
            selection_response = model.generate(selection_request)
            result.raw_responses.append(selection_response.text)
            targets = _selected_targets(
                selection_response.text,
                scene,
                self.max_candidates,
                forbid_root=case.task == "delete",
            )

            preview_records: list[dict[str, Any]] = []
            preview_urls: list[str] = []
            scene_nodes = {node["id"]: node for node in scene["nodes"]}
            for target in targets:
                hidden_svg = _hide_node(case.source_svg, target)
                hidden_png = render_svg_png(
                    hidden_svg,
                    size=self.render_size,
                    background=None,
                )
                difference, difference_png = _counterfactual_difference(
                    visual_context.normal_png,
                    hidden_png,
                    id_map=visual_context.id_map,
                    scene=scene,
                    target=target,
                )
                selected_node = scene_nodes[target]
                record: dict[str, Any] = {
                    "target": target,
                    "target_context": {
                        "tag": selected_node["tag"],
                        "attributes": selected_node.get("attributes", {}),
                        "resolved_style": selected_node.get("resolved_style", {}),
                        "visual": selected_node.get("visual", {}),
                    },
                    "difference": difference,
                }
                if model.supports_images:
                    # Stage-two image 1 is always the original. Each candidate
                    # then contributes its hidden render and derived heatmap.
                    record["without_candidate_image_index"] = len(preview_urls) + 2
                    preview_urls.append(_png_data_url(hidden_png))
                    record["difference_map_image_index"] = len(preview_urls) + 2
                    preview_urls.append(_png_data_url(difference_png))
                preview_records.append(record)

            image_order: list[dict[str, Any]] = []
            patch_images: tuple[str, ...] = ()
            if model.supports_images:
                image_order.append({"index": 1, "kind": "original"})
                for record in preview_records:
                    image_order.extend(
                        (
                            {
                                "index": record["without_candidate_image_index"],
                                "kind": "without_candidate",
                                "target": record["target"],
                            },
                            {
                                "index": record["difference_map_image_index"],
                                "kind": "difference_map",
                                "target": record["target"],
                                "legend": {
                                    "red": "became transparent",
                                    "cyan": "underlayer revealed or repainted",
                                    "green": "became visible",
                                    "yellow": "alpha changed",
                                    "dim_gray": "unchanged reference",
                                },
                            },
                        )
                    )
                patch_images = (
                    visual_context.normal_data_url,
                    *preview_urls,
                )

            final_context = {
                "scene": scene,
                "shortlisted_targets": list(targets),
                "visual_context": {
                    "method": visual_context.method,
                    "unsupported_features": list(
                        visual_context.unsupported_features
                    ),
                },
                "counterfactual_previews": preview_records,
                "image_order": image_order,
                "selection_constraint": (
                    "Every final patch target must be one of shortlisted_targets."
                ),
            }
            result.details = {
                "selected_targets": list(targets),
                "visual_context_method": visual_context.method,
                "counterfactual_previews": preview_records,
                "difference_map_legend": {
                    "red": "became transparent",
                    "cyan": "underlayer revealed or repainted",
                    "green": "became visible",
                    "yellow": "alpha changed",
                    "dim_gray": "unchanged reference",
                },
            }
            patch_schema = (
                DELETE_PATCH_SCHEMA
                if case.task == "delete"
                else patch_json_schema()
            )
            patch_schema_name = (
                DELETE_PATCH_SCHEMA_NAME
                if case.task == "delete"
                else PATCH_SCHEMA_NAME
            )
            patch_request = ModelRequest(
                patch_prompt(
                    case.instruction,
                    "ID-grounded target shortlist and hide-and-rerender checks",
                    json.dumps(final_context, indent=2, sort_keys=True),
                ),
                images=patch_images,
                metadata={"request_id": f"{case.case_id}:semantic-patch"},
                response_schema=patch_schema,
                response_schema_name=patch_schema_name,
            )
            result.model_calls += 1
            patch_response = model.generate(patch_request)
            result.raw_responses.append(patch_response.text)
            result.patch = parse_patch(patch_response.text, repair=True)

            allowed_targets = set(targets)
            final_targets = {
                target
                for operation in result.patch.operations
                for target in operation.targets
            }
            outside = sorted(final_targets - allowed_targets)
            if outside:
                raise PatchError(
                    "final patch targets were not shortlisted: "
                    + ", ".join(outside)
                )

            validate_patch(result.patch, scene, self.policy, task=case.task)
            result.output_svg = apply_patch(case.source_svg, result.patch)

            removed_targets = {
                target
                for operation in result.patch.operations
                if operation.op == "remove_element"
                for target in operation.targets
            }
            completion_details: dict[str, Any] = {
                "enabled": self.qwen_completion,
                "triggered": False,
                "accepted": False,
                "removed_targets": sorted(removed_targets),
            }
            result.details["qwen_completion"] = completion_details
            if self.qwen_completion and removed_targets:
                triggered, triggered_targets = _completion_triggered(
                    preview_records,
                    removed_targets,
                    min_transparent_area_pct=(
                        self.completion_min_transparent_area_pct
                    ),
                    max_revealed_fraction=self.completion_max_revealed_fraction,
                )
                completion_details["triggered"] = triggered
                completion_details["triggered_targets"] = triggered_targets
                if triggered:
                    localized_deletion = result.output_svg
                    try:
                        deletion_png = render_svg_png(
                            localized_deletion,
                            size=self.render_size,
                            background=None,
                        )
                        editable_mask = _change_mask(
                            visual_context.normal_png,
                            deletion_png,
                        )
                        evidence = [
                            record
                            for record in preview_records
                            if record["target"] in triggered_targets
                        ]
                        reconstruction_candidates = (
                            _completion_reconstruction_candidates(
                                case.source_svg,
                                targets,
                                removed_targets,
                                scene_nodes,
                            )
                        )
                        if not reconstruction_candidates:
                            raise PatchError(
                                "no surviving shortlisted node is available "
                                "for Qwen reconstruction"
                            )
                        completion_details["reconstruction_candidates"] = [
                            candidate["target"]
                            for candidate in reconstruction_candidates
                        ]
                        completion_images: tuple[str, ...] = ()
                        if model.supports_images:
                            completion_images = (
                                visual_context.normal_data_url,
                                _png_data_url(deletion_png),
                            )
                        base_prompt = qwen_completion_prompt(
                            case.instruction,
                            localized_deletion,
                            json.dumps(
                                reconstruction_candidates,
                                separators=(",", ":"),
                            ),
                            json.dumps(evidence, separators=(",", ":")),
                        )
                        reconstruction_targets = {
                            candidate["target"]
                            for candidate in reconstruction_candidates
                        }
                        attempts: list[dict[str, Any]] = []
                        completion_details["attempts"] = attempts
                        feedback = ""
                        for attempt_index in range(self.completion_max_attempts):
                            attempt_record: dict[str, Any] = {
                                "attempt": attempt_index + 1,
                                "accepted": False,
                            }
                            attempts.append(attempt_record)
                            prompt = base_prompt + feedback
                            completion_request = ModelRequest(
                                prompt,
                                images=completion_images,
                                metadata={
                                    "request_id": (
                                        f"{case.case_id}:qwen-completion:"
                                        f"{attempt_index + 1}"
                                    ),
                                    "generator": "qwen",
                                },
                                response_schema={
                                    **COMPLETION_SCHEMA,
                                    "properties": {
                                        **COMPLETION_SCHEMA["properties"],
                                        "target": {
                                            "type": "string",
                                            "enum": sorted(
                                                reconstruction_targets
                                            ),
                                        },
                                    },
                                },
                                response_schema_name=COMPLETION_SCHEMA_NAME,
                            )
                            try:
                                result.model_calls += 1
                                completion_response = model.generate(
                                    completion_request
                                )
                                result.raw_responses.append(
                                    completion_response.text
                                )
                                generated_svg = _apply_completion_replacement(
                                    case.source_svg,
                                    localized_deletion,
                                    completion_response.text,
                                    reconstruction_targets,
                                )
                                _validate_generated_completion(
                                    case.source_svg,
                                    generated_svg,
                                )
                                _validate_removed_elements_absent(
                                    case.source_svg,
                                    generated_svg,
                                    removed_targets,
                                )
                                generated_png = render_svg_png(
                                    generated_svg,
                                    size=self.render_size,
                                    background=None,
                                )
                                outside_mse = _outside_edit_mse(
                                    visual_context.normal_png,
                                    generated_png,
                                    editable_mask,
                                )
                                inside_change_mse = _inside_edit_mse(
                                    visual_context.normal_png,
                                    generated_png,
                                    editable_mask,
                                )
                                generated_change_mse = _inside_edit_mse(
                                    deletion_png,
                                    generated_png,
                                    editable_mask,
                                )
                                attempt_record.update(
                                    {
                                        "outside_edit_mse": outside_mse,
                                        "inside_change_mse": inside_change_mse,
                                        "generated_change_mse": (
                                            generated_change_mse
                                        ),
                                    }
                                )
                                if outside_mse > self.completion_max_outside_mse:
                                    raise PatchError(
                                        "Qwen completion changed too much "
                                        "outside the deleted region "
                                        f"({outside_mse:.6f} > "
                                        f"{self.completion_max_outside_mse:.6f})"
                                    )
                                if (
                                    inside_change_mse
                                    < self.completion_min_inside_change_mse
                                ):
                                    raise PatchError(
                                        "Qwen completion visually restored the "
                                        "deleted foreground "
                                        f"({inside_change_mse:.6f} < "
                                        f"{self.completion_min_inside_change_mse:.6f})"
                                    )
                                if (
                                    generated_change_mse
                                    < self.completion_min_generated_change_mse
                                ):
                                    raise PatchError(
                                        "Qwen completion did not reconstruct "
                                        "missing geometry "
                                        f"({generated_change_mse:.6f} < "
                                        f"{self.completion_min_generated_change_mse:.6f})"
                                    )
                                result.output_svg = generated_svg
                                completion_details["accepted"] = True
                                completion_details["accepted_attempt"] = (
                                    attempt_index + 1
                                )
                                completion_details["outside_edit_mse"] = (
                                    outside_mse
                                )
                                completion_details["inside_change_mse"] = (
                                    inside_change_mse
                                )
                                completion_details["generated_change_mse"] = (
                                    generated_change_mse
                                )
                                attempt_record["accepted"] = True
                                break
                            except Exception as attempt_exc:
                                rejection = (
                                    f"{type(attempt_exc).__name__}: "
                                    f"{attempt_exc}"
                                )
                                attempt_record["rejection"] = rejection
                                completion_details["rejection"] = rejection
                                feedback = (
                                    "\n\nThe previous generated candidate was "
                                    f"rejected: {rejection}. Generate a corrected "
                                    "SVG. The deleted foreground element must "
                                    "remain absent, and unrelated pixels must "
                                    "remain unchanged."
                                )
                        completion_details["max_outside_mse"] = (
                            self.completion_max_outside_mse
                        )
                        completion_details["min_inside_change_mse"] = (
                            self.completion_min_inside_change_mse
                        )
                        completion_details["min_generated_change_mse"] = (
                            self.completion_min_generated_change_mse
                        )
                    except Exception as exc:
                        # Completion is a guarded fallback. A rejected or failed
                        # generation never destroys the valid localized deletion.
                        result.output_svg = localized_deletion
                        completion_details["rejection"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        return result
