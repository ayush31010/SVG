from __future__ import annotations

import re
from typing import Any

from svgpatchlab.core.patch import Patch, PatchOperation
from svgpatchlab.types import BenchmarkCase


ROUTABLE_ROOT_TASKS = frozenset(
    {"upside_down", "transparency", "crop_to_half"}
)

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _format_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.12g}"


def _viewbox(scene: dict[str, Any]) -> tuple[float, float, float, float] | None:
    root_id = scene.get("root_id")
    root = next(
        (node for node in scene.get("nodes", ()) if node.get("id") == root_id),
        None,
    )
    if root is None:
        return None
    raw = root.get("attributes", {}).get("viewBox")
    if not isinstance(raw, str):
        return None
    parts = [part for part in re.split(r"[\s,]+", raw.strip()) if part]
    if len(parts) != 4 or any(re.fullmatch(_NUMBER, part) is None for part in parts):
        return None
    x, y, width, height = (float(part) for part in parts)
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _set_root_attribute(
    scene: dict[str, Any],
    name: str,
    value: str,
) -> Patch:
    return Patch(
        operations=(
            PatchOperation(
                op="set_attributes",
                targets=(scene["root_id"],),
                attributes=((name, value),),
            ),
        ),
        version=1,
    )


def _transparency_patch(
    case: BenchmarkCase,
    scene: dict[str, Any],
) -> Patch | None:
    instruction = case.instruction.lower()
    if re.search(r"\b(?:half|one[- ]half|50\s*%)\b", instruction):
        opacity = 0.5
    else:
        match = re.search(
            r"\bopacity\s+(?:to|of)\s+(\d+(?:\.\d+)?)\s*%",
            instruction,
        )
        if match is None:
            return None
        opacity = float(match.group(1)) / 100.0
    if not 0.0 <= opacity <= 1.0:
        return None
    return _set_root_attribute(scene, "opacity", _format_number(opacity))


def _crop_patch(
    case: BenchmarkCase,
    scene: dict[str, Any],
) -> Patch | None:
    box = _viewbox(scene)
    if box is None:
        return None
    instruction = case.instruction.lower()
    keep_match = re.search(r"\bkeep\s+(?:the\s+)?(left|right|top|bottom)\s+half\b", instruction)
    if keep_match is None:
        return None
    keep = keep_match.group(1)
    x, y, width, height = box
    if keep == "left":
        width /= 2.0
    elif keep == "right":
        x += width / 2.0
        width /= 2.0
    elif keep == "top":
        height /= 2.0
    else:
        y += height / 2.0
        height /= 2.0
    value = " ".join(_format_number(item) for item in (x, y, width, height))
    return _set_root_attribute(scene, "viewBox", value)


def _upside_down_patch(
    case: BenchmarkCase,
    scene: dict[str, Any],
) -> Patch | None:
    if not re.search(r"\b(?:upside[- ]down|flip\b.*\bvertically)\b", case.instruction.lower()):
        return None
    box = _viewbox(scene)
    if box is None:
        return None
    root = next(
        node for node in scene["nodes"] if node["id"] == scene["root_id"]
    )
    if "transform" in root.get("attributes", {}):
        # Composing an existing root transform is order-sensitive. Leave that
        # uncommon case to the model instead of silently changing semantics.
        return None
    _, y, _, height = box
    translate_y = 2.0 * y + height
    transform = f"translate(0,{_format_number(translate_y)}) scale(1,-1)"
    return _set_root_attribute(scene, "transform", transform)


def compile_root_task_patch(
    case: BenchmarkCase,
    scene: dict[str, Any],
) -> Patch | None:
    """Compile unambiguous whole-canvas instructions without a model call.

    Returning ``None`` is deliberate: it means the instruction or canvas is
    not specific enough for a safe deterministic rewrite, so the caller should
    continue through its normal model path.
    """

    if case.task == "transparency":
        return _transparency_patch(case, scene)
    if case.task == "crop_to_half":
        return _crop_patch(case, scene)
    if case.task == "upside_down":
        return _upside_down_patch(case, scene)
    return None
