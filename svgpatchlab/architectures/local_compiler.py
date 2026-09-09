"""Deterministic patch compilation after a grounder has selected DOM nodes."""

from __future__ import annotations

import re
from collections.abc import Sequence

from svgpatchlab.core.patch import Patch, PatchOperation
from svgpatchlab.types import BenchmarkCase


_COLOR = (
    r"#[0-9A-Fa-f]{3,8}|black|white|red|green|blue|yellow|cyan|magenta|"
    r"gray|grey|orange|purple"
)


def _target_fill(instruction: str) -> str:
    match = re.search(
        rf"\bto\s+(bright\s+)?({_COLOR})\b",
        instruction,
        re.IGNORECASE,
    )
    if match is not None:
        color = match.group(2)
        if match.group(1) and color.lower() == "red":
            return "#E63946"
        return color.lower() if not color.startswith("#") else color

    # Synthetic and natural paraphrases often omit "to": "make the eye bright
    # red".  The last paint term is the destination in that construction.
    matches = list(re.finditer(rf"\b(bright\s+)?({_COLOR})\b", instruction, re.IGNORECASE))
    if not matches:
        raise ValueError("instruction does not specify the replacement fill")
    final = matches[-1]
    color = final.group(2)
    if final.group(1) and color.lower() == "red":
        return "#E63946"
    return color.lower() if not color.startswith("#") else color


def _contour_attributes(instruction: str) -> tuple[tuple[str, str], ...]:
    color_match = re.search(
        rf"({_COLOR})\s+(?:outline|line|stroke)\b",
        instruction,
        re.IGNORECASE,
    )
    stroke = color_match.group(1) if color_match is not None else "black"
    if not stroke.startswith("#"):
        stroke = stroke.lower()
    width_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:units?|px)?\s*(?:wide|width)\b",
        instruction,
        re.IGNORECASE,
    )
    width = width_match.group(1) if width_match is not None else "1"
    return (("stroke", stroke), ("stroke-width", width))


def compile_grounded_patch(
    case: BenchmarkCase,
    targets: Sequence[str],
) -> Patch:
    """Compile an edit intent around already-grounded source node IDs."""

    selected = tuple(dict.fromkeys(str(target) for target in targets))
    if not selected:
        raise ValueError("grounding produced no target nodes")
    if case.task == "change_color":
        attributes = (("fill", _target_fill(case.instruction)),)
        operation = PatchOperation("set_attributes", selected, attributes)
        return Patch((operation,))
    if case.task == "set_contour":
        operation = PatchOperation(
            "set_attributes", selected, _contour_attributes(case.instruction)
        )
        return Patch((operation,))
    if case.task == "delete":
        return Patch((PatchOperation("remove_element", selected),), version=2)
    raise ValueError(
        f"Graph-MoE deterministic compiler does not support task {case.task!r}"
    )
