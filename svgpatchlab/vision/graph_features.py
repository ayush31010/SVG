"""Feature and relation extraction for instruction-conditioned SVG graphs.

The graph deliberately keeps editable DOM nodes as its prediction units.  It
does not raster-segment the image and then try to recover source ownership.
Render-derived statistics can be attached to the scene before this module is
called; analytic statistics use the exact same feature contract for ablations.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


EXPERT_NAMES = ("attribute", "spatial", "semantic")
ATTRIBUTE_EXPERT, SPATIAL_EXPERT, SEMANTIC_EXPERT = range(3)

TAG_NAMES = (
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polygon",
    "polyline",
    "text",
    "g",
    "other",
)
EDGE_NAMES = (
    "self",
    "parent",
    "child",
    "same_parent",
    "left",
    "right",
    "above",
    "below",
    "overlap",
    "contains",
    "inside",
    "nearest",
)

TAG_SLICE = slice(0, 10)
STYLE_SLICE = slice(10, 19)
GEOMETRY_SLICE = slice(19, 28)
STRUCTURE_SLICE = slice(28, 33)
NODE_FEATURE_DIM = 33

_DEFINITION_TAGS = frozenset(
    {
        "defs",
        "desc",
        "filter",
        "lineargradient",
        "marker",
        "mask",
        "metadata",
        "pattern",
        "radialgradient",
        "script",
        "stop",
        "style",
        "symbol",
        "title",
    }
)
_TARGETABLE_TAGS = frozenset(TAG_NAMES[:-1]) | {"use"}

_NAMED_RGB = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 0.5, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "orange": (1.0, 0.647, 0.0),
    "purple": (0.5, 0.0, 0.5),
}

_SEMANTIC_TERMS = re.compile(
    r"\b(?:eye|mouth|nose|hat|head|body|wheel|petal|flower|face|robot|"
    r"tongue|ear|hand|foot|arm|leg|lens|handle|rim|tail|wing|cloud|sun|moon)\b",
    re.IGNORECASE,
)
_SPATIAL_TERMS = re.compile(
    r"\b(?:left|right|top|bottom|leftmost|rightmost|topmost|bottommost|"
    r"upper|lower|above|below|under|over|"
    r"middle|center|centre|corner|row|column|smallest|largest|biggest|"
    r"nearest|closest|farthest|inside|outside|between)\b",
    re.IGNORECASE,
)
_ATTRIBUTE_REFERENCE = re.compile(
    r"\b(?:with|having)\s+(?:an?\s+)?(?:#[0-9a-f]{3,8}|"
    + "|".join(_NAMED_RGB)
    + r")\s+(?:fill|color|colour)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SVGGraph:
    """A small dependency-free graph ready to be converted to tensors."""

    node_ids: tuple[str, ...]
    node_features: tuple[tuple[float, ...], ...]
    adjacency: tuple[tuple[tuple[float, ...], ...], ...]
    metadata: tuple[dict[str, Any], ...]

    @property
    def node_count(self) -> int:
        return len(self.node_ids)


def parse_color(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"none", "transparent", "currentcolor", "inherit"}:
        return None
    if normalized in _NAMED_RGB:
        return _NAMED_RGB[normalized]
    functional = re.fullmatch(
        r"rgba?\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*[, ]+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)\s*[, ]+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)"
        r"(?:\s*[,/]\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?)?\s*\)",
        normalized,
    )
    if functional is not None:
        channels = []
        for raw in functional.groups():
            if raw.endswith("%"):
                channel = float(raw[:-1]) / 100.0
            else:
                channel = float(raw) / 255.0
            channels.append(max(0.0, min(1.0, channel)))
        return tuple(channels)  # type: ignore[return-value]
    match = re.fullmatch(r"#([0-9a-f]{3}|[0-9a-f]{4}|[0-9a-f]{6}|[0-9a-f]{8})", normalized)
    if match is None:
        return None
    digits = match.group(1)
    if len(digits) in {3, 4}:
        digits = "".join(character * 2 for character in digits)
    return tuple(int(digits[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def infer_reference_type(instruction: str) -> str:
    """Return the supervision class for the sparse expert router.

    An explicit source-paint reference takes precedence because it already
    identifies the target even when the benchmark preamble names an object
    such as a face or sun. Semantic nouns then take precedence over modifiers
    such as ``left eye``. Unknown descriptions are assigned to the semantic
    expert, whose feature view is the least restrictive.
    """

    if _ATTRIBUTE_REFERENCE.search(instruction):
        return "attribute"
    if _SEMANTIC_TERMS.search(instruction):
        return "semantic"
    if _SPATIAL_TERMS.search(instruction):
        return "spatial"
    return "semantic"


def expert_index(name: str) -> int:
    try:
        return EXPERT_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown graph expert: {name}") from exc


def _style_value(node: Mapping[str, Any], name: str) -> str | None:
    attributes = node.get("attributes")
    if isinstance(attributes, Mapping) and isinstance(attributes.get(name), str):
        return str(attributes[name])
    resolved = node.get("resolved_style")
    if isinstance(resolved, Mapping) and isinstance(resolved.get(name), str):
        return str(resolved[name])
    return None


def _direct_style_value(node: Mapping[str, Any], name: str) -> str | None:
    attributes = node.get("attributes")
    if not isinstance(attributes, Mapping):
        return None
    direct = attributes.get(name)
    if isinstance(direct, str):
        return direct
    style = attributes.get("style")
    if isinstance(style, str):
        for declaration in style.split(";"):
            if ":" not in declaration:
                continue
            property_name, value = declaration.split(":", 1)
            if property_name.strip().lower() == name:
                return value.strip()
    return None


def _number(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _viewbox(scene: Mapping[str, Any]) -> tuple[float, float, float, float]:
    root_id = scene.get("root_id", "n0")
    root = next(
        (node for node in scene.get("nodes", ()) if node.get("id") == root_id),
        None,
    )
    if isinstance(root, Mapping):
        attributes = root.get("attributes")
        raw = attributes.get("viewBox") if isinstance(attributes, Mapping) else None
        if isinstance(raw, str):
            parts = [part for part in re.split(r"[\s,]+", raw.strip()) if part]
            if len(parts) == 4:
                values = tuple(_number(part, 0.0) for part in parts)
                if values[2] > 0 and values[3] > 0:
                    return values  # type: ignore[return-value]
    return 0.0, 0.0, 100.0, 100.0


def _targetable_nodes(scene: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    root_id = scene.get("root_id", "n0")
    blocked: set[str] = set()
    result: list[Mapping[str, Any]] = []
    for node in scene.get("nodes", ()):
        if not isinstance(node, Mapping) or not isinstance(node.get("id"), str):
            continue
        node_id = str(node["id"])
        tag = str(node.get("tag", "")).lower()
        parent = node.get("parent")
        if parent in blocked or tag in _DEFINITION_TAGS:
            blocked.add(node_id)
            continue
        if node_id == root_id or tag not in _TARGETABLE_TAGS:
            continue
        visual = node.get("visual")
        if isinstance(visual, Mapping) and visual.get("visible") is False:
            continue
        result.append(node)
    return result


def _node_feature(
    node: Mapping[str, Any],
    viewbox: tuple[float, float, float, float],
) -> tuple[list[float], dict[str, Any]]:
    values = [0.0] * NODE_FEATURE_DIM
    tag = str(node.get("tag", "")).lower()
    tag_index = TAG_NAMES.index(tag) if tag in TAG_NAMES[:-1] else len(TAG_NAMES) - 1
    values[tag_index] = 1.0

    fill = parse_color(_style_value(node, "fill"))
    stroke = parse_color(_style_value(node, "stroke"))
    direct_fill = parse_color(_direct_style_value(node, "fill"))
    direct_stroke = parse_color(_direct_style_value(node, "stroke"))
    if fill is not None:
        values[10:13] = fill
        values[16] = 1.0 if direct_fill is not None else 0.0
    if stroke is not None:
        values[13:16] = stroke
        values[17] = 1.0 if direct_stroke is not None else 0.0
    values[18] = max(0.0, min(1.0, _number(_style_value(node, "opacity"), 1.0)))

    visual = node.get("visual")
    bbox = visual.get("bbox") if isinstance(visual, Mapping) else None
    box: tuple[float, float, float, float] | None = None
    if isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 4:
        parsed = tuple(_number(item, float("nan")) for item in bbox)
        if all(math.isfinite(item) for item in parsed) and parsed[2] >= 0 and parsed[3] >= 0:
            box = parsed  # type: ignore[assignment]
    vb_x, vb_y, vb_width, vb_height = viewbox
    if box is not None:
        x, y, width, height = box
        values[19] = (x - vb_x) / vb_width
        values[20] = (y - vb_y) / vb_height
        values[21] = width / vb_width
        values[22] = height / vb_height
        values[23] = (x + width / 2.0 - vb_x) / vb_width
        values[24] = (y + height / 2.0 - vb_y) / vb_height
        area_pct = visual.get("area_pct", 100.0 * width * height / (vb_width * vb_height))
        values[25] = max(0.0, min(1.0, _number(area_pct, 0.0) / 100.0))
        values[26] = math.tanh(math.log(max(width, 1e-6) / max(height, 1e-6)))
        values[27] = 1.0

    depth = max(0.0, _number(node.get("depth"), 0.0))
    child_index = max(0.0, _number(node.get("child_index"), 0.0))
    values[28] = min(depth, 16.0) / 16.0
    values[29] = min(child_index, 32.0) / 32.0
    values[30] = 1.0 if tag == "g" else 0.0
    values[31] = 1.0 if bool(node.get("text")) else 0.0
    values[32] = 0.0 if isinstance(visual, Mapping) and visual.get("visible") is False else 1.0

    metadata = {
        "id": str(node["id"]),
        "tag": tag,
        "parent": node.get("parent"),
        "bbox": list(box) if box is not None else None,
        "fill": _style_value(node, "fill"),
        "direct_fill": _direct_style_value(node, "fill"),
        "stroke": _style_value(node, "stroke"),
    }
    return values, metadata


def _overlap(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    width = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    height = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    return width * height


def build_svg_graph(scene: Mapping[str, Any]) -> SVGGraph:
    """Convert a scene skeleton with optional visual stats to an SVG graph."""

    nodes = _targetable_nodes(scene)
    if not nodes:
        raise ValueError("SVG graph has no targetable nodes")
    viewbox = _viewbox(scene)
    node_ids: list[str] = []
    features: list[tuple[float, ...]] = []
    metadata: list[dict[str, Any]] = []
    for node in nodes:
        values, item = _node_feature(node, viewbox)
        node_ids.append(str(node["id"]))
        features.append(tuple(values))
        metadata.append(item)

    count = len(node_ids)
    relation = [
        [[0.0 for _ in range(count)] for _ in range(count)]
        for _ in EDGE_NAMES
    ]
    relation[EDGE_NAMES.index("self")] = [
        [1.0 if row == column else 0.0 for column in range(count)]
        for row in range(count)
    ]
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    for target, item in enumerate(metadata):
        parent = item["parent"]
        if parent in index:
            source = index[str(parent)]
            relation[EDGE_NAMES.index("parent")][target][source] = 1.0
            relation[EDGE_NAMES.index("child")][source][target] = 1.0
    for left in range(count):
        for right in range(count):
            if left == right:
                continue
            if metadata[left]["parent"] == metadata[right]["parent"]:
                relation[EDGE_NAMES.index("same_parent")][left][right] = 1.0
            left_box = metadata[left]["bbox"]
            right_box = metadata[right]["bbox"]
            if left_box is None or right_box is None:
                continue
            lcx = left_box[0] + left_box[2] / 2.0
            lcy = left_box[1] + left_box[3] / 2.0
            rcx = right_box[0] + right_box[2] / 2.0
            rcy = right_box[1] + right_box[3] / 2.0
            if rcx < lcx:
                relation[EDGE_NAMES.index("left")][left][right] = 1.0
            if rcx > lcx:
                relation[EDGE_NAMES.index("right")][left][right] = 1.0
            if rcy < lcy:
                relation[EDGE_NAMES.index("above")][left][right] = 1.0
            if rcy > lcy:
                relation[EDGE_NAMES.index("below")][left][right] = 1.0
            intersection = _overlap(left_box, right_box)
            if intersection > 0.0:
                relation[EDGE_NAMES.index("overlap")][left][right] = 1.0
            left_area = left_box[2] * left_box[3]
            right_area = right_box[2] * right_box[3]
            if right_area > 0.0 and math.isclose(intersection, right_area, rel_tol=1e-6, abs_tol=1e-9):
                relation[EDGE_NAMES.index("contains")][left][right] = 1.0
            if left_area > 0.0 and math.isclose(intersection, left_area, rel_tol=1e-6, abs_tol=1e-9):
                relation[EDGE_NAMES.index("inside")][left][right] = 1.0

    for target in range(count):
        target_box = metadata[target]["bbox"]
        if target_box is None or count == 1:
            continue
        tx = target_box[0] + target_box[2] / 2.0
        ty = target_box[1] + target_box[3] / 2.0
        distances = []
        for source in range(count):
            source_box = metadata[source]["bbox"]
            if source == target or source_box is None:
                continue
            sx = source_box[0] + source_box[2] / 2.0
            sy = source_box[1] + source_box[3] / 2.0
            distances.append(((tx - sx) ** 2 + (ty - sy) ** 2, source))
        if distances:
            relation[EDGE_NAMES.index("nearest")][target][min(distances)[1]] = 1.0

    return SVGGraph(
        node_ids=tuple(node_ids),
        node_features=tuple(features),
        adjacency=tuple(
            tuple(tuple(row) for row in matrix) for matrix in relation
        ),
        metadata=tuple(metadata),
    )
