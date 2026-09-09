"""Deterministic SVG set grounding from source-side group evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from svgpatchlab.core import build_scene
from svgpatchlab.core.geometry import node_analytic_stats

from .graph_features import parse_color


_COLOR_TOKEN = re.compile(
    r"#[0-9a-fA-F]{3,8}|\b(?:black|white|red|green|blue|yellow|cyan|"
    r"magenta|gray|grey|orange|purple)\b",
    re.IGNORECASE,
)
_CONTAINMENT_CUE = re.compile(r"\b(?:inside|within|in|on)\b", re.IGNORECASE)
_CONTAINER_TAGS = {
    "circle": {"circle", "ellipse"},
    "ring": {"circle", "ellipse"},
    "square": {"rect"},
    "box": {"rect"},
    "rectangle": {"rect"},
}


@dataclass(frozen=True)
class StructuralGroupPrediction:
    selected_ids: tuple[str, ...]
    rule: str | None
    evidence: dict[str, Any]

    @property
    def abstained(self) -> bool:
        return not self.selected_ids


def _style(node: Mapping[str, Any], name: str) -> str | None:
    resolved = node.get("resolved_style")
    if isinstance(resolved, Mapping) and resolved.get(name) is not None:
        return str(resolved[name])
    attributes = node.get("attributes")
    if isinstance(attributes, Mapping) and attributes.get(name) is not None:
        return str(attributes[name])
    return None


def _bbox(node: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    visual = node.get("visual")
    raw = visual.get("bbox") if isinstance(visual, Mapping) else None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        return None
    try:
        return tuple(float(item) for item in raw)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


class StructuralGroupGrounder:
    """Select a complete node set when the source reference is unambiguous.

    It deliberately abstains on unsupported language. The caller can then
    route to a learned structural or frozen visual expert.
    """

    def predict(
        self,
        source_svg: str,
        target_reference: str,
        candidate_ids: Sequence[str],
    ) -> StructuralGroupPrediction:
        ordered = tuple(dict.fromkeys(candidate_ids))
        if not ordered:
            return StructuralGroupPrediction((), None, {"reason": "no_candidates"})
        stats = node_analytic_stats(source_svg)
        scene = build_scene(source_svg, visual_stats=stats)
        by_id = {
            str(node["id"]): node
            for node in scene.get("nodes", ())
            if isinstance(node, Mapping) and node.get("id") in ordered
        }

        colors = []
        for token in _COLOR_TOKEN.findall(target_reference):
            parsed = parse_color(token)
            if parsed is not None and parsed not in colors:
                colors.append(parsed)
        if colors:
            paint_by_id = {}
            for node_id in ordered:
                node = by_id.get(node_id, {})
                fill = parse_color(_style(node, "fill"))
                stroke = parse_color(_style(node, "stroke"))
                paint = fill if fill is not None else stroke
                if paint is not None:
                    paint_by_id[node_id] = paint
            selected_set = set()
            matched_colors = []
            for wanted in colors:
                distances = {
                    node_id: sum(
                        (channel - target) ** 2
                        for channel, target in zip(paint, wanted)
                    )
                    for node_id, paint in paint_by_id.items()
                }
                best_distance = min(distances.values()) if distances else None
                matched = [
                    node_id
                    for node_id in ordered
                    if best_distance is not None
                    and abs(distances.get(node_id, float("inf")) - best_distance)
                    <= 1e-12
                ]
                selected_set.update(matched)
                if matched:
                    matched_colors.append(list(paint_by_id[matched[0]]))
            selected = [node_id for node_id in ordered if node_id in selected_set]
            if selected and len(selected) < len(ordered):
                return StructuralGroupPrediction(
                    tuple(selected),
                    "source_paint_set",
                    {
                        "mentioned_colors": [list(color) for color in colors],
                        "matched_colors": matched_colors,
                    },
                )

        lowered = target_reference.lower()
        if _CONTAINMENT_CUE.search(lowered):
            requested_tags: set[str] = set()
            for noun, tags in _CONTAINER_TAGS.items():
                if re.search(rf"\b{noun}\b", lowered):
                    requested_tags.update(tags)
            possible = []
            for container_id in ordered:
                container = by_id.get(container_id, {})
                if requested_tags and str(container.get("tag", "")).lower() not in requested_tags:
                    continue
                outer = _bbox(container)
                if outer is None:
                    continue
                ox, oy, ow, oh = outer
                members = []
                for node_id in ordered:
                    if node_id == container_id:
                        continue
                    inner = _bbox(by_id.get(node_id, {}))
                    if inner is None:
                        continue
                    ix, iy, iw, ih = inner
                    if (
                        ix >= ox
                        and iy >= oy
                        and ix + iw <= ox + ow
                        and iy + ih <= oy + oh
                    ):
                        members.append(node_id)
                if 1 < len(members) < len(ordered):
                    possible.append((len(members), ow * oh, container_id, tuple(members)))
            if possible:
                _, _, container_id, members = min(possible)
                return StructuralGroupPrediction(
                    members,
                    "inside_container_set",
                    {"container_id": container_id},
                )

        return StructuralGroupPrediction((), None, {"reason": "no_unambiguous_rule"})
