from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

from svgpatchlab.core.xml import index_tree, local_name, parse_svg, serialize_svg
from svgpatchlab.eval.render import (
    ensure_renderer,
    png_data_url,
    render_svg_png,
)


RESOURCE_TAGS = {
    "clipPath",
    "defs",
    "filter",
    "linearGradient",
    "marker",
    "mask",
    "pattern",
    "radialGradient",
    "stop",
    "style",
    "symbol",
}


@dataclass(frozen=True)
class NodeRender:
    node_id: str
    data_url: str
    bbox: tuple[int, int, int, int]
    visible_pixels: int


def _descendants(children_by_parent: dict[str | None, list[str]], node_id: str) -> set[str]:
    result: set[str] = set()
    stack = [node_id]
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(children_by_parent.get(current, ()))
    return result


def _ancestors(parent_by_id: dict[str, str | None], node_id: str) -> set[str]:
    result: set[str] = set()
    current = parent_by_id.get(node_id)
    while current is not None:
        result.add(current)
        current = parent_by_id.get(current)
    return result


def _resource_subtrees(
    indexed_ids: Iterable[str],
    children_by_parent: dict[str | None, list[str]],
    tag_by_id: dict[str, str],
) -> set[str]:
    result: set[str] = set()
    for node_id in indexed_ids:
        if tag_by_id[node_id] in RESOURCE_TAGS:
            result.update(_descendants(children_by_parent, node_id))
    return result


def _isolated_node_svg(svg: str, node_id: str) -> str:
    root = parse_svg(svg)
    indexed = index_tree(root)
    node_ids = {node.node_id for node in indexed}
    if node_id not in node_ids:
        raise ValueError(f"unknown node id: {node_id}")
    if node_id == "n0":
        return serialize_svg(root)

    parent_by_id = {node.node_id: node.parent_id for node in indexed}
    children_by_parent: dict[str | None, list[str]] = {}
    tag_by_id = {node.node_id: local_name(node.element.tag) for node in indexed}
    for node in indexed:
        children_by_parent.setdefault(node.parent_id, []).append(node.node_id)

    visible = (
        _descendants(children_by_parent, node_id)
        | _ancestors(parent_by_id, node_id)
        | _resource_subtrees(node_ids, children_by_parent, tag_by_id)
    )

    for node in indexed:
        if node.node_id not in visible:
            node.element.attrib["display"] = "none"
    return serialize_svg(root)


def _rgba_panel(image):
    from PIL import Image

    panel = Image.new("RGB", image.size, "white")
    panel.paste(image, mask=image.getchannel("A"))
    return panel


def _alpha_bbox_and_pixels(image) -> tuple[tuple[int, int, int, int] | None, int]:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    histogram = alpha.histogram()
    visible_pixels = sum(count for value, count in enumerate(histogram) if value > 0)
    if bbox is None:
        return None, visible_pixels
    left, top, right, bottom = bbox
    return (left, top, right - left, bottom - top), visible_pixels


def render_node_comparison(svg: str, node_id: str, size: int = 384) -> NodeRender | None:
    """Render full, isolated, and highlighted views of one SVG node."""
    ensure_renderer()
    from PIL import Image, ImageDraw

    full_png = render_svg_png(svg, size=size, background="white")
    isolated_svg = _isolated_node_svg(svg, node_id)
    isolated_png = render_svg_png(isolated_svg, size=size, background=None)

    full = Image.open(io.BytesIO(full_png)).convert("RGBA")
    isolated = Image.open(io.BytesIO(isolated_png)).convert("RGBA")
    bbox, visible_pixels = _alpha_bbox_and_pixels(isolated)
    if bbox is None:
        return None

    highlighted = _rgba_panel(full)
    draw = ImageDraw.Draw(highlighted)
    left, top, width, height = bbox
    stroke_width = max(3, size // 96)
    draw.rectangle(
        (left, top, left + width, top + height),
        outline=(255, 0, 255),
        width=stroke_width,
    )

    label_height = max(22, size // 14)
    canvas = Image.new("RGB", (size * 3, size + label_height), "white")
    canvas.paste(_rgba_panel(full), (0, label_height))
    canvas.paste(_rgba_panel(isolated), (size, label_height))
    canvas.paste(highlighted, (size * 2, label_height))

    label_draw = ImageDraw.Draw(canvas)
    labels = ("full svg", f"isolated {node_id}", "highlighted in context")
    for index, label in enumerate(labels):
        label_draw.text((index * size + 6, 4), label, fill=(20, 20, 20))

    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return NodeRender(
        node_id=node_id,
        data_url=png_data_url(output.getvalue()),
        bbox=bbox,
        visible_pixels=visible_pixels,
    )
