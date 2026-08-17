"""Reusable visual evidence for closed-choice SVG node grounding.

Each candidate is rendered in three complementary ways:

* ``full_context_png`` keeps the whole drawing and highlights the candidate;
* ``local_crop_png`` enlarges the same highlighted region with nearby context;
* ``isolated_png`` shows only the selected node/subtree on transparency.

The contact-sheet helper lays these views out in the caller's candidate order.
It deliberately knows nothing about prompts, target selection, or training.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from svgpatchlab.core.xml import (
    SVG_NAMESPACE,
    index_tree,
    local_name,
    parse_svg,
    serialize_svg,
)
from svgpatchlab.eval.render import RendererUnavailable, render_svg_png


PixelBox = tuple[int, int, int, int]
ViewBox = tuple[float, float, float, float]


_DEFINITION_CONTAINERS = {
    "clipPath",
    "defs",
    "filter",
    "linearGradient",
    "marker",
    "mask",
    "pattern",
    "radialGradient",
    "style",
    "symbol",
}


def _png_data_url(png: bytes) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@dataclass(frozen=True)
class CandidateView:
    """Three aligned raster views for one DOM candidate.

    ``bbox`` and ``crop_box`` use exclusive pixel coordinates in the square
    full-context render. ``bbox`` is ``None`` when the isolated subtree paints
    no pixels. All three PNGs are ``size`` by ``size``.
    """

    node_id: str
    full_context_png: bytes
    local_crop_png: bytes
    isolated_png: bytes
    bbox: PixelBox | None
    crop_box: PixelBox
    size: int

    @property
    def visible(self) -> bool:
        return self.bbox is not None

    @property
    def highlighted_png(self) -> bytes:
        """Alias that makes the purpose of ``full_context_png`` explicit."""
        return self.full_context_png

    @property
    def data_urls(self) -> dict[str, str]:
        return {
            "full_context": _png_data_url(self.full_context_png),
            "local_crop": _png_data_url(self.local_crop_png),
            "isolated": _png_data_url(self.isolated_png),
        }


@dataclass(frozen=True)
class CandidateContactSheet:
    """A labeled PNG contact sheet and its exact candidate/label ordering."""

    png: bytes
    candidate_ids: tuple[str, ...]
    labels: dict[str, str]
    views: tuple[CandidateView, ...]
    columns: int
    show_node_ids: bool

    @property
    def data_url(self) -> str:
        return _png_data_url(self.png)


@dataclass(frozen=True)
class CandidateEvidence:
    """High-fidelity, separately consumable evidence for one candidate.

    Unlike :class:`CandidateView`, ``vector_crop_png`` is rasterized directly
    from an SVG whose viewBox is the padded candidate region.  It therefore
    preserves vector detail instead of enlarging pixels from a whole-scene
    raster.  ``mask_png`` is an opaque black/white ownership cue that remains
    legible when the candidate itself is pale or transparent.

    ``metadata`` deliberately excludes the internal DOM node ID.  A caller can
    associate this object with a model-visible choice label while retaining
    ``node_id`` only in its private choice map.
    """

    node_id: str
    full_context_png: bytes
    vector_crop_png: bytes
    isolated_crop_png: bytes
    mask_png: bytes
    bbox: PixelBox | None
    bbox_viewbox: ViewBox | None
    crop_viewbox: ViewBox
    size: int
    metadata: dict[str, Any]
    appearance_digest: str
    mask_digest: str

    @property
    def visible(self) -> bool:
        return self.bbox is not None

    @property
    def images(self) -> tuple[bytes, bytes, bytes, bytes]:
        """Images in context, vector-crop, isolated-crop, mask order."""
        return (
            self.full_context_png,
            self.vector_crop_png,
            self.isolated_crop_png,
            self.mask_png,
        )

    @property
    def data_urls(self) -> dict[str, str]:
        return {
            "full_context": _png_data_url(self.full_context_png),
            "vector_crop": _png_data_url(self.vector_crop_png),
            "isolated_crop": _png_data_url(self.isolated_crop_png),
            "mask": _png_data_url(self.mask_png),
        }


@dataclass(frozen=True)
class CandidateEvidenceSheet:
    """A high-resolution sheet built from direct vector candidate evidence.

    Each row/card is kept independent until the final composition, so local
    crops are never enlarged from a low-resolution whole-scene raster.  The
    sheet uses context, direct vector crop, and binary mask panels; the latter
    remains unambiguous for pale, transparent, or heavily occluded artwork.
    """

    png: bytes
    candidate_ids: tuple[str, ...]
    labels: dict[str, str]
    evidence: tuple[CandidateEvidence, ...]
    columns: int

    @property
    def data_url(self) -> str:
        return _png_data_url(self.png)


def _pillow():
    try:
        from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont
    except ImportError as exc:  # pragma: no cover - exercised without eval extra
        raise RendererUnavailable(
            "candidate views require Pillow: pip install -e '.[eval]'"
        ) from exc
    return Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def _validate_size(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")


def _validate_padding(crop_padding: float) -> None:
    if not isinstance(crop_padding, (int, float)) or not math.isfinite(crop_padding):
        raise ValueError("crop_padding must be a finite non-negative number")
    if crop_padding < 0:
        raise ValueError("crop_padding must be a finite non-negative number")


def _validated_candidates(svg: str, candidate_ids: Sequence[str]):
    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError("candidate_ids must be a sequence of node IDs, not a string")
    ordered = tuple(candidate_ids)
    if not ordered:
        raise ValueError("candidate_ids must not be empty")
    if any(not isinstance(node_id, str) for node_id in ordered):
        raise TypeError("candidate_ids must contain only strings")
    duplicates = tuple(
        dict.fromkeys(
            node_id for index, node_id in enumerate(ordered) if node_id in ordered[:index]
        )
    )
    if duplicates:
        raise ValueError(f"duplicate candidate node IDs: {', '.join(duplicates)}")

    root = parse_svg(svg)
    indexed = index_tree(root)
    known = {node.node_id for node in indexed}
    unknown = tuple(node_id for node_id in ordered if node_id not in known)
    if unknown:
        raise ValueError(f"candidate node IDs not found in SVG: {', '.join(unknown)}")
    return root, indexed, ordered


def _descendant_ids(indexed, node_id: str) -> set[str]:
    by_id = {node.node_id: node for node in indexed}
    descendants: set[str] = set()
    for candidate in indexed:
        current: str | None = candidate.node_id
        while current is not None:
            if current == node_id:
                descendants.add(candidate.node_id)
                break
            current = by_id[current].parent_id
    return descendants


def _ancestor_ids(indexed, node_ids: set[str]) -> set[str]:
    by_id = {node.node_id: node for node in indexed}
    ancestors: set[str] = set()
    for node_id in node_ids:
        current = by_id[node_id].parent_id
        while current is not None:
            ancestors.add(current)
            current = by_id[current].parent_id
    return ancestors


def _definition_content_ids(indexed) -> set[str]:
    """Return definition/style nodes and all of their descendants."""
    content: set[str] = set()
    inside_definition: dict[str, bool] = {}
    for node in indexed:
        in_definition = (
            inside_definition.get(node.parent_id or "", False)
            or local_name(node.element.tag) in _DEFINITION_CONTAINERS
        )
        inside_definition[node.node_id] = in_definition
        if in_definition:
            content.add(node.node_id)
    return content


def _definition_support_ids(indexed) -> set[str]:
    """Return definition/style nodes plus the ancestors needed to reach them."""
    support = _definition_content_ids(indexed)
    support.update(_ancestor_ids(indexed, support))
    return support


def _element_document_id(element: Any) -> str | None:
    for name, value in element.attrib.items():
        if local_name(name) == "id" and value:
            return value
    return None


def _use_reference(element: Any) -> str | None:
    if local_name(element.tag) != "use":
        return None
    for name, value in element.attrib.items():
        if local_name(name) != "href":
            continue
        value = value.strip()
        if value.startswith("#") and len(value) > 1:
            return value[1:]
    return None


def _ordinary_use_reference_roots(
    indexed,
    selected: set[str],
    keep: set[str],
) -> tuple[str, ...]:
    """Find ordinary referenced subtrees that must be moved under ``defs``.

    A selected ``use`` needs its referenced element to remain renderable, but
    leaving an ordinary source in place also paints it at its original
    location. Moving the smallest referenced roots under ``defs`` preserves
    fragment lookup without exposing those originals. References already in a
    definition container, or already inside the selected subtree, need no move.
    """
    by_node_id = {node.node_id: node for node in indexed}
    by_document_id: dict[str, str] = {}
    for node in indexed:
        document_id = _element_document_id(node.element)
        if document_id is not None:
            by_document_id.setdefault(document_id, node.node_id)

    definition_content = _definition_content_ids(indexed)
    pending = [
        node.node_id
        for node in indexed
        if node.node_id in selected and _use_reference(node.element) is not None
    ]
    seen_uses: set[str] = set()
    referenced: set[str] = set()
    while pending:
        use_id = pending.pop()
        if use_id in seen_uses:
            continue
        seen_uses.add(use_id)
        reference = _use_reference(by_node_id[use_id].element)
        source_id = by_document_id.get(reference or "")
        if source_id is None:
            continue

        source_subtree = _descendant_ids(indexed, source_id)
        pending.extend(
            candidate_id
            for candidate_id in source_subtree
            if _use_reference(by_node_id[candidate_id].element) is not None
        )
        if source_id not in definition_content and source_id not in keep:
            referenced.add(source_id)

    # If one referenced source contains another, moving the ancestor already
    # carries the descendant (and its ID) into the definition container.
    roots: list[str] = []
    for node in indexed:
        if node.node_id not in referenced:
            continue
        current = node.parent_id
        nested = False
        while current is not None:
            if current in referenced:
                nested = True
                break
            current = by_node_id[current].parent_id
        if not nested:
            roots.append(node.node_id)
    return tuple(roots)


def _move_reference_roots_to_defs(
    root_copy: Any,
    copy_indexed,
    reference_roots: tuple[str, ...],
) -> set[str]:
    """Move referenced ordinary sources under ``defs`` and return their IDs."""
    if not reference_roots:
        return set()
    by_id = {node.node_id: node for node in copy_indexed}
    defs = next(
        (child for child in list(root_copy) if local_name(child.tag) == "defs"),
        None,
    )
    if defs is None:
        defs = ET.Element(f"{{{SVG_NAMESPACE}}}defs")
        root_copy.insert(0, defs)

    moved: set[str] = set()
    for source_id in reference_roots:
        source = by_id[source_id]
        if source.parent_id is None:
            continue
        parent = by_id[source.parent_id].element
        try:
            parent.remove(source.element)
        except ValueError:  # already moved as part of another source subtree
            continue
        source.element.tail = None
        defs.append(source.element)
        moved.update(_descendant_ids(copy_indexed, source_id))
    return moved


def _trim_ancestor_text(
    copy_indexed,
    target_id: str,
    ancestors: set[str],
) -> None:
    """Remove character data outside the selected element boundary.

    ElementTree stores text before a child on the parent and text after a child
    on the child's ``tail``. Those strings are not indexed nodes, so merely
    hiding sibling elements is insufficient when selecting a nested ``tspan``.
    """
    by_id = {node.node_id: node.element for node in copy_indexed}
    for ancestor_id in ancestors:
        ancestor = by_id[ancestor_id]
        ancestor.text = None
        for child in list(ancestor):
            child.tail = None
    # The selected root's own tail belongs to its parent, not its subtree.
    by_id[target_id].tail = None


def _hide_element(element: Any) -> None:
    original = element.attrib.get("style", "").strip().rstrip(";")
    hidden = "display:none!important"
    element.attrib["style"] = f"{original};{hidden}" if original else hidden


def isolate_svg_subtree(svg: str, node_id: str) -> str:
    """Return an SVG in which only ``node_id`` and its subtree can paint.

    Ancestors remain intact so inherited paint, transforms, clipping, and
    opacity still apply. Definition/style subtrees are also retained so URL
    paints, masks, clip paths, symbols, and CSS references keep working.
    """
    root, indexed, ordered = _validated_candidates(svg, (node_id,))
    target_id = ordered[0]
    selected = _descendant_ids(indexed, target_id)
    ancestors = _ancestor_ids(indexed, {target_id})
    keep = selected | ancestors | _definition_support_ids(indexed)
    reference_roots = _ordinary_use_reference_roots(indexed, selected, keep)

    root_copy = copy.deepcopy(root)
    copy_indexed = index_tree(root_copy)
    moved = _move_reference_roots_to_defs(
        root_copy,
        copy_indexed,
        reference_roots,
    )
    _trim_ancestor_text(copy_indexed, target_id, ancestors)
    for node in copy_indexed:
        if node.node_id not in keep and node.node_id not in moved:
            _hide_element(node.element)
    return serialize_svg(root_copy)


def _decode_png(png: bytes):
    Image, *_ = _pillow()
    try:
        with Image.open(io.BytesIO(png)) as source:
            return source.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise RendererUnavailable("renderer returned an invalid PNG") from exc


def _encode_png(image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _resampling_lanczos(Image):
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:  # pragma: no cover - Pillow < 9.1
        return Image.LANCZOS


def _fit_square(image, box: PixelBox, size: int):
    Image, *_ = _pillow()
    cropped = image.crop(box)
    if cropped.size == (size, size):
        return cropped
    return cropped.resize((size, size), _resampling_lanczos(Image))


def _square_crop_box(
    bbox: PixelBox | None,
    canvas_width: int,
    canvas_height: int,
    crop_padding: float,
) -> PixelBox:
    if bbox is None:
        return (0, 0, canvas_width, canvas_height)

    x0, y0, x1, y1 = bbox
    extent = max(x1 - x0, y1 - y0)
    # A small minimum gives tiny marks enough surrounding context to remain
    # recognizable after they are enlarged.
    side = max(8, int(math.ceil(extent * (1.0 + 2.0 * crop_padding))))
    side = min(side, canvas_width, canvas_height)
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = min(max(left, 0), canvas_width - side)
    top = min(max(top, 0), canvas_height - side)
    return (left, top, left + side, top + side)


def _numeric_length(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized.endswith("px"):
        normalized = normalized[:-2].strip()
    try:
        number = float(normalized)
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _source_viewbox(root: Any, fallback_size: int) -> ViewBox:
    raw = root.attrib.get("viewBox", "")
    pieces = raw.replace(",", " ").split()
    if len(pieces) == 4:
        try:
            x, y, width, height = (float(piece) for piece in pieces)
        except ValueError:
            pass
        else:
            if all(math.isfinite(value) for value in (x, y, width, height)) and (
                width > 0 and height > 0
            ):
                return (x, y, width, height)
    width = _numeric_length(root.attrib.get("width"))
    height = _numeric_length(root.attrib.get("height"))
    if width is not None and height is not None:
        return (0.0, 0.0, width, height)
    return (0.0, 0.0, float(fallback_size), float(fallback_size))


def _viewbox_pixel_transform(
    viewbox: ViewBox,
    viewport_width: int,
    viewport_height: int,
    preserve_aspect_ratio: str,
) -> tuple[float, float, float, float]:
    """Map SVG user units into the forced raster viewport."""
    _, _, viewbox_width, viewbox_height = viewbox
    pieces = preserve_aspect_ratio.strip().split()
    if pieces and pieces[0] == "defer":
        pieces = pieces[1:]
    alignment = pieces[0] if pieces else "xMidYMid"
    mode = pieces[1] if len(pieces) > 1 else "meet"
    scale_x = viewport_width / viewbox_width
    scale_y = viewport_height / viewbox_height
    if alignment == "none":
        return scale_x, scale_y, 0.0, 0.0

    scale = max(scale_x, scale_y) if mode == "slice" else min(scale_x, scale_y)
    remaining_x = viewport_width - viewbox_width * scale
    remaining_y = viewport_height - viewbox_height * scale
    offset_x = (
        0.0
        if alignment.startswith("xMin")
        else remaining_x
        if alignment.startswith("xMax")
        else remaining_x / 2.0
    )
    offset_y = (
        0.0
        if "YMin" in alignment
        else remaining_y
        if "YMax" in alignment
        else remaining_y / 2.0
    )
    return scale, scale, offset_x, offset_y


def _pixel_box_to_viewbox(
    box: PixelBox,
    viewbox: ViewBox,
    viewport_size: int,
    preserve_aspect_ratio: str,
) -> ViewBox:
    vb_x, vb_y, vb_width, vb_height = viewbox
    scale_x, scale_y, offset_x, offset_y = _viewbox_pixel_transform(
        viewbox,
        viewport_size,
        viewport_size,
        preserve_aspect_ratio,
    )
    x0, y0, x1, y1 = box
    user_x0 = vb_x + (x0 - offset_x) / scale_x
    user_y0 = vb_y + (y0 - offset_y) / scale_y
    user_x1 = vb_x + (x1 - offset_x) / scale_x
    user_y1 = vb_y + (y1 - offset_y) / scale_y
    user_x0 = min(max(user_x0, vb_x), vb_x + vb_width)
    user_y0 = min(max(user_y0, vb_y), vb_y + vb_height)
    user_x1 = min(max(user_x1, vb_x), vb_x + vb_width)
    user_y1 = min(max(user_y1, vb_y), vb_y + vb_height)
    if user_x1 <= user_x0 or user_y1 <= user_y0:
        return viewbox
    return (user_x0, user_y0, user_x1 - user_x0, user_y1 - user_y0)


def _format_number(value: float) -> str:
    normalized = 0.0 if abs(value) < 1e-12 else value
    return f"{normalized:.12g}"


def _with_viewbox(svg: str, viewbox: ViewBox) -> str:
    root = parse_svg(svg)
    root.attrib["viewBox"] = " ".join(_format_number(value) for value in viewbox)
    return serialize_svg(root)


def _checked_render(
    svg: str,
    *,
    size: int,
    background: str | None,
):
    image = _decode_png(render_svg_png(svg, size=size, background=background))
    if image.size != (size, size):
        raise RendererUnavailable(
            f"renderer returned {image.width}x{image.height}; expected {size}x{size}"
        )
    return image


def _binary_mask(alpha):
    """Return an opaque, high-contrast mask without discarding antialiasing."""
    Image, *_ = _pillow()
    opaque = Image.new("L", alpha.size, 255)
    return Image.merge("RGBA", (alpha, alpha, alpha, opaque))


def _style_declarations(value: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for declaration in value.split(";"):
        if ":" not in declaration:
            continue
        name, item = declaration.split(":", 1)
        if name.strip():
            declarations[name.strip().lower()] = item.strip()
    return declarations


def _local_attributes(element: Any) -> dict[str, str]:
    values = {
        local_name(name).lower(): value for name, value in element.attrib.items()
    }
    values.update(_style_declarations(element.attrib.get("style", "")))
    return values


def _opacity_number(value: str) -> float | None:
    normalized = value.lower().replace("!important", "").strip()
    try:
        result = (
            float(normalized[:-1]) / 100.0
            if normalized.endswith("%")
            else float(normalized)
        )
    except ValueError:
        return None
    return min(max(result, 0.0), 1.0) if math.isfinite(result) else None


def _candidate_metadata(
    indexed,
    node_id: str,
    bbox_viewbox: ViewBox | None,
    crop_viewbox: ViewBox,
    source_viewbox: ViewBox,
    painted_area_fraction: float,
) -> dict[str, Any]:
    by_id = {node.node_id: node for node in indexed}
    node = by_id[node_id]
    lineage = []
    current = node
    while current is not None:
        lineage.append(current)
        current = by_id.get(current.parent_id or "")
    lineage.reverse()

    resolved = {
        "fill": "black",
        "stroke": "none",
        "color": "black",
        "fill-opacity": "1",
        "stroke-opacity": "1",
    }
    opacity_product = 1.0
    for ancestor in lineage:
        attributes = _local_attributes(ancestor.element)
        for name in tuple(resolved):
            if name in attributes:
                resolved[name] = attributes[name]
        opacity = _opacity_number(attributes.get("opacity", "1"))
        if opacity is not None:
            opacity_product *= opacity

    vb_x, vb_y, vb_width, vb_height = source_viewbox
    normalized_bbox: list[float] | None = None
    center: list[float] | None = None
    if bbox_viewbox is not None:
        box_x, box_y, box_width, box_height = bbox_viewbox
        normalized_bbox = [
            round((box_x - vb_x) / vb_width, 6),
            round((box_y - vb_y) / vb_height, 6),
            round(box_width / vb_width, 6),
            round(box_height / vb_height, 6),
        ]
        center = [
            round((box_x + box_width / 2.0 - vb_x) / vb_width, 6),
            round((box_y + box_height / 2.0 - vb_y) / vb_height, 6),
        ]

    return {
        "tag": local_name(node.element.tag),
        "depth": node.depth,
        "child_count": len(list(node.element)),
        "descendant_count": len(_descendant_ids(indexed, node_id)) - 1,
        "visible": bbox_viewbox is not None,
        "bbox_normalized": normalized_bbox,
        "center_normalized": center,
        "painted_area_fraction": round(painted_area_fraction, 6),
        "crop_viewbox": [round(value, 6) for value in crop_viewbox],
        "fill": resolved["fill"],
        "stroke": resolved["stroke"],
        "color": resolved["color"],
        "fill_opacity": resolved["fill-opacity"],
        "stroke_opacity": resolved["stroke-opacity"],
        "effective_opacity": round(opacity_product, 6),
    }


def _highlight_context(context, isolated, highlight_color: str):
    Image, ImageChops, _, ImageEnhance, ImageFilter, _ = _pillow()
    if context.size != isolated.size:
        raise RendererUnavailable("context and isolated renders have different sizes")

    color = Image.new("RGBA", context.size, highlight_color)
    white = Image.new("RGBA", context.size, "white")
    opaque_context = Image.alpha_composite(white, context)
    highlighted = Image.blend(white, opaque_context, 0.48)

    alpha = isolated.getchannel("A")
    if alpha.getbbox() is None:
        return highlighted

    # Restore the selected subtree at full strength, then add a translucent
    # tint and a high-contrast outer halo. Rendering the subtree separately
    # also makes a fully occluded target locatable without deleting context.
    highlighted.alpha_composite(isolated)
    tint_alpha = ImageEnhance.Brightness(alpha).enhance(0.34)
    tint = color.copy()
    tint.putalpha(tint_alpha)
    highlighted.alpha_composite(tint)

    radius = max(3, (min(context.size) // 64) * 2 + 1)
    if radius % 2 == 0:
        radius += 1
    dilated = alpha.filter(ImageFilter.MaxFilter(radius))
    halo_alpha = ImageChops.subtract(dilated, alpha)
    halo_alpha = ImageEnhance.Brightness(halo_alpha).enhance(1.8)
    halo = color.copy()
    halo.putalpha(halo_alpha)
    highlighted.alpha_composite(halo)
    return highlighted


def _render_views_from_validated(
    svg: str,
    candidate_ids: tuple[str, ...],
    *,
    size: int,
    crop_padding: float,
    highlight_color: str,
) -> tuple[CandidateView, ...]:
    context = _decode_png(render_svg_png(svg, size=size, background="white"))
    if context.size != (size, size):
        raise RendererUnavailable(
            f"renderer returned {context.width}x{context.height}; expected {size}x{size}"
        )

    views: list[CandidateView] = []
    for node_id in candidate_ids:
        isolated_svg = isolate_svg_subtree(svg, node_id)
        isolated_full = _decode_png(
            render_svg_png(isolated_svg, size=size, background=None)
        )
        if isolated_full.size != context.size:
            raise RendererUnavailable(
                "context and isolated renders have different dimensions"
            )
        bbox = isolated_full.getchannel("A").getbbox()
        crop_box = _square_crop_box(bbox, context.width, context.height, crop_padding)
        highlighted = _highlight_context(context, isolated_full, highlight_color)
        local_crop = _fit_square(highlighted, crop_box, size)
        isolated = _fit_square(isolated_full, crop_box, size)
        views.append(
            CandidateView(
                node_id=node_id,
                full_context_png=_encode_png(highlighted),
                local_crop_png=_encode_png(local_crop),
                isolated_png=_encode_png(isolated),
                bbox=bbox,
                crop_box=crop_box,
                size=size,
            )
        )
    return tuple(views)


def render_candidate_views(
    svg: str,
    candidate_ids: Sequence[str],
    *,
    size: int = 192,
    crop_padding: float = 0.2,
    highlight_color: str = "#ff2d55",
) -> tuple[CandidateView, ...]:
    """Render ordered context/crop/isolated evidence for several candidates.

    The full SVG is rasterized once and reused. Each candidate adds one
    transparent subtree render. Unknown or duplicate IDs fail before any
    renderer call, preventing label/order drift at the model boundary.
    """
    _validate_size(size)
    _validate_padding(crop_padding)
    _, _, ordered = _validated_candidates(svg, candidate_ids)
    return _render_views_from_validated(
        svg,
        ordered,
        size=size,
        crop_padding=crop_padding,
        highlight_color=highlight_color,
    )


def render_candidate_view(
    svg: str,
    node_id: str,
    *,
    size: int = 192,
    crop_padding: float = 0.2,
    highlight_color: str = "#ff2d55",
) -> CandidateView:
    """Render the three evidence views for a single candidate node/subtree."""
    return render_candidate_views(
        svg,
        (node_id,),
        size=size,
        crop_padding=crop_padding,
        highlight_color=highlight_color,
    )[0]


def render_candidate_evidence(
    svg: str,
    candidate_ids: Sequence[str],
    *,
    size: int = 384,
    crop_padding: float = 0.25,
    highlight_color: str = "#ff2d55",
) -> tuple[CandidateEvidence, ...]:
    """Render high-fidelity multi-image evidence in candidate order.

    Each candidate receives four independent square images: a highlighted full
    scene, a highlighted local crop, the isolated local crop, and a binary
    ownership mask.  Local images are fresh SVG rasterizations after changing
    the root viewBox; this function never enlarges a crop from the full-scene
    raster.

    The full scene is shared.  Each candidate then costs three rasterizations:
    one full-canvas isolation used to locate it, plus direct context and
    isolated renders of its padded vector crop.
    """
    _validate_size(size)
    _validate_padding(crop_padding)
    root, indexed, ordered = _validated_candidates(svg, candidate_ids)
    source_viewbox = _source_viewbox(root, size)
    preserve_aspect_ratio = root.attrib.get(
        "preserveAspectRatio",
        "xMidYMid meet",
    )
    context = _checked_render(svg, size=size, background="white")

    evidence: list[CandidateEvidence] = []
    for node_id in ordered:
        isolated_svg = isolate_svg_subtree(svg, node_id)
        isolated_full = _checked_render(
            isolated_svg,
            size=size,
            background=None,
        )
        alpha = isolated_full.getchannel("A")
        bbox = alpha.getbbox()
        crop_box = _square_crop_box(
            bbox,
            context.width,
            context.height,
            crop_padding,
        )
        bbox_viewbox = (
            _pixel_box_to_viewbox(
                bbox,
                source_viewbox,
                size,
                preserve_aspect_ratio,
            )
            if bbox is not None
            else None
        )
        crop_viewbox = _pixel_box_to_viewbox(
            crop_box,
            source_viewbox,
            size,
            preserve_aspect_ratio,
        )

        cropped_context = _checked_render(
            _with_viewbox(svg, crop_viewbox),
            size=size,
            background="white",
        )
        cropped_isolated = _checked_render(
            _with_viewbox(isolated_svg, crop_viewbox),
            size=size,
            background=None,
        )
        highlighted_context = _highlight_context(
            context,
            isolated_full,
            highlight_color,
        )
        highlighted_crop = _highlight_context(
            cropped_context,
            cropped_isolated,
            highlight_color,
        )
        cropped_alpha = cropped_isolated.getchannel("A")
        painted_pixels = sum(alpha.histogram()[1:])
        painted_area_fraction = painted_pixels / float(size * size)
        metadata = _candidate_metadata(
            indexed,
            node_id,
            bbox_viewbox,
            crop_viewbox,
            source_viewbox,
            painted_area_fraction,
        )
        evidence.append(
            CandidateEvidence(
                node_id=node_id,
                full_context_png=_encode_png(highlighted_context),
                vector_crop_png=_encode_png(highlighted_crop),
                isolated_crop_png=_encode_png(cropped_isolated),
                mask_png=_encode_png(_binary_mask(cropped_alpha)),
                bbox=bbox,
                bbox_viewbox=bbox_viewbox,
                crop_viewbox=crop_viewbox,
                size=size,
                metadata=metadata,
                appearance_digest=hashlib.sha256(
                    isolated_full.tobytes()
                ).hexdigest(),
                mask_digest=hashlib.sha256(alpha.tobytes()).hexdigest(),
            )
        )
    return tuple(evidence)


def render_single_candidate_evidence(
    svg: str,
    node_id: str,
    *,
    size: int = 384,
    crop_padding: float = 0.25,
    highlight_color: str = "#ff2d55",
) -> CandidateEvidence:
    """Render one candidate using :func:`render_candidate_evidence`."""
    return render_candidate_evidence(
        svg,
        (node_id,),
        size=size,
        crop_padding=crop_padding,
        highlight_color=highlight_color,
    )[0]


def _alphabetic_label(index: int) -> str:
    value = index + 1
    pieces: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        pieces.append(chr(ord("A") + remainder))
    return "".join(reversed(pieces))


def _validated_labels(
    candidate_ids: tuple[str, ...], labels: Mapping[str, str] | None
) -> dict[str, str]:
    if labels is None:
        return {
            node_id: _alphabetic_label(index)
            for index, node_id in enumerate(candidate_ids)
        }
    missing = tuple(node_id for node_id in candidate_ids if node_id not in labels)
    if missing:
        raise ValueError(f"labels missing candidate node IDs: {', '.join(missing)}")
    result = {node_id: labels[node_id] for node_id in candidate_ids}
    if any(not isinstance(label, str) or not label.strip() for label in result.values()):
        raise ValueError("candidate labels must be non-empty strings")
    if len(set(result.values())) != len(result):
        raise ValueError("candidate labels must be unique")
    return result


def _font(ImageFont, size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:  # pragma: no cover - depends on Pillow packaging
        return ImageFont.load_default()


def _on_checkerboard(image):
    Image, *_ = _pillow()
    tile = max(4, min(image.size) // 16)
    background = Image.new("RGBA", image.size, "#ffffff")
    pixels = background.load()
    for y in range(image.height):
        for x in range(image.width):
            if (x // tile + y // tile) % 2:
                pixels[x, y] = (232, 232, 232, 255)
    background.alpha_composite(image)
    return background


def _contact_sheet(
    views: tuple[CandidateView, ...],
    labels: dict[str, str],
    columns: int,
    show_node_ids: bool,
):
    Image, _, ImageDraw, _, _, ImageFont = _pillow()
    panel_size = views[0].size
    gap = max(4, panel_size // 24)
    outer = max(8, panel_size // 12)
    header_height = max(28, panel_size // 6)
    footer_height = max(18, panel_size // 10)
    card_width = panel_size * 3 + gap * 2 + outer * 2
    card_height = header_height + panel_size + footer_height + outer
    rows = math.ceil(len(views) / columns)
    sheet = Image.new(
        "RGBA",
        (columns * card_width + (columns + 1) * gap,
         rows * card_height + (rows + 1) * gap),
        "#e8ebef",
    )
    draw = ImageDraw.Draw(sheet)
    label_font = _font(ImageFont, max(18, panel_size // 8), bold=True)
    id_font = _font(ImageFont, max(11, panel_size // 15))
    caption_font = _font(ImageFont, max(10, panel_size // 16), bold=True)
    captions = ("CONTEXT", "CROP", "ISOLATED")

    for index, view in enumerate(views):
        row, column = divmod(index, columns)
        card_x = gap + column * card_width
        card_y = gap + row * card_height
        draw.rounded_rectangle(
            (card_x, card_y, card_x + card_width - 1, card_y + card_height - 1),
            radius=max(6, panel_size // 24),
            fill="#ffffff",
            outline="#c3c9d1",
            width=max(1, panel_size // 96),
        )
        label = labels[view.node_id]
        draw.text(
            (card_x + outer, card_y + max(2, header_height // 8)),
            label,
            fill="#111827",
            font=label_font,
        )
        if show_node_ids:
            label_width = draw.textbbox((0, 0), label, font=label_font)[2]
            draw.text(
                (card_x + outer + label_width + gap, card_y + header_height // 4),
                view.node_id,
                fill="#667085",
                font=id_font,
            )

        images = (
            _decode_png(view.full_context_png),
            _decode_png(view.local_crop_png),
            _on_checkerboard(_decode_png(view.isolated_png)),
        )
        image_y = card_y + header_height
        for panel_index, (panel, caption) in enumerate(zip(images, captions)):
            image_x = card_x + outer + panel_index * (panel_size + gap)
            sheet.alpha_composite(panel, (image_x, image_y))
            draw.rectangle(
                (image_x, image_y, image_x + panel_size - 1, image_y + panel_size - 1),
                outline="#aeb6c1",
                width=max(1, panel_size // 96),
            )
            draw.text(
                (image_x, image_y + panel_size + max(2, footer_height // 8)),
                caption,
                fill="#475467",
                font=caption_font,
            )
        if not view.visible:
            warning = "NO PAINTED PIXELS"
            warning_x = card_x + outer + 2 * (panel_size + gap)
            warning_y = image_y + panel_size // 2
            draw.rectangle(
                (warning_x + 4, warning_y - 10,
                 warning_x + panel_size - 4, warning_y + 10),
                fill="#ffffff",
            )
            draw.text(
                (warning_x + 8, warning_y - 8),
                warning,
                fill="#b42318",
                font=caption_font,
            )
    return sheet


def _evidence_metadata_caption(item: CandidateEvidence) -> str:
    metadata = item.metadata
    center = metadata.get("center_normalized")
    bbox = metadata.get("bbox_normalized")
    center_text = (
        f"({float(center[0]):.2f},{float(center[1]):.2f})"
        if isinstance(center, list) and len(center) == 2
        else "unknown"
    )
    size_text = (
        f"({float(bbox[2]):.2f},{float(bbox[3]):.2f})"
        if isinstance(bbox, list) and len(bbox) == 4
        else "unknown"
    )
    return (
        f"tag={metadata.get('tag', 'unknown')}  center={center_text}  "
        f"size={size_text}  depth={metadata.get('depth', 'unknown')}  "
        f"children={metadata.get('child_count', 'unknown')}"
    )


def _evidence_sheet(
    evidence: tuple[CandidateEvidence, ...],
    labels: dict[str, str],
    columns: int,
):
    Image, _, ImageDraw, _, _, ImageFont = _pillow()
    panel_size = evidence[0].size
    if any(item.size != panel_size for item in evidence):
        raise ValueError("candidate evidence images must have one common size")
    gap = max(5, panel_size // 24)
    outer = max(10, panel_size // 12)
    header_height = max(34, panel_size // 6)
    footer_height = max(20, panel_size // 10)
    metadata_height = max(24, panel_size // 9)
    card_width = panel_size * 3 + gap * 2 + outer * 2
    card_height = (
        header_height + panel_size + footer_height + metadata_height + outer
    )
    rows = math.ceil(len(evidence) / columns)
    sheet = Image.new(
        "RGBA",
        (
            columns * card_width + (columns + 1) * gap,
            rows * card_height + (rows + 1) * gap,
        ),
        "#e8ebef",
    )
    draw = ImageDraw.Draw(sheet)
    label_font = _font(ImageFont, max(22, panel_size // 7), bold=True)
    caption_font = _font(ImageFont, max(11, panel_size // 17), bold=True)
    metadata_font = _font(ImageFont, max(10, panel_size // 19))
    captions = ("FULL CONTEXT", "DIRECT VECTOR CROP", "OWNERSHIP MASK")

    for index, item in enumerate(evidence):
        row, column = divmod(index, columns)
        card_x = gap + column * card_width
        card_y = gap + row * card_height
        draw.rounded_rectangle(
            (card_x, card_y, card_x + card_width - 1, card_y + card_height - 1),
            radius=max(7, panel_size // 24),
            fill="#ffffff",
            outline="#c3c9d1",
            width=max(1, panel_size // 96),
        )
        draw.text(
            (card_x + outer, card_y + max(2, header_height // 10)),
            labels[item.node_id],
            fill="#111827",
            font=label_font,
        )
        image_y = card_y + header_height
        panels = (
            _decode_png(item.full_context_png),
            _decode_png(item.vector_crop_png),
            _decode_png(item.mask_png),
        )
        for panel_index, (panel, caption) in enumerate(zip(panels, captions)):
            image_x = card_x + outer + panel_index * (panel_size + gap)
            sheet.alpha_composite(panel, (image_x, image_y))
            draw.rectangle(
                (image_x, image_y, image_x + panel_size - 1, image_y + panel_size - 1),
                outline="#8b95a5",
                width=max(1, panel_size // 96),
            )
            draw.text(
                (image_x, image_y + panel_size + max(2, footer_height // 8)),
                caption,
                fill="#344054",
                font=caption_font,
            )
        draw.text(
            (
                card_x + outer,
                image_y + panel_size + footer_height + max(1, metadata_height // 8),
            ),
            _evidence_metadata_caption(item),
            fill="#475467",
            font=metadata_font,
        )
    return sheet


def render_candidate_evidence_sheet(
    svg: str,
    candidate_ids: Sequence[str],
    *,
    size: int = 256,
    columns: int | None = None,
    labels: Mapping[str, str] | None = None,
    show_node_ids: bool = False,
    crop_padding: float = 0.25,
    highlight_color: str = "#ff2d55",
) -> CandidateEvidenceSheet:
    """Render a labeled high-resolution sheet from direct vector evidence."""
    _validate_size(size)
    _validate_padding(crop_padding)
    if show_node_ids:
        raise ValueError("high-fidelity candidate evidence never exposes node IDs")
    _, _, ordered = _validated_candidates(svg, candidate_ids)
    resolved_labels = _validated_labels(ordered, labels)
    if columns is None:
        columns = max(1, math.ceil(math.sqrt(len(ordered) / 3.0)))
    if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
        raise ValueError("columns must be a positive integer")
    columns = min(columns, len(ordered))
    evidence = render_candidate_evidence(
        svg,
        ordered,
        size=size,
        crop_padding=crop_padding,
        highlight_color=highlight_color,
    )
    sheet = _evidence_sheet(evidence, resolved_labels, columns)
    return CandidateEvidenceSheet(
        png=_encode_png(sheet),
        candidate_ids=ordered,
        labels=resolved_labels,
        evidence=evidence,
        columns=columns,
    )


def render_candidate_contact_sheet(
    svg: str,
    candidate_ids: Sequence[str],
    *,
    size: int = 192,
    columns: int | None = None,
    labels: Mapping[str, str] | None = None,
    show_node_ids: bool = True,
    crop_padding: float = 0.2,
    highlight_color: str = "#ff2d55",
) -> CandidateContactSheet:
    """Render one labeled context/crop/isolated card per candidate.

    Candidate order is preserved exactly. ``labels`` maps node ID to the
    caller-owned visible choice label (for example ``{"n3": "A"}``). If it
    is omitted, labels are assigned A, B, ... in supplied order.
    """
    _validate_size(size)
    _validate_padding(crop_padding)
    if not isinstance(show_node_ids, bool):
        raise ValueError("show_node_ids must be a boolean")
    _, _, ordered = _validated_candidates(svg, candidate_ids)
    resolved_labels = _validated_labels(ordered, labels)
    if columns is None:
        # Each card is roughly 3:1, so fewer columns produces a compact sheet.
        columns = max(1, math.ceil(math.sqrt(len(ordered) / 3.0)))
    if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
        raise ValueError("columns must be a positive integer")
    columns = min(columns, len(ordered))

    views = _render_views_from_validated(
        svg,
        ordered,
        size=size,
        crop_padding=crop_padding,
        highlight_color=highlight_color,
    )
    sheet = _contact_sheet(views, resolved_labels, columns, show_node_ids)
    return CandidateContactSheet(
        png=_encode_png(sheet),
        candidate_ids=ordered,
        labels=resolved_labels,
        views=views,
        columns=columns,
        show_node_ids=show_node_ids,
    )


__all__ = [
    "CandidateContactSheet",
    "CandidateEvidence",
    "CandidateEvidenceSheet",
    "CandidateView",
    "isolate_svg_subtree",
    "render_candidate_contact_sheet",
    "render_candidate_evidence",
    "render_candidate_evidence_sheet",
    "render_candidate_view",
    "render_candidate_views",
    "render_single_candidate_evidence",
]
