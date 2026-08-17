"""Per-node geometry computed from path data alone, without rendering.

This is the ablation baseline for render-derived visual stats. It reports the
same four fields the hide-and-diff measurement reports -- bbox, area_pct,
position, color -- but derives them by parsing the SVG's own geometry rather
than by rasterizing anything.

The difference between the two is the entire argument for rendering. Analytic
geometry describes a node's *nominal* extent: where the shape would be if
nothing covered it, and what fill it declares. Render-derived stats describe a
node's *visible contribution*: the pixels that actually change when it is
hidden. For a simple unoccluded fill the two should be close, but stroke,
anti-aliasing, clipping, filters, and curve flattening can still separate the
measurements. For a node lying under something else they can diverge sharply,
and only the rendered measurement observes that composited contribution.

Accordingly this module never reports ``visible``. It cannot: occlusion is not
knowable from the source alone, which is precisely what the ablation tests.
"""
from __future__ import annotations

from typing import Any

from svgpatchlab.eval.render import _position_word

from .xml import index_tree, local_name, parse_svg

#: Segments per curve when flattening for the area estimate. Emoji artwork is
#: small and smooth; 24 keeps the shoelace area within a fraction of a percent
#: of the true filled area while staying cheap.
_FLATTEN_STEPS = 24

_SHAPE_TAGS = {"path", "rect", "circle", "ellipse", "line", "polygon", "polyline"}


def _viewbox(root) -> tuple[float, float, float, float]:
    raw = (root.attrib.get("viewBox") or "").replace(",", " ").split()
    if len(raw) == 4:
        try:
            x, y, width, height = (float(value) for value in raw)
            if width > 0 and height > 0:
                return x, y, width, height
        except ValueError:
            pass
    return 0.0, 0.0, 100.0, 100.0


def _shape_for(element, matrix):
    """Build an svgelements shape for one element, with transforms applied."""
    from svgelements import (
        Circle,
        Ellipse,
        Line,
        Path,
        Polygon,
        Polyline,
        Rect,
    )

    tag = local_name(element.tag)
    attributes = element.attrib

    def number(name: str, default: float = 0.0) -> float:
        try:
            return float(attributes.get(name, default))
        except (TypeError, ValueError):
            return default

    try:
        if tag == "path":
            if not attributes.get("d"):
                return None
            shape = Path(attributes["d"])
        elif tag == "rect":
            if number("width") <= 0 or number("height") <= 0:
                return None
            shape = Rect(
                x=number("x"), y=number("y"),
                width=number("width"), height=number("height"),
                rx=number("rx"), ry=number("ry"),
            )
        elif tag == "circle":
            if number("r") <= 0:
                return None
            shape = Circle(cx=number("cx"), cy=number("cy"), r=number("r"))
        elif tag == "ellipse":
            if number("rx") <= 0 or number("ry") <= 0:
                return None
            shape = Ellipse(
                cx=number("cx"), cy=number("cy"), rx=number("rx"), ry=number("ry")
            )
        elif tag == "line":
            shape = Line(
                number("x1"), number("y1"), number("x2"), number("y2")
            )
        elif tag in ("polygon", "polyline"):
            points = (attributes.get("points") or "").strip()
            if not points:
                return None
            shape = (Polygon if tag == "polygon" else Polyline)(points)
        else:
            return None
    except Exception:
        # Malformed geometry is treated as absent rather than fatal; the
        # rendered path has the same tolerance.
        return None

    try:
        path = Path(shape)
        if matrix is not None:
            path *= matrix
        return path
    except Exception:
        return None


def _subpath_points(path) -> list[list[tuple[float, float]]]:
    """Flatten each path subpath independently.

    A single SVG path may contain several contours. Joining those contours
    into one point list creates artificial edges between them and makes both
    holes and disjoint shapes report the wrong area.
    """
    try:
        subpaths = list(path.as_subpaths())
    except (AttributeError, TypeError):
        subpaths = [path]

    result: list[list[tuple[float, float]]] = []
    for subpath in subpaths:
        points: list[tuple[float, float]] = []
        for segment in subpath.segments():
            try:
                start = segment.start
                end = segment.end
            except AttributeError:
                continue
            if type(segment).__name__ == "Move":
                if end is not None:
                    points.append((float(end.x), float(end.y)))
                continue
            if end is None:
                continue
            if not points and start is not None:
                points.append((float(start.x), float(start.y)))
            try:
                for step in range(1, _FLATTEN_STEPS + 1):
                    point = segment.point(step / _FLATTEN_STEPS)
                    points.append((float(point.x), float(point.y)))
            except Exception:
                points.append((float(end.x), float(end.y)))
        if len(points) >= 3:
            result.append(points)
    return result


def _polygon_geometry(
    points: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """Return signed area and centroid for one flattened contour."""
    if len(points) < 3:
        return 0.0, 0.0, 0.0
    cross_sum = 0.0
    centroid_x_sum = 0.0
    centroid_y_sum = 0.0
    for index in range(len(points)):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % len(points)]
        cross = x0 * y1 - x1 * y0
        cross_sum += cross
        centroid_x_sum += (x0 + x1) * cross
        centroid_y_sum += (y0 + y1) * cross
    if abs(cross_sum) <= 1e-12:
        return 0.0, 0.0, 0.0
    signed_area = cross_sum / 2.0
    return (
        signed_area,
        centroid_x_sum / (3.0 * cross_sum),
        centroid_y_sum / (3.0 * cross_sum),
    )


def _point_in_polygon(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    """Return whether a point is inside a flattened contour."""
    px, py = point
    inside = False
    for index in range(len(polygon)):
        x0, y0 = polygon[index]
        x1, y1 = polygon[(index + 1) % len(polygon)]
        if (y0 > py) == (y1 > py):
            continue
        crossing_x = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
        if px < crossing_x:
            inside = not inside
    return inside


def _path_fill_geometry(path, fill_rule: str) -> tuple[float, float | None, float | None]:
    """Return filled area and centroid for non-intersecting path contours.

    SVG's nonzero and evenodd rules differ for nested contours. Tracking each
    contour's immediate container and winding state handles ordinary compound
    emoji paths (holes, islands, and disjoint components) without introducing
    false connecting edges. Self-intersecting or mutually intersecting
    contours remain an approximation, as does curve flattening itself.
    """
    contours: list[dict[str, Any]] = []
    for points in _subpath_points(path):
        signed_area, centroid_x, centroid_y = _polygon_geometry(points)
        if abs(signed_area) <= 1e-12:
            continue
        contours.append(
            {
                "points": points,
                "signed_area": signed_area,
                "area": abs(signed_area),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
            }
        )
    if not contours:
        return 0.0, None, None

    parents: list[int | None] = []
    for index, contour in enumerate(contours):
        probe = contour["points"][0]
        containers = [
            candidate_index
            for candidate_index, candidate in enumerate(contours)
            if candidate_index != index
            and candidate["area"] > contour["area"] + 1e-12
            and _point_in_polygon(probe, candidate["points"])
        ]
        parents.append(
            min(containers, key=lambda item: contours[item]["area"])
            if containers
            else None
        )

    contributions: list[float] = [0.0] * len(contours)
    if fill_rule.strip().lower() == "evenodd":
        depths: dict[int, int] = {}

        def depth(index: int) -> int:
            if index not in depths:
                parent = parents[index]
                depths[index] = 0 if parent is None else depth(parent) + 1
            return depths[index]

        for index, contour in enumerate(contours):
            contributions[index] = contour["area"] * (
                1.0 if depth(index) % 2 == 0 else -1.0
            )
    else:
        windings: dict[int, int] = {}

        def winding(index: int) -> int:
            if index not in windings:
                parent = parents[index]
                outside = 0 if parent is None else winding(parent)
                direction = 1 if contours[index]["signed_area"] > 0.0 else -1
                windings[index] = outside + direction
            return windings[index]

        for index, contour in enumerate(contours):
            parent = parents[index]
            outside = 0 if parent is None else winding(parent)
            inside = winding(index)
            contributions[index] = contour["area"] * (
                float(inside != 0) - float(outside != 0)
            )

    area = sum(contributions)
    if area <= 1e-12:
        return 0.0, None, None
    centroid_x = sum(
        contribution * contour["centroid_x"]
        for contribution, contour in zip(contributions, contours)
    ) / area
    centroid_y = sum(
        contribution * contour["centroid_y"]
        for contribution, contour in zip(contributions, contours)
    ) / area
    return area, centroid_x, centroid_y


def _style_property(element, name: str) -> str | None:
    if name in element.attrib:
        return element.attrib[name]
    for declaration in element.attrib.get("style", "").split(";"):
        if ":" not in declaration:
            continue
        property_name, value = declaration.split(":", 1)
        if property_name.strip() == name:
            return value.strip()
    return None


def _polygon_points(path) -> list[tuple[float, float]]:
    """Flatten a single-contour path for backward compatibility."""
    subpaths = _subpath_points(path)
    return subpaths[0] if subpaths else []


def _shoelace_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index in range(len(points)):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _matrix_for(element, parent_matrix):
    from svgelements import Matrix

    transform = element.attrib.get("transform")
    if not transform:
        return parent_matrix
    try:
        local = Matrix(transform)
    except Exception:
        return parent_matrix
    return local if parent_matrix is None else local * parent_matrix


def node_analytic_stats(svg: str) -> dict[str, dict[str, Any]]:
    """Return per-node analytic geometry keyed by node id.

    Mirrors :func:`svgpatchlab.eval.render.node_visual_stats` in shape so the
    two can be swapped behind one architecture, but omits ``visible`` because
    occlusion cannot be determined without rasterizing.
    """
    root = parse_svg(svg)
    nodes = index_tree(root)
    vb_x, vb_y, vb_width, vb_height = _viewbox(root)
    canvas_area = vb_width * vb_height

    by_id = {node.node_id: node for node in nodes}
    matrices: dict[str, Any] = {}
    fill_rules: dict[str, str] = {}
    fills: dict[str, str | None] = {}
    for node in nodes:
        parent = matrices.get(node.parent_id) if node.parent_id else None
        matrices[node.node_id] = _matrix_for(node.element, parent)
        inherited_fill_rule = (
            fill_rules.get(node.parent_id, "nonzero")
            if node.parent_id
            else "nonzero"
        )
        fill_rules[node.node_id] = (
            _style_property(node.element, "fill-rule") or inherited_fill_rule
        )
        inherited_fill = fills.get(node.parent_id) if node.parent_id else None
        direct_fill = _style_property(node.element, "fill")
        fills[node.node_id] = (
            direct_fill if direct_fill is not None else inherited_fill
        )

    # Own geometry per node, then unioned upward so a group reports the extent
    # of its descendants, matching how hiding a group behaves.
    own: dict[
        str,
        tuple[
            tuple[float, float, float, float],
            float,
            float | None,
            float | None,
        ],
    ] = {}
    for node in nodes:
        if local_name(node.element.tag) not in _SHAPE_TAGS:
            continue
        path = _shape_for(node.element, matrices[node.node_id])
        if path is None:
            continue
        try:
            box = path.bbox()
        except Exception:
            box = None
        if not box or any(value is None for value in box):
            continue
        area, centroid_x, centroid_y = _path_fill_geometry(
            path, fill_rules[node.node_id]
        )
        own[node.node_id] = (
            (box[0], box[1], box[2], box[3]),
            area,
            centroid_x,
            centroid_y,
        )

    descendants: dict[str, list[str]] = {node.node_id: [] for node in nodes}
    for node in nodes:
        current = node.node_id
        while current is not None:
            descendants[current].append(node.node_id)
            current = by_id[current].parent_id

    stats: dict[str, dict[str, Any]] = {}
    for node in nodes:
        boxes = [own[child][0] for child in descendants[node.node_id] if child in own]
        if not boxes:
            continue
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        area = sum(own[child][1] for child in descendants[node.node_id] if child in own)

        weighted = [
            own[child]
            for child in descendants[node.node_id]
            if child in own
            and own[child][1] > 0.0
            and own[child][2] is not None
            and own[child][3] is not None
        ]
        if area > 0.0 and weighted:
            centre_x = sum(item[1] * item[2] for item in weighted) / area
            centre_y = sum(item[1] * item[3] for item in weighted) / area
        else:
            centre_x = (x0 + x1) / 2.0
            centre_y = (y0 + y1) / 2.0
        entry: dict[str, Any] = {
            "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
            "area_pct": round(100.0 * min(area, canvas_area) / canvas_area, 2)
            if canvas_area
            else 0.0,
            "position": _position_word(
                (centre_x - vb_x) / vb_width if vb_width else 0.5,
                (centre_y - vb_y) / vb_height if vb_height else 0.5,
            ),
        }
        fill = fills[node.node_id]
        if fill and fill.lower() != "none":
            entry["color"] = fill.lower()
        stats[node.node_id] = entry
    return stats
