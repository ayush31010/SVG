from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any


class RendererUnavailable(RuntimeError):
    pass


def _dependencies():
    # OSError: cairosvg is installed but the native cairo library is missing
    # (the usual state on Windows without a GTK runtime).
    try:
        import cairosvg
        import numpy as np
        from PIL import Image
    except (ImportError, OSError) as exc:
        raise RendererUnavailable(
            "rendering requires the optional dependencies: pip install -e '.[eval]'"
        ) from exc
    return cairosvg, np, Image


def ensure_renderer() -> None:
    _dependencies()


def render_svg_png(
    svg: str,
    size: int = 72,
    background: str | None = "white",
) -> bytes:
    cairosvg, _, _ = _dependencies()
    return cairosvg.svg2png(
        bytestring=svg.encode(),
        output_width=size,
        output_height=size,
        background_color=background,
    )


def render_svg_array(svg: str, size: int = 72, background: str = "white"):
    _, np, Image = _dependencies()
    png = render_svg_png(svg, size=size, background=background)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255.0


def png_data_url(png: bytes) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_svg_data_url(svg: str, size: int = 512) -> str:
    return png_data_url(render_svg_png(svg, size=size, background="white"))


def image_mse(candidate_svg: str, answer_svg: str, size: int = 72) -> float:
    _, np, _ = _dependencies()
    candidate = render_svg_array(candidate_svg, size=size)
    answer = render_svg_array(answer_svg, size=size)
    return float(np.mean((candidate - answer) ** 2))


def render_svg_rgba(svg: str, size: int = 64):
    """Render to a float32 RGBA array in [0, 1] over a transparent background."""
    _, np, Image = _dependencies()
    png = render_svg_png(svg, size=size, background=None)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), dtype=np.float32) / 255.0


def _parse_viewbox(root) -> tuple[float, float, float, float] | None:
    raw = root.attrib.get("viewBox", "")
    parts = raw.replace(",", " ").split()
    if len(parts) == 4:
        try:
            x, y, w, h = (float(p) for p in parts)
        except ValueError:
            return None
        if w > 0 and h > 0:
            return x, y, w, h
    return None


_POSITION_COLUMNS = ("left", "", "right")
_POSITION_ROWS = ("top", "", "bottom")


def _position_word(cx_rel: float, cy_rel: float) -> str:
    row = _POSITION_ROWS[min(int(cy_rel * 3), 2)]
    column = _POSITION_COLUMNS[min(int(cx_rel * 3), 2)]
    if row and column:
        return f"{row}-{column}"
    return row or column or "center"


def _stats_from_diff(
    full,
    without,
    viewbox: tuple[float, float, float, float],
    threshold: float = 0.02,
) -> dict[str, Any]:
    """Reduce a full-vs-node-hidden render pair to compact per-node stats.

    Both inputs are float RGBA arrays from render_svg_rgba. The difference
    mask is the node's *visible* contribution, so fully occluded nodes and
    non-rendering nodes (defs, empty groups) come back as visible: false.
    """
    import numpy as np

    diff = np.abs(full - without).max(axis=2)
    mask = diff > threshold
    if not mask.any():
        return {"visible": False}

    height, width = mask.shape
    rows = np.any(mask, axis=1).nonzero()[0]
    columns = np.any(mask, axis=0).nonzero()[0]
    vb_x, vb_y, vb_w, vb_h = viewbox
    x0 = vb_x + columns[0] / width * vb_w
    x1 = vb_x + (columns[-1] + 1) / width * vb_w
    y0 = vb_y + rows[0] / height * vb_h
    y1 = vb_y + (rows[-1] + 1) / height * vb_h

    cy, cx = np.nonzero(mask)
    centroid_col = (cx.mean() + 0.5) / width
    centroid_row = (cy.mean() + 0.5) / height

    stats: dict[str, Any] = {
        "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
        "area_pct": round(100.0 * mask.sum() / mask.size, 2),
        "position": _position_word(centroid_col, centroid_row),
    }

    opaque = mask & (full[:, :, 3] > 0.1)
    sample = full[opaque if opaque.any() else mask][:, :3]
    red, green, blue = (int(round(float(np.median(sample[:, i])) * 255)) for i in range(3))
    stats["color"] = f"#{red:02x}{green:02x}{blue:02x}"
    return stats


def _hide_element(element) -> str | None:
    """Append display:none to the element's style; return the prior style."""
    original = element.attrib.get("style")
    element.attrib["style"] = (f"{original};" if original else "") + "display:none"
    return original


def _restore_element(element, original_style: str | None) -> None:
    if original_style is None:
        element.attrib.pop("style", None)
    else:
        element.attrib["style"] = original_style


def node_visual_stats(
    svg: str,
    node_ids: list[str] | None = None,
    size: int = 64,
    threshold: float = 0.02,
) -> dict[str, dict[str, Any]]:
    """Compute per-node visual stats by diffing full vs node-hidden renders.

    For each node the SVG is re-rendered with that node display:none'd and the
    difference against the full render gives its visible footprint — occlusion
    aware and faithful to inherited styles, unlike an isolated render. Costs
    len(node_ids) + 1 rasterizations at `size` px.

    Hiding a group hides its subtree, so a group's stats cover its descendants.
    """
    from svgpatchlab.core.xml import index_tree, parse_svg, serialize_svg

    root = parse_svg(svg)
    indexed = index_tree(root)
    wanted = set(node_ids) if node_ids is not None else None

    full = render_svg_rgba(svg, size=size)
    viewbox = _parse_viewbox(root) or (0.0, 0.0, float(size), float(size))

    stats: dict[str, dict[str, Any]] = {}
    for node in indexed:
        if wanted is not None and node.node_id not in wanted:
            continue
        original_style = _hide_element(node.element)
        try:
            without = render_svg_rgba(serialize_svg(root), size=size)
        finally:
            _restore_element(node.element, original_style)
        stats[node.node_id] = _stats_from_diff(full, without, viewbox, threshold=threshold)
    return stats


class VisualStatsCache:
    """Disk cache for node_visual_stats keyed by SVG content and render size.

    Benchmark inputs are frozen, so each SVG's stats are computed once and
    shared across runs and architectures.
    """

    FORMAT = "svgpatchlab.visual_stats.v1"

    def __init__(self, cache_dir: str | Path = ".cache/visual_stats"):
        self.cache_dir = Path(cache_dir)

    def _path(self, svg: str, size: int) -> Path:
        key = hashlib.sha256(f"{self.FORMAT}:{size}:{svg}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def get_or_compute(self, svg: str, size: int = 64) -> dict[str, dict[str, Any]]:
        path = self._path(svg, size)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))["stats"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        stats = node_visual_stats(svg, size=size)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {"format": self.FORMAT, "size": size, "stats": stats}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return stats


def render_node_mask(svg: str, node_id: str, size: int = 64) -> bytes:
    """Render SVG with only the given node visible, all others transparent.

    Used by Plan A vision module to generate per-node difference images.
    """
    import copy
    from svgpatchlab.core.xml import index_tree, parse_svg, serialize_svg

    root = parse_svg(svg)
    indexed = index_tree(root)
    node_map = {node.node_id: node for node in indexed}
    if node_id not in node_map:
        raise ValueError(f"node {node_id} not found in SVG")

    ancestors: set[str] = set()
    current = node_map[node_id].parent_id
    while current:
        ancestors.add(current)
        current = node_map[current].parent_id

    root_copy = copy.deepcopy(root)
    for node in index_tree(root_copy):
        if node.node_id == node_id:
            pass
        elif node.node_id in ancestors:
            for attr in ("fill", "stroke", "opacity"):
                node.element.attrib.pop(attr, None)
        else:
            node.element.attrib["opacity"] = "0"

    return render_svg_png(serialize_svg(root_copy), size=size, background=None)
