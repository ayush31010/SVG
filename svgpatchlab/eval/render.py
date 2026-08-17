from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RendererUnavailable(RuntimeError):
    pass


class IDBufferUnsupported(RuntimeError):
    """Raised when an SVG cannot be attributed safely with a flat ID buffer."""

    def __init__(self, features: tuple[str, ...] | list[str]):
        self.features = tuple(features)
        details = ", ".join(self.features)
        super().__init__(f"SVG ID buffer does not support: {details}")


def _chromium_executable() -> str | None:
    """Find an installed Chromium-family browser for the Windows fallback."""
    for name in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "chrome",
        "msedge",
    ):
        found = shutil.which(name)
        if found:
            return found

    candidates: list[Path] = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        root = os.environ.get(variable)
        if not root:
            continue
        root_path = Path(root)
        candidates.extend(
            (
                root_path / "Google/Chrome/Application/chrome.exe",
                root_path / "Microsoft/Edge/Application/msedge.exe",
            )
        )
    candidates.extend(
        (
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        )
    )
    return next((str(path) for path in candidates if path.is_file()), None)


class _ChromiumSVGRenderer:
    """Small CairoSVG-compatible adapter used when native Cairo is unavailable.

    A restrictive CSP, disabled JavaScript, and background-networking flags
    prevent SVG input from reading local/network data. Rendering input and
    output live in a fresh temporary directory.
    """

    def __init__(self, executable: str):
        self.executable = executable

    def svg2png(
        self,
        *,
        bytestring: bytes,
        output_width: int,
        output_height: int,
        background_color: str | None,
    ) -> bytes:
        from svgpatchlab.core.xml import parse_svg, serialize_svg

        width = int(output_width)
        height = int(output_height)
        if width <= 0 or height <= 0:
            raise ValueError("render dimensions must be positive")
        svg = serialize_svg(parse_svg(bytestring.decode("utf-8")))
        background = "transparent" if background_color is None else background_color
        if any(character in background for character in '{};<>"\'\\'):
            raise ValueError("unsafe background color")
        html = (
            "<!doctype html>"
            '<meta http-equiv="Content-Security-Policy" '
            "content=\"default-src 'none'; img-src data:; "
            "font-src data:; style-src 'unsafe-inline'\">"
            "<style>"
            "html,body{margin:0;width:100%;height:100%;overflow:hidden;"
            f"background:{background}}}"
            "svg{display:block;width:100%!important;height:100%!important}"
            "</style>"
            f"{svg}"
        )

        with tempfile.TemporaryDirectory(prefix="svgpatchlab-render-") as directory:
            root = Path(directory)
            source = root / "render.html"
            output = root / "render.png"
            source.write_text(html, encoding="utf-8")
            command = [
                self.executable,
                "--headless=new",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-default-apps",
                "--disable-extensions",
                "--disable-gpu",
                "--disable-sync",
                "--hide-scrollbars",
                "--metrics-recording-only",
                "--no-default-browser-check",
                "--no-first-run",
                "--blink-settings=scriptEnabled=false",
                "--default-background-color=00000000",
                "--force-device-scale-factor=1",
                f"--window-size={width},{height}",
                f"--screenshot={output}",
                source.as_uri(),
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
            except subprocess.TimeoutExpired as exc:
                raise RendererUnavailable(
                    "headless Chromium SVG rendering timed out"
                ) from exc
            # On Windows the Chrome launcher can return just before its headless
            # child flushes the screenshot, even with an isolated profile.
            for _ in range(50):
                if output.exists() and output.stat().st_size > 0:
                    break
                time.sleep(0.1)
            if completed.returncode != 0 or not output.exists():
                details = completed.stderr.decode(errors="replace").strip()
                raise RendererUnavailable(
                    "headless Chromium SVG rendering failed "
                    f"(exit code {completed.returncode})"
                    + (f": {details}" if details else "")
                )
            return output.read_bytes()


def _png_data_url(png: bytes) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@dataclass(frozen=True)
class SVGIDMap:
    """One-render SVG element ownership buffer.

    `rgb` is the rendered uint8 RGB image and `labels` is an int32 image with
    indexes into `leaf_node_ids` (-1 means no confidently decoded owner).
    Colors are deterministic functions of DOM node IDs.
    """

    png: bytes
    rgb: Any
    labels: Any
    node_colors: dict[str, str]
    color_nodes: dict[str, str]
    leaf_node_ids: tuple[str, ...]
    size: int

    @property
    def data_url(self) -> str:
        return _png_data_url(self.png)


@dataclass(frozen=True)
class SVGVisualContext:
    """Normal render, optional ID render, and decoded per-node visual stats."""

    normal_png: bytes
    normal_rgba: Any
    id_map: SVGIDMap | None
    stats: dict[str, dict[str, Any]]
    method: str
    unsupported_features: tuple[str, ...] = ()

    @property
    def normal_data_url(self) -> str:
        return _png_data_url(self.normal_png)

    @property
    def id_data_url(self) -> str | None:
        return self.id_map.data_url if self.id_map is not None else None


def _dependencies():
    # OSError: cairosvg is installed but the native cairo library is missing
    # (the usual state on Windows without a GTK runtime).
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RendererUnavailable(
            "rendering requires numpy and Pillow: pip install -e '.[eval]'"
        ) from exc

    try:
        import cairosvg

        # Importing CairoSVG can succeed before the native library is touched
        # in some installations, so resolve the surface eagerly.
        _ = cairosvg.surface
        renderer: Any = cairosvg
    except (ImportError, OSError):
        chromium = _chromium_executable()
        if chromium is None:
            raise RendererUnavailable(
                "rendering requires native Cairo or an installed Chrome/Chromium browser"
            )
        renderer = _ChromiumSVGRenderer(chromium)
    return renderer, np, Image


_RENDERER_PROBED = False
_RENDERER_PROBE_ERROR: RendererUnavailable | None = None


def ensure_renderer() -> None:
    global _RENDERER_PROBED, _RENDERER_PROBE_ERROR
    if _RENDERER_PROBE_ERROR is not None:
        raise _RENDERER_PROBE_ERROR
    if _RENDERER_PROBED:
        return
    renderer, _, _ = _dependencies()
    if isinstance(renderer, _ChromiumSVGRenderer):
        try:
            renderer.svg2png(
                bytestring=(
                    b'<svg xmlns="http://www.w3.org/2000/svg" '
                    b'viewBox="0 0 1 1"><rect width="1" height="1"/></svg>'
                ),
                output_width=1,
                output_height=1,
                background_color=None,
            )
        except RendererUnavailable as exc:
            _RENDERER_PROBE_ERROR = exc
            raise
    _RENDERER_PROBED = True


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


def render_svg_data_url(svg: str, size: int = 512) -> str:
    png = render_svg_png(svg, size=size, background="white")
    return _png_data_url(png)


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


def _png_rgba_array(png: bytes):
    """Decode PNG bytes to a writable uint8 RGBA array."""
    _, np, Image = _dependencies()
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), dtype=np.uint8).copy()


_PAINTABLE_TAGS = {
    "circle",
    "ellipse",
    "line",
    "path",
    "polygon",
    "polyline",
    "rect",
    "text",
}

_DEFINITION_CONTAINERS = {
    "clipPath",
    "defs",
    "desc",
    "filter",
    "linearGradient",
    "marker",
    "mask",
    "metadata",
    "pattern",
    "radialGradient",
    "style",
    "symbol",
    "title",
}

_UNSUPPORTED_ID_TAGS = {
    "animate",
    "animateColor",
    "animateMotion",
    "animateTransform",
    "foreignObject",
    "image",
    "script",
    "set",
    "style",
    "use",
}


def _style_declarations(value: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for declaration in value.split(";"):
        if ":" not in declaration:
            continue
        name, item = declaration.split(":", 1)
        if name.strip():
            declarations[name.strip().lower()] = item.strip()
    return declarations


def _attribute_map(element) -> dict[str, str]:
    """Return local, lower-case attribute names plus inline style values."""
    from svgpatchlab.core.xml import local_name

    attributes = {local_name(name).lower(): value for name, value in element.attrib.items()}
    attributes.update(_style_declarations(element.attrib.get("style", "")))
    return attributes


def _opacity_is_endpoint(value: str) -> bool:
    """Whether opacity is exactly zero or one (both are ID-buffer safe)."""
    normalized = value.lower().replace("!important", "").strip()
    try:
        if normalized.endswith("%"):
            number = float(normalized[:-1]) / 100.0
        else:
            number = float(normalized)
    except ValueError:
        return False
    return number in (0.0, 1.0)


def _id_buffer_unsupported_features(root, indexed) -> tuple[str, ...]:
    """Conservatively identify constructs that destroy one-pixel/one-owner IDs.

    Geometry, transforms, painter order, and clip paths are supported. Features
    that composite multiple owners, paint holes/textures, clone content, embed
    raster content, or apply CSS outside inline/presentation attributes are
    routed to counterfactual rendering instead.
    """
    from svgpatchlab.core.xml import local_name

    reasons: list[str] = []
    for node in indexed:
        element = node.element
        tag = local_name(element.tag)
        attributes = _attribute_map(element)
        prefix = f"{node.node_id}:<{tag}>"

        if tag in _UNSUPPORTED_ID_TAGS:
            reasons.append(f"{prefix} {tag}")

        if tag == "text" and list(element):
            reasons.append(f"{prefix} complex-text")

        for name in ("filter", "mask", "marker-start", "marker-mid", "marker-end"):
            value = attributes.get(name, "").strip().lower()
            if value and value != "none":
                reasons.append(f"{prefix} {name}")

        for name in ("fill", "stroke"):
            value = attributes.get(name, "").strip().lower()
            if "url(" in value:
                reasons.append(f"{prefix} {name}-paint-server")
            if (
                value.startswith("rgba(")
                or value.startswith("hsla(")
                or (value.startswith("#") and len(value) in (5, 9))
            ):
                reasons.append(f"{prefix} alpha-color")

        for name in ("opacity", "fill-opacity", "stroke-opacity"):
            value = attributes.get(name)
            if value is not None and not _opacity_is_endpoint(value):
                reasons.append(f"{prefix} fractional-{name}")

        blend = attributes.get("mix-blend-mode", "").strip().lower()
        if blend and blend != "normal":
            reasons.append(f"{prefix} mix-blend-mode")

    # Preserve document order while removing duplicate descriptions.
    return tuple(dict.fromkeys(reasons))


def id_buffer_unsupported_features(svg: str) -> tuple[str, ...]:
    """Return reasons the fast ID-buffer path would be unsafe, if any."""
    from svgpatchlab.core.xml import index_tree, parse_svg

    root = parse_svg(svg)
    return _id_buffer_unsupported_features(root, index_tree(root))


def _renderable_leaf_nodes(indexed):
    """Select paintable DOM nodes outside definition-only subtrees."""
    from svgpatchlab.core.xml import local_name

    in_definition: dict[str, bool] = {}
    leaves = []
    for node in indexed:
        tag = local_name(node.element.tag)
        parent_is_definition = in_definition.get(node.parent_id or "", False)
        is_definition = parent_is_definition or tag in _DEFINITION_CONTAINERS
        in_definition[node.node_id] = is_definition
        if not is_definition and tag in _PAINTABLE_TAGS:
            leaves.append(node)
    return leaves


def _id_rgb(node_id: str) -> tuple[int, int, int]:
    """Deterministic collision-free color for index_tree's n<number> IDs."""
    if not node_id.startswith("n"):
        raise ValueError(f"invalid SVG node ID: {node_id}")
    try:
        ordinal = int(node_id[1:]) + 1
    except ValueError as exc:
        raise ValueError(f"invalid SVG node ID: {node_id}") from exc
    if not 0 < ordinal < 2**24:
        raise ValueError("SVG ID buffer supports at most 16,777,214 DOM nodes")

    # Multiplication by an odd number permutes the 24-bit integer space. Nearby
    # DOM IDs therefore receive well-separated colors without any collisions.
    code = (ordinal * 0x9E3779) & 0xFFFFFF
    return (code >> 16) & 0xFF, (code >> 8) & 0xFF, code & 0xFF


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _resolved_leaf_paints(indexed) -> dict[str, dict[str, str]]:
    """Resolve the inherited fill/stroke subset needed for flat recoloring."""
    resolved: dict[str, dict[str, str]] = {}
    for node in indexed:
        inherited = dict(
            resolved.get(node.parent_id or "", {"fill": "black", "stroke": "none"})
        )
        declarations = _style_declarations(node.element.attrib.get("style", ""))
        for name in ("fill", "stroke"):
            if name in node.element.attrib:
                inherited[name] = node.element.attrib[name]
            if name in declarations:
                inherited[name] = declarations[name].replace("!important", "").strip()
        resolved[node.node_id] = inherited
    return resolved


def _paint_is_none(value: str) -> bool:
    return value.strip().lower().replace("!important", "") in ("none", "transparent")


def _force_leaf_id_paint(element, paint: dict[str, str], color: str) -> None:
    """Override paint only; geometry, transforms, clipping, and order stay intact."""
    original = element.attrib.get("style", "").strip().rstrip(";")
    declarations = []
    for name in ("fill", "stroke"):
        replacement = "none" if _paint_is_none(paint.get(name, "none")) else color
        declarations.append(f"{name}:{replacement}!important")
    forced = ";".join(declarations)
    element.attrib["style"] = f"{original};{forced}" if original else forced


def _decode_id_labels(
    rgba,
    leaf_node_ids: tuple[str, ...],
    node_rgbs: dict[str, tuple[int, int, int]],
    tolerance: int,
):
    """Decode rendered flat colors to leaf indexes, rejecting uncertain pixels."""
    import numpy as np

    if tolerance < 0 or tolerance > 255:
        raise ValueError("decode tolerance must be between 0 and 255")

    height, width = rgba.shape[:2]
    labels = np.full((height, width), -1, dtype=np.int32)
    if not leaf_node_ids:
        return labels

    rgb = rgba[:, :, :3].astype(np.int32)
    best_distance = np.full((height, width), 3 * 255 * 255 + 1, dtype=np.int32)
    best_label = np.full((height, width), -1, dtype=np.int32)
    for label, node_id in enumerate(leaf_node_ids):
        color = np.asarray(node_rgbs[node_id], dtype=np.int32)
        delta = rgb - color
        distance = np.sum(delta * delta, axis=2, dtype=np.int32)
        better = distance < best_distance
        best_distance[better] = distance[better]
        best_label[better] = label

    confident = (rgba[:, :, 3] > 0) & (best_distance <= 3 * tolerance * tolerance)
    labels[confident] = best_label[confident]
    return labels


def _render_svg_id_map_from_parsed(
    root,
    indexed,
    size: int,
    decode_tolerance: int,
) -> SVGIDMap:
    from svgpatchlab.core.xml import index_tree, serialize_svg

    root_copy = copy.deepcopy(root)
    copy_indexed = index_tree(root_copy)
    leaves = _renderable_leaf_nodes(copy_indexed)
    leaf_node_ids = tuple(node.node_id for node in leaves)
    paints = _resolved_leaf_paints(copy_indexed)
    node_rgbs = {node_id: _id_rgb(node_id) for node_id in leaf_node_ids}
    node_colors = {node_id: _rgb_hex(rgb) for node_id, rgb in node_rgbs.items()}

    for node in leaves:
        _force_leaf_id_paint(
            node.element,
            paints[node.node_id],
            node_colors[node.node_id],
        )

    png = render_svg_png(serialize_svg(root_copy), size=size, background=None)
    rgba = _png_rgba_array(png)
    labels = _decode_id_labels(rgba, leaf_node_ids, node_rgbs, decode_tolerance)
    color_nodes = {color: node_id for node_id, color in node_colors.items()}
    return SVGIDMap(
        png=png,
        rgb=rgba[:, :, :3].copy(),
        labels=labels,
        node_colors=node_colors,
        color_nodes=color_nodes,
        leaf_node_ids=leaf_node_ids,
        size=size,
    )


def render_svg_id_map(
    svg: str,
    size: int = 64,
    decode_tolerance: int = 12,
) -> SVGIDMap:
    """Render and decode a flat-color SVG element ownership map.

    The function performs one rasterization. It raises IDBufferUnsupported for
    complex compositing/paint features; callers that want the automatic
    counterfactual fallback should use render_svg_visual_context instead.
    """
    from svgpatchlab.core.xml import index_tree, parse_svg

    root = parse_svg(svg)
    indexed = index_tree(root)
    unsupported = _id_buffer_unsupported_features(root, indexed)
    if unsupported:
        raise IDBufferUnsupported(unsupported)
    return _render_svg_id_map_from_parsed(root, indexed, size, decode_tolerance)


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
    preserve_aspect_ratio: str = "xMidYMid meet",
) -> dict[str, Any]:
    """Reduce a full-vs-node-hidden render pair to compact per-node stats.

    Both inputs are float RGBA arrays from render_svg_rgba. The difference
    mask is the node's *visible* contribution, so fully occluded nodes and
    non-rendering nodes (defs, empty groups) come back as visible: false.
    """
    import numpy as np

    diff = np.abs(full - without).max(axis=2)
    mask = diff > threshold
    return _stats_from_mask(
        full,
        mask,
        viewbox,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )


def _viewbox_pixel_transform(
    viewbox: tuple[float, float, float, float],
    viewport_width: int,
    viewport_height: int,
    preserve_aspect_ratio: str,
) -> tuple[float, float, float, float]:
    """Return SVG user-unit scales and viewport offsets."""

    _, _, viewbox_width, viewbox_height = viewbox
    raw_parts = preserve_aspect_ratio.strip().split()
    if raw_parts and raw_parts[0] == "defer":
        raw_parts = raw_parts[1:]
    alignment = raw_parts[0] if raw_parts else "xMidYMid"
    mode = raw_parts[1] if len(raw_parts) > 1 else "meet"

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


def _stats_from_mask(
    full,
    mask,
    viewbox: tuple[float, float, float, float],
    preserve_aspect_ratio: str = "xMidYMid meet",
) -> dict[str, Any]:
    """Reduce an ownership mask and the original RGBA render to node stats."""
    import numpy as np

    if not mask.any():
        return {"visible": False}

    height, width = mask.shape
    rows = np.any(mask, axis=1).nonzero()[0]
    columns = np.any(mask, axis=0).nonzero()[0]
    vb_x, vb_y, vb_w, vb_h = viewbox
    scale_x, scale_y, offset_x, offset_y = _viewbox_pixel_transform(
        viewbox,
        width,
        height,
        preserve_aspect_ratio,
    )

    def viewbox_x(pixel: float) -> float:
        return min(max(vb_x + (pixel - offset_x) / scale_x, vb_x), vb_x + vb_w)

    def viewbox_y(pixel: float) -> float:
        return min(max(vb_y + (pixel - offset_y) / scale_y, vb_y), vb_y + vb_h)

    x0 = viewbox_x(float(columns[0]))
    x1 = viewbox_x(float(columns[-1] + 1))
    y0 = viewbox_y(float(rows[0]))
    y1 = viewbox_y(float(rows[-1] + 1))

    cy, cx = np.nonzero(mask)
    centroid_x = viewbox_x(float(cx.mean() + 0.5))
    centroid_y = viewbox_y(float(cy.mean() + 0.5))
    centroid_col = (centroid_x - vb_x) / vb_w
    centroid_row = (centroid_y - vb_y) / vb_h

    stats: dict[str, Any] = {
        "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
        "area_pct": round(100.0 * mask.sum() / mask.size, 2),
        "position": _position_word(centroid_col, centroid_row),
    }

    opaque = mask & (full[:, :, 3] > 0.1)
    sample = full[opaque if opaque.any() else mask][:, :3]
    rgb = np.rint(np.clip(sample, 0.0, 1.0) * 255.0).astype(np.uint8)
    bins = rgb.astype(np.uint16) // 16
    bin_keys = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    dominant_key = int(np.bincount(bin_keys).argmax())
    dominant_pixels = rgb[bin_keys == dominant_key]
    red, green, blue = (
        int(round(float(np.median(dominant_pixels[:, channel]))))
        for channel in range(3)
    )
    stats["color"] = f"#{red:02x}{green:02x}{blue:02x}"
    return stats


def _stats_from_id_map(
    normal_rgba,
    id_map: SVGIDMap,
    indexed,
    viewbox: tuple[float, float, float, float],
    node_ids: list[str] | None = None,
    preserve_aspect_ratio: str = "xMidYMid meet",
) -> dict[str, dict[str, Any]]:
    """Aggregate visible leaf labels upward through the SVG DOM."""
    import numpy as np

    wanted = set(node_ids) if node_ids is not None else None
    node_by_id = {node.node_id: node for node in indexed}
    label_by_leaf = {
        node_id: label for label, node_id in enumerate(id_map.leaf_node_ids)
    }
    descendant_labels: dict[str, list[int]] = {
        node.node_id: [] for node in indexed
    }
    for leaf_id, label in label_by_leaf.items():
        current: str | None = leaf_id
        while current is not None:
            descendant_labels[current].append(label)
            current = node_by_id[current].parent_id

    full = normal_rgba.astype(np.float32) / 255.0
    stats: dict[str, dict[str, Any]] = {}
    for node in indexed:
        if wanted is not None and node.node_id not in wanted:
            continue
        labels = descendant_labels[node.node_id]
        if not labels:
            node_stats: dict[str, Any] = {"visible": False}
            if node.node_id in id_map.node_colors:
                node_stats["id_color"] = id_map.node_colors[node.node_id]
            stats[node.node_id] = node_stats
            continue
        if len(labels) == 1:
            mask = id_map.labels == labels[0]
        else:
            mask = np.isin(id_map.labels, labels)
        node_stats = _stats_from_mask(
            full,
            mask,
            viewbox,
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
        if node.node_id in id_map.node_colors:
            node_stats["id_color"] = id_map.node_colors[node.node_id]
        stats[node.node_id] = node_stats
    return stats


def _hide_element(element) -> str | None:
    """Append display:none to the element's style; return the prior style."""
    original = element.attrib.get("style")
    element.attrib["style"] = (
        (f"{original};" if original else "") + "display:none!important"
    )
    return original


def _restore_element(element, original_style: str | None) -> None:
    if original_style is None:
        element.attrib.pop("style", None)
    else:
        element.attrib["style"] = original_style


def _counterfactual_visual_stats(
    root,
    indexed,
    full,
    viewbox: tuple[float, float, float, float],
    node_ids: list[str] | None,
    size: int,
    threshold: float,
    preserve_aspect_ratio: str,
) -> dict[str, dict[str, Any]]:
    from svgpatchlab.core.xml import serialize_svg

    wanted = set(node_ids) if node_ids is not None else None
    stats: dict[str, dict[str, Any]] = {}
    for node in indexed:
        if wanted is not None and node.node_id not in wanted:
            continue
        original_style = _hide_element(node.element)
        try:
            without = render_svg_rgba(serialize_svg(root), size=size)
        finally:
            _restore_element(node.element, original_style)
        stats[node.node_id] = _stats_from_diff(
            full,
            without,
            viewbox,
            threshold=threshold,
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
    return stats


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
    from svgpatchlab.core.xml import index_tree, parse_svg

    root = parse_svg(svg)
    indexed = index_tree(root)

    full = render_svg_rgba(svg, size=size)
    viewbox = _parse_viewbox(root) or (0.0, 0.0, float(size), float(size))
    preserve_aspect_ratio = root.attrib.get(
        "preserveAspectRatio",
        "xMidYMid meet",
    )
    return _counterfactual_visual_stats(
        root,
        indexed,
        full,
        viewbox,
        node_ids,
        size,
        threshold,
        preserve_aspect_ratio,
    )


def render_svg_visual_context(
    svg: str,
    node_ids: list[str] | None = None,
    size: int = 64,
    decode_tolerance: int = 12,
    fallback: bool = True,
    threshold: float = 0.02,
) -> SVGVisualContext:
    """Render reusable model context and fast, occlusion-aware node stats.

    Supported flat SVGs cost two rasterizations total: the original and a
    unique-color ID buffer. Groups are unions of their visible descendant leaf
    IDs. When complex paints/compositing are detected, `fallback=True` routes
    to the existing N+1 counterfactual method; `fallback=False` raises
    IDBufferUnsupported instead.
    """
    from svgpatchlab.core.xml import index_tree, parse_svg

    root = parse_svg(svg)
    indexed = index_tree(root)
    unsupported = _id_buffer_unsupported_features(root, indexed)
    if unsupported and not fallback:
        raise IDBufferUnsupported(unsupported)

    normal_png = render_svg_png(svg, size=size, background=None)
    normal_rgba = _png_rgba_array(normal_png)
    viewbox = _parse_viewbox(root) or (0.0, 0.0, float(size), float(size))
    preserve_aspect_ratio = root.attrib.get(
        "preserveAspectRatio",
        "xMidYMid meet",
    )

    if unsupported:
        stats = _counterfactual_visual_stats(
            root,
            indexed,
            normal_rgba.astype("float32") / 255.0,
            viewbox,
            node_ids,
            size,
            threshold,
            preserve_aspect_ratio,
        )
        return SVGVisualContext(
            normal_png=normal_png,
            normal_rgba=normal_rgba,
            id_map=None,
            stats=stats,
            method="counterfactual",
            unsupported_features=unsupported,
        )

    id_map = _render_svg_id_map_from_parsed(
        root,
        indexed,
        size=size,
        decode_tolerance=decode_tolerance,
    )
    stats = _stats_from_id_map(
        normal_rgba,
        id_map,
        indexed,
        viewbox,
        node_ids=node_ids,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )
    return SVGVisualContext(
        normal_png=normal_png,
        normal_rgba=normal_rgba,
        id_map=id_map,
        stats=stats,
        method="id_buffer",
    )


def id_buffer_visual_stats(
    svg: str,
    node_ids: list[str] | None = None,
    size: int = 64,
    decode_tolerance: int = 12,
    fallback: bool = True,
) -> dict[str, dict[str, Any]]:
    """Convenience wrapper returning only stats from render_svg_visual_context."""
    return render_svg_visual_context(
        svg,
        node_ids=node_ids,
        size=size,
        decode_tolerance=decode_tolerance,
        fallback=fallback,
    ).stats


class VisualStatsCache:
    """Disk cache for node_visual_stats keyed by SVG content and render size.

    Benchmark inputs are frozen, so each SVG's stats are computed once and
    shared across runs and architectures.
    """

    FORMAT = "svgpatchlab.visual_stats.v2"

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
