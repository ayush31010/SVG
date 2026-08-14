from __future__ import annotations

import base64
import io


class RendererUnavailable(RuntimeError):
    pass


def _cairosvg():
    import os
    import sys
    if sys.platform == "darwin":
        hb_path = "/opt/homebrew/lib"
        curr_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
        if hb_path not in curr_dyld:
            os.environ["DYLD_LIBRARY_PATH"] = f"{hb_path}:{curr_dyld}".rstrip(":")
    try:
        import cairosvg
    except ImportError as exc:
        raise RendererUnavailable(
            "SVG rendering requires the optional dependencies: pip install -e '.[vision]'"
        ) from exc
    return cairosvg


def _pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RendererUnavailable(
            "image processing requires the optional dependencies: pip install -e '.[vision]'"
        ) from exc
    return Image


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RendererUnavailable(
            "MSE evaluation requires the optional dependencies: pip install -e '.[eval]'"
        ) from exc
    return np


def ensure_renderer() -> None:
    _cairosvg()
    _pillow()
    _numpy()


def ensure_svg_renderer() -> None:
    _cairosvg()
    _pillow()


def render_svg_png(
    svg: str,
    size: int = 72,
    background: str | None = "white",
) -> bytes:
    cairosvg = _cairosvg()
    return cairosvg.svg2png(
        bytestring=svg.encode(),
        output_width=size,
        output_height=size,
        background_color=background,
    )


def render_svg_array(svg: str, size: int = 72, background: str = "white"):
    np = _numpy()
    Image = _pillow()
    png = render_svg_png(svg, size=size, background=background)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255.0


def png_data_url(png: bytes) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_mse(candidate_svg: str, answer_svg: str, size: int = 72) -> float:
    np = _numpy()
    candidate = render_svg_array(candidate_svg, size=size)
    answer = render_svg_array(answer_svg, size=size)
    return float(np.mean((candidate - answer) ** 2))
