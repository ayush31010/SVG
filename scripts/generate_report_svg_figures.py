from __future__ import annotations

import json
import re
from pathlib import Path

from svgpatchlab.architectures.semantic import _counterfactual_difference
from svgpatchlab.core.xml import index_tree, parse_svg, serialize_svg
from svgpatchlab.eval.render import render_svg_png


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "output" / "pdf" / "figures"
RENDER_SIZE = 512

COMPLETE_QUERY = (
    ROOT / "SVGEditBench" / "2_SetContour" / "query" / "1f324.txt"
)
COMPLETED_OUTPUT_CANDIDATES = (
    ROOT / "runs" / "qwen-completion-live-smoke-server" / "output.svg",
    ROOT / "runs" / "qwen-completion-live-smoke" / "output.svg",
)

# Frozen input used by tmp/qwen_completion_live_smoke.py and recorded in the
# corresponding server result. Only the upper half of the intended sun exists.
INCOMPLETE_SOURCE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">'
    '<path fill="#FFAC33" d="M8 18A10 10 0 0 1 28 18H8Z"/>'
    '<rect x="5" y="18" width="26" height="14" rx="7" fill="#E1E8ED"/>'
    "</svg>"
)


def _svg_from_query(path: Path) -> str:
    match = re.search(
        r"```svg\s*(<svg\b.*?</svg>)\s*```",
        path.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"no fenced SVG found in {path}")
    return match.group(1)


def _first_existing(paths: tuple[Path, ...]) -> Path:
    try:
        return next(path for path in paths if path.is_file())
    except StopIteration as exc:
        raise FileNotFoundError(
            "none of the expected paths exists: "
            + ", ".join(str(path) for path in paths)
        ) from exc


def _without_node(svg: str, node_id: str, *, remove: bool) -> str:
    root = parse_svg(svg)
    indexed = index_tree(root)
    selected = next(node for node in indexed if node.node_id == node_id)
    if remove:
        parent = next(
            node.element for node in indexed if node.node_id == selected.parent_id
        )
        parent.remove(selected.element)
    else:
        original = selected.element.attrib.get("style", "").strip().rstrip(";")
        hidden = "display:none!important"
        selected.element.attrib["style"] = (
            f"{original};{hidden}" if original else hidden
        )
    return serialize_svg(root)


def _write_render(name: str, svg: str) -> None:
    (FIGURE_DIR / name).write_bytes(
        render_svg_png(svg, size=RENDER_SIZE, background="white")
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    complete_source = _svg_from_query(COMPLETE_QUERY)
    temporary_hidden = _without_node(complete_source, "n3", remove=False)
    complete_deleted = _without_node(complete_source, "n3", remove=True)

    full_rgba = render_svg_png(
        complete_source, size=RENDER_SIZE, background=None
    )
    hidden_rgba = render_svg_png(
        temporary_hidden, size=RENDER_SIZE, background=None
    )
    difference, difference_png = _counterfactual_difference(
        full_rgba, hidden_rgba
    )

    incomplete_deleted = _without_node(
        INCOMPLETE_SOURCE, "n2", remove=True
    )
    completed_output = _first_existing(COMPLETED_OUTPUT_CANDIDATES)
    completed = completed_output.read_text(encoding="utf-8")

    _write_render("attribution-full.png", complete_source)
    _write_render("attribution-cloud-hidden.png", temporary_hidden)
    (FIGURE_DIR / "attribution-difference.png").write_bytes(difference_png)
    _write_render("complete-cloud-deleted.png", complete_deleted)
    _write_render("incomplete-before.png", INCOMPLETE_SOURCE)
    _write_render("incomplete-cloud-deleted.png", incomplete_deleted)
    _write_render("incomplete-qwen-completed.png", completed)

    manifest = {
        "render_size": RENDER_SIZE,
        "attribution_source": str(COMPLETE_QUERY.relative_to(ROOT)),
        "attribution_hidden_node": "n3",
        "attribution_difference": difference,
        "complete_deletion": "SVGEditBench source with n3 removed",
        "incomplete_source": "tmp/qwen_completion_live_smoke.py::SOURCE",
        "completed_output": str(completed_output.relative_to(ROOT)),
    }
    (FIGURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
