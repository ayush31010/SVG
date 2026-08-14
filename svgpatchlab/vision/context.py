from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from svgpatchlab.core.patch import extract_json_object
from svgpatchlab.models import ModelAdapter
from svgpatchlab.types import ModelRequest

from .render import render_node_comparison


PROMPT_VERSION = 1
DEFAULT_TAGS = {
    "circle",
    "ellipse",
    "g",
    "image",
    "line",
    "path",
    "polygon",
    "polyline",
    "rect",
    "text",
    "use",
}


class VisionContextAnnotator:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.max_nodes = int(config.get("max_nodes", 48))
        self.image_size = int(config.get("image_size", 384))
        self.min_visible_pixels = int(config.get("min_visible_pixels", 12))
        self.include_root = bool(config.get("include_root", False))
        self.tags = set(config.get("tags", sorted(DEFAULT_TAGS)))
        cache_dir = config.get("cache_dir")
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def annotate(
        self,
        svg: str,
        scene: dict[str, Any],
        model: ModelAdapter | None,
        request_id: str,
    ) -> dict[str, Any]:
        if not self.enabled:
            return scene
        if model is None:
            raise ValueError("vision_context is enabled but no vision model is configured")

        annotated = copy.deepcopy(scene)
        svg_hash = hashlib.sha256(svg.encode()).hexdigest()
        count = 0

        for node in annotated["nodes"]:
            if count >= self.max_nodes:
                break
            if not self._is_candidate(node):
                continue

            node_id = str(node["id"])
            cached = self._read_cache(svg_hash, node_id)
            if cached is not None:
                node["visual_context"] = cached
                count += 1
                continue

            rendered = render_node_comparison(svg, node_id, size=self.image_size)
            if rendered is None or rendered.visible_pixels < self.min_visible_pixels:
                continue

            annotation = self._describe_node(
                node=node,
                rendered=rendered,
                model=model,
                request_id=f"{request_id}:vision:{node_id}",
            )
            annotation["bbox"] = list(rendered.bbox)
            annotation["visible_pixels"] = rendered.visible_pixels
            node["visual_context"] = annotation
            self._write_cache(svg_hash, node_id, annotation)
            count += 1

        annotated["visual_context"] = {
            "format": "svgpatchlab.visual_context.v1",
            "annotated_node_count": count,
            "image_size": self.image_size,
            "prompt_version": PROMPT_VERSION,
        }
        return annotated

    def _is_candidate(self, node: dict[str, Any]) -> bool:
        if node.get("id") == "n0" and not self.include_root:
            return False
        return str(node.get("tag")) in self.tags

    def _cache_path(self, svg_hash: str, node_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        name = f"{node_id}-s{self.image_size}-v{PROMPT_VERSION}.json"
        return self.cache_dir / svg_hash[:16] / name

    def _read_cache(self, svg_hash: str, node_id: str) -> dict[str, Any] | None:
        path = self._cache_path(svg_hash, node_id)
        if path is None or not path.exists():
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return self._normalize_annotation(value)

    def _write_cache(self, svg_hash: str, node_id: str, annotation: dict[str, Any]) -> None:
        path = self._cache_path(svg_hash, node_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(annotation, indent=2, sort_keys=True) + "\n")

    def _describe_node(
        self,
        node: dict[str, Any],
        rendered,
        model: ModelAdapter,
        request_id: str,
    ) -> dict[str, Any]:
        prompt = _vision_prompt(node)
        response = model.generate(
            ModelRequest(
                prompt=prompt,
                images=(rendered.data_url,),
                metadata={"request_id": request_id},
            )
        )
        try:
            parsed = extract_json_object(response.text)
        except Exception:
            parsed = {"summary": response.text.strip(), "labels": []}
        return self._normalize_annotation(parsed)

    def _normalize_annotation(self, value: dict[str, Any]) -> dict[str, Any]:
        labels = value.get("labels", [])
        if not isinstance(labels, list):
            labels = []
        labels = [
            str(label).strip()[:64]
            for label in labels
            if isinstance(label, (str, int, float)) and str(label).strip()
        ][:8]
        summary = str(value.get("summary", "")).strip()[:240]
        role = str(value.get("role", "")).strip()[:80]
        result: dict[str, Any] = {
            "summary": summary,
            "labels": labels,
        }
        if role:
            result["role"] = role
        if "confidence" in value:
            try:
                result["confidence"] = max(0.0, min(1.0, float(value["confidence"])))
            except (TypeError, ValueError):
                pass
        for field in ("bbox", "visible_pixels"):
            if field in value:
                result[field] = value[field]
        return result


def _vision_prompt(node: dict[str, Any]) -> str:
    node_json = json.dumps(node, sort_keys=True)
    return f"""You are annotating one SVG DOM node for an SVG editing system.
The attached image has three panels:
1. left: the full SVG render
2. middle: only the selected node or node subtree
3. right: the full SVG render with the selected node highlighted in magenta

Compare the isolated node with the full image and describe what visible object
or object-part this node represents in context. Use semantic terms a user might
say in an edit instruction, such as window, eye, roof, shadow, outline, handle,
letter, background, wheel, or highlight. If the node is only a tiny fragment,
say that clearly.

SVG node metadata:
{node_json}

Return only JSON with this shape:
{{"summary":"short phrase","labels":["label1","label2"],"role":"object|part|style|shadow|outline|background|unknown","confidence":0.0}}
"""
