from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NAMESPACE)


class SVGParseError(ValueError):
    pass


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_svg(svg: str, max_bytes: int = 2_000_000) -> ET.Element:
    encoded = svg.encode("utf-8")
    if len(encoded) > max_bytes:
        raise SVGParseError(f"SVG exceeds {max_bytes} bytes")
    upper = svg.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise SVGParseError("DTD and entity declarations are not allowed")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise SVGParseError(str(exc)) from exc
    if local_name(root.tag) != "svg":
        raise SVGParseError("document root must be <svg>")
    return root


def serialize_svg(root: ET.Element) -> str:
    return ET.tostring(root, encoding="unicode", short_empty_elements=True)


@dataclass(frozen=True)
class IndexedNode:
    node_id: str
    element: ET.Element
    parent_id: str | None
    depth: int
    child_index: int


def index_tree(root: ET.Element) -> list[IndexedNode]:
    indexed: list[IndexedNode] = []

    def visit(element: ET.Element, parent_id: str | None, depth: int, child_index: int) -> None:
        node_id = f"n{len(indexed)}"
        indexed.append(IndexedNode(node_id, element, parent_id, depth, child_index))
        for index, child in enumerate(list(element)):
            visit(child, node_id, depth + 1, index)

    visit(root, None, 0, 0)
    return indexed


def element_fingerprint(element: ET.Element) -> str:
    payload = ET.tostring(element, encoding="utf-8", short_empty_elements=True)
    return hashlib.sha256(payload).hexdigest()


def protected_geometry(root: ET.Element) -> dict[str, dict[str, str]]:
    protected: dict[str, dict[str, str]] = {}
    for node in index_tree(root):
        attrs: dict[str, str] = {}
        for name in ("d", "points"):
            if name in node.element.attrib:
                attrs[name] = hashlib.sha256(node.element.attrib[name].encode()).hexdigest()
        if attrs:
            protected[node.node_id] = attrs
    return protected


def normalized_tree(root: ET.Element) -> list[tuple[str, str, tuple[tuple[str, str], ...], str]]:
    result = []
    for node in index_tree(root):
        text = (node.element.text or "").strip()
        result.append(
            (
                node.node_id,
                local_name(node.element.tag),
                tuple(sorted(node.element.attrib.items())),
                text,
            )
        )
    return result

