from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

TAGS = ["svg", "g", "rect", "circle", "ellipse", "path", "line", "polygon", "polyline", "text", "use", "image"]

# Relation types for RGCNConv
REL_PARENT_OF = 0
REL_CHILD_OF = 1
REL_SIBLING_OF = 2
REL_SAME_FILL_AS = 3
NUM_RELATIONS = 4

VISION_SCALAR_DIM = 4  # [has_vision, confidence, log1p(visible_pixels)/10, num_labels/8]

_text_encoder: SentenceTransformer | None = None

def get_text_encoder() -> SentenceTransformer:
    global _text_encoder
    if _text_encoder is None:
        _text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _text_encoder

def vision_scalar_features(annotation: dict[str, Any] | None) -> list[float]:
    if annotation is None:
        return [0.0] * VISION_SCALAR_DIM
    confidence = float(annotation.get("confidence", 0.0)) if "confidence" in annotation else 0.0
    visible_pixels = float(annotation.get("visible_pixels", 0.0))
    num_labels = len(annotation.get("labels", []))
    return [1.0, confidence, math.log1p(visible_pixels) / 10.0, num_labels / 8.0]

def hex_to_rgb(hexcolor: str) -> tuple[float, float, float]:
    h = str(hexcolor).lstrip("#")
    if len(h) != 6:
        return (0.0, 0.0, 0.0)
    try:
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (0.0, 0.0, 0.0)

def node_fill(node: dict[str, Any]) -> str | None:
    return node.get("resolved_style", {}).get("fill") or node.get("attributes", {}).get("fill")

def node_features(node: dict[str, Any]) -> list[float]:
    tag_onehot = [1.0 if node["tag"] == t else 0.0 for t in TAGS]
    depth = [float(node["depth"])]
    fill = node_fill(node)
    has_fill = [1.0 if fill else 0.0]
    rgb = list(hex_to_rgb(fill)) if fill else [0.0, 0.0, 0.0]
    return tag_onehot + depth + has_fill + rgb

def scene_to_graph_inputs(
    scene: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int], torch.Tensor, list[str]]:
    """
    Convert a DOM scene dictionary into PyTorch tensors for GNN input.
    """
    nodes = scene["nodes"]
    id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    x = torch.tensor([node_features(n) for n in nodes], dtype=torch.float)

    vision_scalars, vision_summaries = [], []
    for n in nodes:
        ann = n.get("visual_context")
        vision_scalars.append(vision_scalar_features(ann))
        vision_summaries.append(ann["summary"] if ann else "")
    vision_scalar_x = torch.tensor(vision_scalars, dtype=torch.float)

    edges, edge_types = [], []
    def add_edge(u: int, v: int, rel: int):
        edges.append((u, v))
        edge_types.append(rel)

    # parent_of / child_of
    children_by_parent = defaultdict(list)
    for n in nodes:
        if n["parent"] is not None:
            p_idx, c_idx = id_to_idx[n["parent"]], id_to_idx[n["id"]]
            add_edge(p_idx, c_idx, REL_PARENT_OF)
            add_edge(c_idx, p_idx, REL_CHILD_OF)
            children_by_parent[n["parent"]].append(n["id"])

    # sibling_of
    for sibling_ids in children_by_parent.values():
        for i in sibling_ids:
            for j in sibling_ids:
                if i != j:
                    add_edge(id_to_idx[i], id_to_idx[j], REL_SIBLING_OF)

    # same_fill_as
    fill_groups = defaultdict(list)
    for n in nodes:
        fill = node_fill(n)
        if fill:
            fill_groups[fill].append(n["id"])
    for ids in fill_groups.values():
        if len(ids) > 1:
            for i in ids:
                for j in ids:
                    if i != j:
                        add_edge(id_to_idx[i], id_to_idx[j], REL_SAME_FILL_AS)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    return x, edge_index, edge_type, id_to_idx, vision_scalar_x, vision_summaries
