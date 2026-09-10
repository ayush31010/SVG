from __future__ import annotations

from .base import Architecture
from .patching import SkeletonPatchArchitecture
from .gnn_patching import GnnPatchArchitecture


ARCHITECTURES: dict[str, type[Architecture]] = {
    "skeleton_patch": SkeletonPatchArchitecture,
    "gnn_patch": GnnPatchArchitecture,
}


def create_architecture(name: str, config: dict | None = None) -> Architecture:
    try:
        architecture_class = ARCHITECTURES[name]
    except KeyError as exc:
        raise ValueError(f"unknown architecture: {name}") from exc
    config = config or {}
    if architecture_class is SkeletonPatchArchitecture:
        return architecture_class(
            vision_context_config=config.get("vision_context"),
        )
    if architecture_class is GnnPatchArchitecture:
        return architecture_class(
            vision_context_config=config.get("vision_context"),
            gnn_model_path=config.get("gnn_model_path", "gnn_node_targeting.pt"),
            use_attention=config.get("use_attention", True),
        )
    return architecture_class()
