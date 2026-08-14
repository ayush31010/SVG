from __future__ import annotations

from .base import Architecture
from .patching import SkeletonPatchArchitecture


ARCHITECTURES: dict[str, type[Architecture]] = {
    "skeleton_patch": SkeletonPatchArchitecture,
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
    return architecture_class()
