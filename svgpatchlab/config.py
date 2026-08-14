from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open() as handle:
        config = json.load(handle)
    if "dataset" not in config or "architecture" not in config:
        raise ValueError("config requires dataset and architecture sections")
    model = config.get("model")
    if isinstance(model, dict) and "config_file" in model:
        config["model"] = _load_model_reference(config_path, model)

    vision_context = config.get("architecture", {}).get("vision_context")
    if isinstance(vision_context, dict):
        vision_model = vision_context.get("model")
        if isinstance(vision_model, dict) and "config_file" in vision_model:
            vision_context["model"] = _load_model_reference(config_path, vision_model)
    config["_config_path"] = str(config_path.resolve())
    return config


def load_model_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def _load_model_reference(config_path: Path, model: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(model["config_file"])
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    with model_path.open() as handle:
        loaded_model = json.load(handle)
    loaded_model.update({key: value for key, value in model.items() if key != "config_file"})
    return loaded_model
