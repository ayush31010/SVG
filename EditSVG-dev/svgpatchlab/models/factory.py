from __future__ import annotations

from typing import Any

from .base import ModelAdapter, ModelNotConfigured
from .openai_compatible import OpenAICompatibleAdapter
from .replay import ReplayAdapter


def create_model(config: dict[str, Any] | None) -> ModelAdapter:
    if not config or config.get("adapter", "none") == "none":
        return ModelNotConfigured()
    adapter = config.get("adapter")
    if adapter == "openai_compatible":
        return OpenAICompatibleAdapter(config)
    if adapter == "replay":
        return ReplayAdapter(config["path"])
    raise ValueError(f"unknown model adapter: {adapter}")
