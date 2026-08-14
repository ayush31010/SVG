from __future__ import annotations

import json
from pathlib import Path

from svgpatchlab.types import ModelRequest, ModelResponse

from .base import ModelAdapter


class ReplayAdapter(ModelAdapter):
    """Replay saved responses for deterministic tests and offline evaluation."""

    def __init__(self, path: str | Path):
        self.responses: dict[str, str] = {}
        with Path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                self.responses[str(item["request_id"])] = str(item["response"])

    def generate(self, request: ModelRequest) -> ModelResponse:
        request_id = str(request.metadata.get("request_id", ""))
        if request_id not in self.responses:
            raise KeyError(f"no replay response for {request_id}")
        return ModelResponse(self.responses[request_id], {"adapter": "replay"})

