from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    task: str
    emoji_id: str
    instruction: str
    source_svg: str
    answer_svg: str
    query_path: Path
    answer_path: Path

    @property
    def case_id(self) -> str:
        return f"{self.task}/{self.emoji_id}"


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    images: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    #: JSON Schema the response must satisfy. Adapters configured for
    #: structured output send it to the server so decoding is constrained;
    #: adapters without that support ignore it and the parser catches drift.
    response_schema: dict[str, Any] | None = None
    response_schema_name: str = "response"


@dataclass(frozen=True)
class ModelResponse:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureResult:
    output_svg: str | None = None
    patch: Any | None = None
    raw_responses: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    model_calls: int = 0
