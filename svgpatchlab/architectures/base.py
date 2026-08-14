from __future__ import annotations

from abc import ABC, abstractmethod

from svgpatchlab.models import ModelAdapter
from svgpatchlab.types import ArchitectureResult, BenchmarkCase


class Architecture(ABC):
    name: str
    requires_model: bool = True

    @abstractmethod
    def run(
        self,
        case: BenchmarkCase,
        model: ModelAdapter,
        vision_model: ModelAdapter | None = None,
    ) -> ArchitectureResult:
        raise NotImplementedError
