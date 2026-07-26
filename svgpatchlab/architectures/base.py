from __future__ import annotations

from abc import ABC, abstractmethod

from svgpatchlab.models import ModelAdapter
from svgpatchlab.types import ArchitectureResult, BenchmarkCase


class Architecture(ABC):
    name: str
    requires_model: bool = True
    requires_renderer: bool = False

    @abstractmethod
    def run(self, case: BenchmarkCase, model: ModelAdapter) -> ArchitectureResult:
        raise NotImplementedError
