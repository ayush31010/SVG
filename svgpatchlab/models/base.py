from __future__ import annotations

from abc import ABC, abstractmethod
import time

from svgpatchlab.types import ModelRequest, ModelResponse


class ModelAdapter(ABC):
    """The only interface experiment code uses to call a model."""

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError


class ModelNotConfigured(ModelAdapter):
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise RuntimeError("this architecture requires a configured model adapter")


class RecordingModelAdapter(ModelAdapter):
    """Collect model-independent latency and usage records around any adapter."""

    def __init__(self, wrapped: ModelAdapter):
        self.wrapped = wrapped
        self.records: list[dict] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.perf_counter()
        try:
            response = self.wrapped.generate(request)
        except Exception:
            self.records.append(
                {
                    "request_id": request.metadata.get("request_id"),
                    "latency_seconds": time.perf_counter() - started,
                    "ok": False,
                }
            )
            raise
        self.records.append(
            {
                "request_id": request.metadata.get("request_id"),
                "latency_seconds": time.perf_counter() - started,
                "ok": True,
                "metadata": response.metadata,
            }
        )
        return response
