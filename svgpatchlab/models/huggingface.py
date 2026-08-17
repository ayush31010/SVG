from __future__ import annotations

from contextlib import contextmanager
import inspect
import threading
from typing import Any

from svgpatchlab.types import ModelRequest, ModelResponse

from .base import ModelAdapter


def _supports_adapter_disable(model: Any) -> bool:
    return callable(getattr(model, "disable_adapter", None)) or (
        callable(getattr(model, "disable_adapters", None))
        and callable(getattr(model, "enable_adapters", None))
    )


def _active_adapter_names(model: Any) -> tuple[str, ...]:
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        active = getattr(model, "active_adapter", None)
        if callable(active):
            active = active()
    if isinstance(active, str):
        return (active,) if active else ()
    if isinstance(active, (list, tuple, set, frozenset)):
        return tuple(str(name) for name in active if str(name))
    return ()


@contextmanager
def _adapter_disabled(model: Any):
    """Temporarily disable PEFT for both supported loading APIs.

    Direct PEFT models expose the singular context manager, while recent
    Transformers pipelines inject adapters into the base model and expose the
    plural enable/disable methods instead.
    """
    singular = getattr(model, "disable_adapter", None)
    if callable(singular):
        with singular():
            yield
        return
    disable = getattr(model, "disable_adapters", None)
    enable = getattr(model, "enable_adapters", None)
    if not callable(disable) or not callable(enable):
        raise RuntimeError("loaded model cannot temporarily disable its PEFT adapter")
    disable()
    try:
        yield
    finally:
        enable()


def _resolve_torch_dtype(torch_module: Any, value: Any) -> Any:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return value
    if value == "auto":
        return "auto"
    name = value.removeprefix("torch.")
    if hasattr(torch_module, name):
        return getattr(torch_module, name)
    raise ValueError(f"unknown torch dtype: {value}")


class HuggingFaceAdapter(ModelAdapter):
    """Lazy local Transformers pipeline adapter.

    The adapter is intentionally generic. Switch models by changing `model` and
    `task` in configuration rather than changing experiment code.
    """

    def __init__(self, config: dict[str, Any]):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("install svgpatchlab[hf] to use the Hugging Face adapter") from exc

        self.task = str(config.get("task", "text-generation"))
        self.supports_images = self.task == "image-text-to-text"
        adapter_response_schemas = config.get("adapter_response_schemas", [])
        if not isinstance(adapter_response_schemas, list) or not all(
            isinstance(name, str) and name for name in adapter_response_schemas
        ):
            raise ValueError("adapter_response_schemas must be a list of names")
        self.adapter_response_schemas = frozenset(adapter_response_schemas)
        schema_max_new_tokens = config.get("schema_max_new_tokens", {})
        if not isinstance(schema_max_new_tokens, dict) or not all(
            isinstance(name, str)
            and name
            and isinstance(limit, int)
            and not isinstance(limit, bool)
            and limit > 0
            for name, limit in schema_max_new_tokens.items()
        ):
            raise ValueError(
                "schema_max_new_tokens must map response schema names to "
                "positive integers"
            )
        self.schema_max_new_tokens = dict(schema_max_new_tokens)
        self._generation_lock = threading.RLock()
        kwargs: dict[str, Any] = {"model": str(config["model"])}
        for name in ("device", "device_map", "trust_remote_code"):
            if name in config:
                kwargs[name] = config[name]

        import torch
        model_kwargs: dict[str, Any] = {}

        quantization = config.get("quantization")
        if quantization in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("quantization requires bitsandbytes: pip install bitsandbytes") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(quantization == "4bit"),
                load_in_8bit=(quantization == "8bit"),
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs.setdefault("device_map", "auto")

        dtype = _resolve_torch_dtype(torch, config.get("dtype", config.get("torch_dtype")))
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        self.pipeline = pipeline(self.task, **kwargs)
        max_new_tokens = int(config.get("max_new_tokens", 512))
        do_sample = bool(config.get("do_sample", False))
        generation_controls: dict[str, Any] = {"do_sample": do_sample}
        if do_sample:
            generation_controls["temperature"] = float(
                config.get("temperature", 0.2)
            )
        self.pipeline_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
        }
        sanitize = getattr(self.pipeline, "_sanitize_parameters", None)
        if (
            self.task == "image-text-to-text"
            and callable(sanitize)
            and "generate_kwargs" in inspect.signature(sanitize).parameters
        ):
            # Transformers 5.x otherwise forwards generation controls into the
            # multimodal processor, where they are ignored.
            self.pipeline_kwargs["generate_kwargs"] = generation_controls
        else:
            self.pipeline_kwargs.update(generation_controls)
        if self.task == "text-generation":
            self.pipeline_kwargs["clean_up_tokenization_spaces"] = False

        if self.adapter_response_schemas:
            loaded_model = getattr(self.pipeline, "model", None)
            if not _supports_adapter_disable(loaded_model):
                raise RuntimeError(
                    "adapter_response_schemas requires a PEFT model with "
                    "temporary adapter-disable support"
                )
            if not _active_adapter_names(loaded_model):
                raise RuntimeError(
                    "adapter_response_schemas was configured, but the model "
                    "has no loaded active PEFT adapter"
                )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self.task == "image-text-to-text":
            content: list[dict[str, str]] = [
                *({"type": "image", "url": image} for image in request.images),
                {"type": "text", "text": request.prompt},
            ]
            model_input: Any = [{"role": "user", "content": content}]
        else:
            if request.images:
                raise RuntimeError(
                    "Hugging Face image inputs require task='image-text-to-text'"
                )
            model_input = request.prompt
        adapter_active = (
            not self.adapter_response_schemas
            or request.response_schema_name in self.adapter_response_schemas
        )
        pipeline_kwargs = dict(self.pipeline_kwargs)
        if "generate_kwargs" in pipeline_kwargs:
            pipeline_kwargs["generate_kwargs"] = dict(
                pipeline_kwargs["generate_kwargs"]
            )
        token_limit = self.schema_max_new_tokens.get(request.response_schema_name)
        if token_limit is not None:
            pipeline_kwargs["max_new_tokens"] = token_limit
        # PEFT activation is mutable model-wide state. Serialize generation so
        # a concurrent rerank cannot overlap an adapter-disabled patch call.
        with self._generation_lock:
            if adapter_active:
                result = self.pipeline(model_input, **pipeline_kwargs)
            else:
                # A node-grounding adapter should shape only the closed-choice
                # request. Disabling it for patch generation preserves the base
                # model's general instruction following without loading a
                # second 7B copy.
                with _adapter_disabled(self.pipeline.model):
                    result = self.pipeline(model_input, **pipeline_kwargs)
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            assistant_messages = [
                message for message in generated if message.get("role") == "assistant"
            ]
            text = assistant_messages[-1]["content"] if assistant_messages else str(generated[-1])
        else:
            text = str(generated)
            if self.task == "text-generation" and text.startswith(request.prompt):
                text = text[len(request.prompt) :]
        return ModelResponse(
            text=text,
            metadata={
                "adapter": "huggingface",
                "peft_adapter_active": adapter_active,
            },
        )
