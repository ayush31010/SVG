from __future__ import annotations

import copy
from typing import Any

from svgpatchlab.types import ModelRequest, ModelResponse

from .base import ModelAdapter


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
            from transformers import GenerationConfig, pipeline
        except ImportError as exc:
            raise RuntimeError("install svgpatchlab[hf] to use the Hugging Face adapter") from exc

        self.task = str(config.get("task", "text-generation"))
        self.supports_images = self.task == "image-text-to-text"
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
        default_generation = getattr(getattr(self.pipeline, "model", None), "generation_config", None)
        self.generation_config = (
            copy.deepcopy(default_generation) if default_generation is not None else GenerationConfig()
        )
        self.generation_config.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.generation_config.max_length = None
        self.generation_config.do_sample = bool(config.get("do_sample", False))
        if self.generation_config.do_sample:
            self.generation_config.temperature = float(config.get("temperature", 0.2))
        self.pipeline_kwargs = {"generation_config": self.generation_config}
        if self.task == "text-generation":
            self.pipeline_kwargs["clean_up_tokenization_spaces"] = False

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
        result = self.pipeline(model_input, **self.pipeline_kwargs)
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
        return ModelResponse(text=text, metadata={"adapter": "huggingface"})
