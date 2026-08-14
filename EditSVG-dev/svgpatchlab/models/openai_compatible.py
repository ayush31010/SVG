from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from svgpatchlab.types import ModelRequest, ModelResponse

from .base import ModelAdapter


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for vLLM, SGLang, llama.cpp, and compatible HTTP servers."""

    def __init__(self, config: dict[str, Any]):
        self.base_url = str(config.get("base_url", "http://localhost:8000/v1")).rstrip("/")
        self.model = str(config["model"])
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 512))
        self.timeout = float(config.get("timeout", 120))
        self.json_mode = bool(config.get("json_mode", False))
        self.top_p = config.get("top_p")
        self.endpoint = str(config.get("endpoint", "chat_completions"))
        if self.endpoint not in {"chat_completions", "completions"}:
            raise ValueError(f"unknown OpenAI-compatible endpoint: {self.endpoint}")
        self.stop = config.get("stop")
        self.extra_body = dict(config.get("extra_body", {}))
        api_key = config.get("api_key")
        api_key_env = config.get("api_key_env")
        self.api_key = str(api_key or (os.getenv(str(api_key_env)) if api_key_env else "") or "")

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self.endpoint == "completions":
            return self._generate_completion(request)
        return self._generate_chat_completion(request)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _read_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload.update(self.extra_body)
        if self.stop is not None:
            payload["stop"] = self.stop
        body = json.dumps(payload).encode()
        http_request = urllib.request.Request(
            f"{self.base_url}/{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            details = exc.read().decode(errors="replace")
            raise RuntimeError(f"model server returned HTTP {exc.code}: {details}") from exc

    def _generate_completion(self, request: ModelRequest) -> ModelResponse:
        if request.images:
            raise RuntimeError("raw completions endpoint does not support image inputs")
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        result = self._read_json("completions", payload)
        choice = result["choices"][0]
        return ModelResponse(
            text=choice.get("text", ""),
            metadata={"usage": result.get("usage", {}), "model": result.get("model")},
        )

    def _generate_chat_completion(self, request: ModelRequest) -> ModelResponse:
        if request.images:
            content: str | list[dict[str, Any]] = [
                {"type": "text", "text": request.prompt},
                *(
                    {
                        "type": "image_url",
                        "image_url": {"url": image},
                    }
                    for image in request.images
                ),
            ]
        else:
            content = request.prompt
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        result = self._read_json("chat/completions", payload)
        choice = result["choices"][0]
        return ModelResponse(
            text=choice["message"]["content"],
            metadata={"usage": result.get("usage", {}), "model": result.get("model")},
        )
