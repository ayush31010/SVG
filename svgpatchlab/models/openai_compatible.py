from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from svgpatchlab.types import ModelRequest, ModelResponse

from .base import ModelAdapter


class PromptTruncatedError(RuntimeError):
    """The server reported far fewer prompt tokens than the prompt can contain.

    Raised when a request almost certainly overflowed the server's context
    window and was silently cut off. Architectures deliberately let this
    propagate instead of recording it as a model mistake: a truncated prompt is
    a misconfigured run, not a wrong answer, and scoring it as one is how an
    earlier 500-case matrix reported a result it had not measured.
    """


#: Intact prompts in the reference run tokenized at 2.78 to 3.66 characters per
#: token; truncated ones reported 5.76 to 6.18. The default sits between, with
#: roughly 20% margin either side. Raise it for a tokenizer with larger tokens.
DEFAULT_MAX_CHARS_PER_PROMPT_TOKEN = 4.5


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for vLLM, SGLang, llama.cpp, and compatible HTTP servers."""

    def __init__(self, config: dict[str, Any]):
        self.base_url = str(config.get("base_url", "http://localhost:8000/v1")).rstrip("/")
        self.model = str(config["model"])
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 512))
        self.timeout = float(config.get("timeout", 120))
        self.json_mode = bool(config.get("json_mode", False))
        # "response_format" sends an OpenAI-style json_schema block (Ollama,
        # recent vLLM, SGLang). "guided_json" sends vLLM's older field. "off"
        # ignores any schema on the request.
        self.structured_output = str(config.get("structured_output", "off"))
        if self.structured_output not in {"off", "response_format", "guided_json"}:
            raise ValueError(
                f"unknown structured_output mode: {self.structured_output}"
            )
        self.on_truncated_prompt = str(config.get("on_truncated_prompt", "error"))
        if self.on_truncated_prompt not in {"error", "warn", "ignore"}:
            raise ValueError(
                f"unknown on_truncated_prompt mode: {self.on_truncated_prompt}"
            )
        self.max_chars_per_prompt_token = float(
            config.get(
                "max_chars_per_prompt_token", DEFAULT_MAX_CHARS_PER_PROMPT_TOKEN
            )
        )
        self.top_p = config.get("top_p")
        self.endpoint = str(config.get("endpoint", "chat_completions"))
        if self.endpoint not in {"chat_completions", "completions"}:
            raise ValueError(f"unknown OpenAI-compatible endpoint: {self.endpoint}")
        self.supports_images = bool(
            config.get("supports_images", self.endpoint == "chat_completions")
        ) and self.endpoint == "chat_completions"
        self.stop = config.get("stop")
        self.extra_body = dict(config.get("extra_body", {}))
        api_key = config.get("api_key")
        api_key_env = config.get("api_key_env")
        self.api_key = str(api_key or (os.getenv(str(api_key_env)) if api_key_env else "") or "")

    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.images and not self.supports_images:
            raise RuntimeError("this OpenAI-compatible model configuration does not support images")
        if self.endpoint == "completions":
            return self._generate_completion(request)
        return self._generate_chat_completion(request)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _apply_schema(self, payload: dict[str, Any], request: ModelRequest) -> None:
        """Ask the server to constrain decoding to the request's schema."""
        if self.structured_output == "off" or request.response_schema is None:
            return
        if self.structured_output == "guided_json":
            payload["guided_json"] = request.response_schema
            return
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.response_schema_name,
                "strict": True,
                "schema": request.response_schema,
            },
        }

    def _check_prompt_intact(
        self, prompt: str, usage: dict[str, Any], request_id: str
    ) -> bool:
        """Detect a prompt the server silently truncated. Returns True if truncated.

        Servers do not report truncation, so the only available signal is that
        the reported prompt-token count is implausibly small for the text sent.
        """
        if self.on_truncated_prompt == "ignore" or not prompt:
            return False
        reported = usage.get("prompt_tokens")
        if reported is None:
            return False
        ratio = len(prompt) / reported if reported else float("inf")
        if ratio <= self.max_chars_per_prompt_token:
            return False
        detail = (
            f"{request_id or 'request'}: server reported {reported} prompt tokens "
            f"for {len(prompt)} characters ({ratio:.1f} chars/token, limit "
            f"{self.max_chars_per_prompt_token}). The prompt was almost certainly "
            f"truncated at the context window, so the model never saw all of it. "
            f"Raise the server's context length, shorten the scene, or set "
            f"on_truncated_prompt to 'warn' to record and continue."
        )
        if self.on_truncated_prompt == "error":
            raise PromptTruncatedError(detail)
        print(f"WARNING: prompt truncated -- {detail}", file=sys.stderr)
        return True

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
        self._apply_schema(payload, request)
        result = self._read_json("completions", payload)
        choice = result["choices"][0]
        usage = result.get("usage", {})
        truncated = self._check_prompt_intact(
            request.prompt, usage, str(request.metadata.get("request_id", ""))
        )
        return ModelResponse(
            text=choice.get("text", ""),
            metadata={
                "usage": usage,
                "model": result.get("model"),
                "prompt_truncated": truncated,
            },
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
        self._apply_schema(payload, request)
        result = self._read_json("chat/completions", payload)
        choice = result["choices"][0]
        usage = result.get("usage", {})
        truncated = self._check_prompt_intact(
            request.prompt, usage, str(request.metadata.get("request_id", ""))
        )
        return ModelResponse(
            text=choice["message"]["content"],
            metadata={
                "usage": usage,
                "model": result.get("model"),
                "prompt_truncated": truncated,
            },
        )
