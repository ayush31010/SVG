from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from svgpatchlab.config import load_model_config
from svgpatchlab.models.base import RecordingModelAdapter
from svgpatchlab.models.huggingface import HuggingFaceAdapter
from svgpatchlab.models.openai_compatible import OpenAICompatibleAdapter
from svgpatchlab.types import ModelRequest


class FakeGenerationConfig:
    def __init__(self):
        self.max_length = 20


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePipeline:
    def __init__(self):
        self.model = SimpleNamespace(generation_config=FakeGenerationConfig())
        self.calls = []

    def __call__(self, model_input, **kwargs):
        self.calls.append((model_input, kwargs))
        return [{"generated_text": f"{model_input}patched"}]


class HuggingFaceAdapterTests(unittest.TestCase):
    def test_default_context_v3_config_builds_image_peft_pipeline(self):
        captured = {}

        class FakePeftModel:
            def __init__(self):
                self.disabled = False

            def disable_adapters(self):
                self.disabled = True

            def enable_adapters(self):
                self.disabled = False

            def active_adapters(self):
                return ["default"]

        class FakeImagePipeline:
            def __init__(self):
                self.model = FakePeftModel()

            def _sanitize_parameters(
                self, max_new_tokens=None, generate_kwargs=None
            ):
                return {}, {}, {}

        def fake_pipeline(task, **kwargs):
            captured["task"] = task
            captured["kwargs"] = kwargs
            return FakeImagePipeline()

        root = Path(__file__).resolve().parents[1]
        config = load_model_config(
            root / "configs" / "models" / "qwen2.5vl-7b-huggingface-grounded.json"
        )
        fake_transformers = SimpleNamespace(pipeline=fake_pipeline)
        fake_torch = SimpleNamespace(bfloat16="bfloat16")
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            adapter = HuggingFaceAdapter(config)

        self.assertTrue(adapter.supports_images)
        self.assertEqual(captured["task"], "image-text-to-text")
        self.assertTrue(
            captured["kwargs"]["model"]
            .replace("\\", "/")
            .endswith("node-grounding-qwen2.5-vl-7b-context-v3/adapter")
        )
        self.assertEqual(captured["kwargs"]["device_map"], "auto")
        self.assertEqual(
            captured["kwargs"]["model_kwargs"]["dtype"], "bfloat16"
        )
        self.assertEqual(
            adapter.adapter_response_schemas,
            frozenset({"svgpatchlab_candidate_rerank_v1"}),
        )

    def test_peft_adapter_can_be_scoped_to_candidate_rerank_schema(self):
        states = []
        token_limits = []

        class FakePeftModel:
            def __init__(self):
                self.generation_config = FakeGenerationConfig()
                self.disabled = False

            def disable_adapters(self):
                self.disabled = True

            def enable_adapters(self):
                self.disabled = False

            def active_adapters(self):
                return ["default"]

        class ScopedPipeline:
            def __init__(self):
                self.model = FakePeftModel()

            def __call__(self, model_input, **kwargs):
                del model_input
                states.append(self.model.disabled)
                token_limits.append(kwargs.get("max_new_tokens"))
                return [{"generated_text": "ok"}]

        fake_transformers = SimpleNamespace(
            GenerationConfig=FakeGenerationConfig,
            pipeline=lambda task, **kwargs: ScopedPipeline(),
        )
        with patch.dict(
            sys.modules,
            {"torch": SimpleNamespace(), "transformers": fake_transformers},
        ):
            adapter = HuggingFaceAdapter(
                {
                    "model": "example/adapter",
                    "task": "text-generation",
                    "adapter_response_schemas": [
                        "svgpatchlab_candidate_rerank_v1"
                    ],
                    "schema_max_new_tokens": {
                        "svgpatchlab_candidate_rerank_v1": 32
                    },
                    "max_new_tokens": 512,
                }
            )

        selected = adapter.generate(
            ModelRequest(
                "select",
                response_schema_name="svgpatchlab_candidate_rerank_v1",
            )
        )
        patched = adapter.generate(
            ModelRequest("patch", response_schema_name="svgpatchlab_patch_v2")
        )
        selected_again = adapter.generate(
            ModelRequest(
                "select again",
                response_schema_name="svgpatchlab_candidate_rerank_v1",
            )
        )

        self.assertEqual(states, [False, True, False])
        self.assertEqual(token_limits, [32, 512, 32])
        self.assertTrue(selected.metadata["peft_adapter_active"])
        self.assertFalse(patched.metadata["peft_adapter_active"])
        self.assertTrue(selected_again.metadata["peft_adapter_active"])

    def test_scoped_adapter_rejects_a_model_without_loaded_adapter(self):
        fake_transformers = SimpleNamespace(
            GenerationConfig=FakeGenerationConfig,
            pipeline=lambda task, **kwargs: FakePipeline(),
        )
        with patch.dict(
            sys.modules,
            {"torch": SimpleNamespace(), "transformers": fake_transformers},
        ):
            with self.assertRaisesRegex(RuntimeError, "PEFT model"):
                HuggingFaceAdapter(
                    {
                        "model": "example/base-model",
                        "adapter_response_schemas": ["selection"],
                    }
                )

    def test_legacy_torch_dtype_config_is_sent_as_dtype(self):
        captured = {}

        def fake_pipeline(task, **kwargs):
            captured["task"] = task
            captured["kwargs"] = kwargs
            captured["pipeline"] = FakePipeline()
            return captured["pipeline"]

        fake_transformers = SimpleNamespace(
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
            GenerationConfig=FakeGenerationConfig,
            pipeline=fake_pipeline,
        )
        fake_torch = SimpleNamespace(float16="float16", bfloat16="bfloat16")

        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "transformers": fake_transformers,
            },
        ):
            adapter = HuggingFaceAdapter(
                {
                    "model": "example/model",
                    "task": "text-generation",
                    "torch_dtype": "auto",
                    "quantization": "4bit",
                    "max_new_tokens": 128,
                    "do_sample": True,
                    "temperature": 0.4,
                }
            )

        kwargs = captured["kwargs"]
        self.assertNotIn("torch_dtype", kwargs)
        self.assertEqual(kwargs["device_map"], "auto")
        self.assertEqual(kwargs["model_kwargs"]["dtype"], "auto")
        self.assertTrue(kwargs["model_kwargs"]["quantization_config"].kwargs["load_in_4bit"])

        self.assertEqual(adapter.pipeline_kwargs["max_new_tokens"], 128)
        self.assertTrue(adapter.pipeline_kwargs["do_sample"])
        self.assertEqual(adapter.pipeline_kwargs["temperature"], 0.4)

        response = adapter.generate(ModelRequest("prompt "))
        self.assertEqual(response.text, "patched")
        _, call_kwargs = captured["pipeline"].calls[-1]
        self.assertEqual(call_kwargs["max_new_tokens"], 128)
        self.assertTrue(call_kwargs["do_sample"])
        self.assertFalse(call_kwargs["clean_up_tokenization_spaces"])

    def test_text_generation_rejects_images_instead_of_silently_ignoring_them(self):
        captured = {}

        def fake_pipeline(task, **kwargs):
            captured["pipeline"] = FakePipeline()
            return captured["pipeline"]

        fake_transformers = SimpleNamespace(
            GenerationConfig=FakeGenerationConfig,
            pipeline=fake_pipeline,
        )
        fake_torch = SimpleNamespace()

        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "transformers": fake_transformers,
            },
        ):
            adapter = HuggingFaceAdapter(
                {
                    "model": "example/model",
                    "task": "text-generation",
                }
            )

        self.assertFalse(adapter.supports_images)
        with self.assertRaisesRegex(RuntimeError, "image-text-to-text"):
            adapter.generate(ModelRequest("prompt", images=("data:image/png;base64,AA==",)))

    def test_image_text_pipeline_advertises_image_support(self):
        class FakeImagePipeline(FakePipeline):
            def _sanitize_parameters(
                self, max_new_tokens=None, generate_kwargs=None
            ):
                return {}, {}, {}

        def fake_pipeline(task, **kwargs):
            return FakeImagePipeline()

        fake_transformers = SimpleNamespace(
            GenerationConfig=FakeGenerationConfig,
            pipeline=fake_pipeline,
        )
        fake_torch = SimpleNamespace()

        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "transformers": fake_transformers,
            },
        ):
            adapter = HuggingFaceAdapter(
                {
                    "model": "example/vision-model",
                    "task": "image-text-to-text",
                }
            )

        self.assertTrue(adapter.supports_images)
        self.assertTrue(RecordingModelAdapter(adapter).supports_images)
        self.assertEqual(adapter.pipeline_kwargs["max_new_tokens"], 512)
        self.assertEqual(
            adapter.pipeline_kwargs["generate_kwargs"], {"do_sample": False}
        )

    def test_openai_compatible_image_capability_tracks_endpoint_and_override(self):
        chat = OpenAICompatibleAdapter(
            {"model": "example", "endpoint": "chat_completions"}
        )
        completion = OpenAICompatibleAdapter(
            {"model": "example", "endpoint": "completions"}
        )
        disabled = OpenAICompatibleAdapter(
            {
                "model": "example",
                "endpoint": "chat_completions",
                "supports_images": False,
            }
        )

        self.assertTrue(chat.supports_images)
        self.assertFalse(completion.supports_images)
        self.assertFalse(disabled.supports_images)


if __name__ == "__main__":
    unittest.main()
