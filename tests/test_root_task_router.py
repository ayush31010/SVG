from __future__ import annotations

import unittest
from pathlib import Path

from svgpatchlab.architectures.factory import create_architecture
from svgpatchlab.architectures.root_tasks import compile_root_task_patch
from svgpatchlab.core import build_scene
from svgpatchlab.models.base import ModelAdapter
from svgpatchlab.types import BenchmarkCase, ModelRequest, ModelResponse


SOURCE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="10 20 80 40">'
    '<rect x="10" y="20" width="80" height="40" fill="#f00"/>'
    "</svg>"
)


def _case(task: str, instruction: str) -> BenchmarkCase:
    return BenchmarkCase(
        task=task,
        emoji_id="root-route",
        instruction=instruction,
        source_svg=SOURCE,
        answer_svg=SOURCE,
        query_path=Path("."),
        answer_path=Path("."),
    )


class _FailIfCalledModel(ModelAdapter):
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("root-routed cases must not call the model")


class RootTaskCompilerTests(unittest.TestCase):
    def test_transparency_by_half(self):
        patch = compile_root_task_patch(
            _case("transparency", "Make this image transparent by half."),
            build_scene(SOURCE),
        )
        self.assertEqual(
            patch.operations[0].attributes_dict,
            {"opacity": "0.5"},
        )

    def test_crop_supports_all_keep_directions(self):
        expected = {
            "left": "10 20 40 40",
            "right": "50 20 40 40",
            "top": "10 20 80 20",
            "bottom": "10 40 80 20",
        }
        for direction, viewbox in expected.items():
            with self.subTest(direction=direction):
                patch = compile_root_task_patch(
                    _case(
                        "crop_to_half",
                        f"Trim the other half and keep the {direction} half.",
                    ),
                    build_scene(SOURCE),
                )
                self.assertEqual(
                    patch.operations[0].attributes_dict,
                    {"viewBox": viewbox},
                )

    def test_upside_down_respects_viewbox_offset(self):
        patch = compile_root_task_patch(
            _case("upside_down", "Flip this image upside down."),
            build_scene(SOURCE),
        )
        self.assertEqual(
            patch.operations[0].attributes_dict,
            {"transform": "translate(0,80) scale(1,-1)"},
        )

    def test_ambiguous_instruction_falls_back(self):
        patch = compile_root_task_patch(
            _case("crop_to_half", "Crop this image."),
            build_scene(SOURCE),
        )
        self.assertIsNone(patch)


class RoutedArchitectureTests(unittest.TestCase):
    def test_routed_strict_architecture_bypasses_model(self):
        architecture = create_architecture("routed_strict_skeleton_patch")
        result = architecture.run(
            _case("transparency", "Make this image transparent by half."),
            _FailIfCalledModel(),
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.model_calls, 0)
        self.assertTrue(result.details["root_task_router"]["routed"])
        self.assertIn('opacity="0.5"', result.output_svg)

    def test_semantic_architecture_bypasses_renderer_and_model(self):
        architecture = create_architecture("semantic_id_patch")
        result = architecture.run(
            _case("crop_to_half", "Keep the right half."),
            _FailIfCalledModel(),
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.model_calls, 0)
        self.assertIn('viewBox="50 20 40 40"', result.output_svg)


if __name__ == "__main__":
    unittest.main()
