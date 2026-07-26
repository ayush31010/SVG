from __future__ import annotations

import io
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from svgpatchlab.architectures.semantic import (
    SemanticIdPatchArchitecture,
    _completion_reconstruction_candidates,
    _completion_triggered,
    _counterfactual_difference,
    _snap_generated_geometry,
    _selected_targets,
)
from svgpatchlab.core.xml import parse_svg
from svgpatchlab.architectures.factory import create_architecture
from svgpatchlab.core import build_scene
from svgpatchlab.eval.render import SVGIDMap, SVGVisualContext
from svgpatchlab.models.base import ModelAdapter
from svgpatchlab.types import BenchmarkCase, ModelRequest, ModelResponse


SOURCE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
    "<g>"
    '<rect x="0" y="0" width="10" height="10" fill="#ff0000"/>'
    '<circle cx="15" cy="15" r="4" fill="#0000ff"/>'
    "</g>"
    "</svg>"
)

ANSWER_WITHOUT_RECT = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">'
    "<g>"
    '<circle cx="15" cy="15" r="4" fill="#0000ff"/>'
    "</g>"
    "</svg>"
)


def _png(color: tuple[int, int, int, int]) -> bytes:
    image = Image.new("RGBA", (8, 8), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _visual_context() -> SVGVisualContext:
    normal_png = _png((255, 0, 0, 255))
    id_png = _png((0x3C, 0x6E, 0xF2, 255))
    normal_rgba = np.asarray(
        Image.open(io.BytesIO(normal_png)).convert("RGBA"), dtype=np.uint8
    )
    id_rgba = np.asarray(
        Image.open(io.BytesIO(id_png)).convert("RGBA"), dtype=np.uint8
    )
    labels = np.zeros((8, 8), dtype=np.int32)
    id_map = SVGIDMap(
        png=id_png,
        rgb=id_rgba[:, :, :3],
        labels=labels,
        node_colors={"n2": "#3c6ef2", "n3": "#daa66b"},
        color_nodes={"#3c6ef2": "n2", "#daa66b": "n3"},
        leaf_node_ids=("n2", "n3"),
        size=8,
    )
    return SVGVisualContext(
        normal_png=normal_png,
        normal_rgba=normal_rgba,
        id_map=id_map,
        stats={
            "n0": {
                "bbox": [0, 0, 20, 20],
                "area_pct": 100,
                "position": "center",
                "color": "#ff0000",
            },
            "n1": {
                "bbox": [0, 0, 20, 20],
                "area_pct": 100,
                "position": "center",
                "color": "#ff0000",
            },
            "n2": {
                "bbox": [0, 0, 10, 10],
                "area_pct": 25,
                "position": "top-left",
                "color": "#ff0000",
                "id_color": "#3c6ef2",
            },
            "n3": {
                "bbox": [11, 11, 8, 8],
                "area_pct": 12.5,
                "position": "bottom-right",
                "color": "#0000ff",
                "id_color": "#daa66b",
            },
        },
        method="id_buffer",
    )


def _case(task: str = "delete") -> BenchmarkCase:
    return BenchmarkCase(
        task=task,
        emoji_id="synthetic",
        instruction="Remove the red square.",
        source_svg=SOURCE_SVG,
        answer_svg=ANSWER_WITHOUT_RECT,
        query_path=Path("."),
        answer_path=Path("."),
    )


class SequenceModel(ModelAdapter):
    def __init__(self, responses: list[str], *, supports_images: bool):
        self.responses = list(responses)
        self.supports_images = supports_images
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(self.responses.pop(0))


class TargetSelectionTests(unittest.TestCase):
    def test_deduplicates_caps_and_keeps_first_ranked_related_node(self):
        scene = build_scene(SOURCE_SVG)
        targets = _selected_targets(
            '{"targets":["n2","n2","n1","n3"]}',
            scene,
            2,
            forbid_root=True,
        )
        self.assertEqual(targets, ("n2", "n3"))

    def test_unknown_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown IDs"):
            _selected_targets(
                '{"targets":["n99"]}',
                build_scene(SOURCE_SVG),
                3,
                forbid_root=True,
            )

    def test_delete_cannot_select_root(self):
        with self.assertRaisesRegex(ValueError, "SVG root"):
            _selected_targets(
                '{"targets":["n0"]}',
                build_scene(SOURCE_SVG),
                3,
                forbid_root=True,
            )


class SemanticArchitectureTests(unittest.TestCase):
    def test_architecture_is_registered_with_configurable_limits(self):
        architecture = create_architecture(
            "semantic_id_patch",
            render_size=96,
            max_candidates=2,
        )
        self.assertIsInstance(architecture, SemanticIdPatchArchitecture)
        self.assertEqual(architecture.render_size, 96)
        self.assertEqual(architecture.max_candidates, 2)

    def _run(self, model: SequenceModel):
        architecture = SemanticIdPatchArchitecture(render_size=64, max_candidates=3)
        preview_png = _png((0, 0, 255, 255))
        with mock.patch.object(
            architecture,
            "_visual_context",
            return_value=_visual_context(),
        ), mock.patch(
            "svgpatchlab.architectures.semantic.render_svg_png",
            return_value=preview_png,
        ) as render_preview:
            result = architecture.run(_case(), model)
        return result, render_preview

    def test_multimodal_flow_uses_id_pair_then_only_shortlisted_preview(self):
        model = SequenceModel(
            [
                '{"targets":["n2"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n2"]}]}',
            ],
            supports_images=True,
        )
        result, render_preview = self._run(model)

        self.assertIsNone(result.error)
        self.assertIsNotNone(result.output_svg)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(render_preview.call_count, 1)
        hidden_svg = render_preview.call_args.args[0]
        self.assertIn("display:none", hidden_svg)

        self.assertEqual(len(model.requests), 2)
        self.assertEqual(
            model.requests[0].response_schema_name,
            "svgpatchlab_target_selection_v1",
        )
        self.assertEqual(
            model.requests[0].response_schema["properties"]["targets"]["maxItems"],
            3,
        )
        self.assertEqual(
            model.requests[1].response_schema_name,
            "svgpatchlab_delete_patch_v1",
        )
        operation_schema = model.requests[1].response_schema["properties"][
            "operations"
        ]["items"]
        self.assertEqual(
            operation_schema["properties"]["op"]["enum"],
            ["remove_element"],
        )
        self.assertEqual(len(model.requests[0].images), 2)
        self.assertEqual(len(model.requests[1].images), 3)
        self.assertIn('"id_color": "#3c6ef2"', model.requests[0].prompt)
        self.assertIn('"shortlisted_targets": [', model.requests[1].prompt)
        self.assertIn('"target": "n2"', model.requests[1].prompt)
        self.assertIn('"kind": "without_candidate"', model.requests[1].prompt)
        self.assertIn('"kind": "difference_map"', model.requests[1].prompt)
        self.assertEqual(result.details["selected_targets"], ["n2"])
        difference = result.details["counterfactual_previews"][0]["difference"]
        self.assertIn("dominant_color_transitions", difference)
        self.assertIn("components", difference)
        self.assertIn("ownership", difference)

    def test_text_only_model_gets_numeric_visual_context_without_images(self):
        model = SequenceModel(
            [
                '{"targets":["n2"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n2"]}]}',
            ],
            supports_images=False,
        )
        result, _ = self._run(model)

        self.assertIsNone(result.error)
        self.assertEqual(model.requests[0].images, ())
        self.assertEqual(model.requests[1].images, ())
        self.assertIn("No images are attached", model.requests[0].prompt)
        self.assertIn("changed_area_pct", model.requests[1].prompt)
        self.assertIn("transparent_hole_area_pct", model.requests[1].prompt)

    def test_complex_svg_still_gives_vision_model_the_normal_render(self):
        model = SequenceModel(
            [
                '{"targets":["n2"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n2"]}]}',
            ],
            supports_images=True,
        )
        architecture = SemanticIdPatchArchitecture(render_size=64)
        fallback = replace(
            _visual_context(),
            id_map=None,
            method="counterfactual",
            unsupported_features=("n2:<rect> filter",),
        )
        with mock.patch.object(
            architecture,
            "_visual_context",
            return_value=fallback,
        ), mock.patch(
            "svgpatchlab.architectures.semantic.render_svg_png",
            return_value=_png((0, 0, 255, 255)),
        ):
            result = architecture.run(_case(), model)

        self.assertIsNone(result.error)
        self.assertEqual(len(model.requests[0].images), 1)
        self.assertIn("ID image could not be produced", model.requests[0].prompt)

    def test_final_patch_cannot_escape_shortlist(self):
        model = SequenceModel(
            [
                '{"targets":["n2"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n3"]}]}',
            ],
            supports_images=False,
        )
        result, _ = self._run(model)

        self.assertIsNone(result.output_svg)
        self.assertIn("not shortlisted: n3", result.error)
        self.assertEqual(result.model_calls, 2)

    def test_bad_stage_one_stops_before_preview_and_second_call(self):
        model = SequenceModel(
            ['{"targets":["n99"]}'],
            supports_images=False,
        )
        result, render_preview = self._run(model)

        self.assertIsNone(result.output_svg)
        self.assertIn("unknown IDs", result.error)
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(render_preview.call_count, 0)
        self.assertEqual(len(model.requests), 1)

    def test_delete_patch_version_is_repaired(self):
        model = SequenceModel(
            [
                '{"targets":["n2"]}',
                '{"version":1,"operations":[{"op":"remove_element","targets":["n2"]}]}',
            ],
            supports_images=False,
        )
        result, _ = self._run(model)

        self.assertIsNone(result.error)
        self.assertEqual(result.patch.version, 2)


class CounterfactualDifferenceTests(unittest.TestCase):
    def test_captures_removed_revealed_and_spatial_context_without_extra_render(self):
        original = np.zeros((8, 8, 4), dtype=np.uint8)
        hidden = np.zeros((8, 8, 4), dtype=np.uint8)
        original[1:3, 1:3] = (255, 0, 0, 255)
        original[5:7, 5:7] = (0, 0, 255, 255)
        hidden[5:7, 5:7] = (0, 255, 0, 255)

        def encode(array: np.ndarray) -> bytes:
            output = io.BytesIO()
            Image.fromarray(array, mode="RGBA").save(output, format="PNG")
            return output.getvalue()

        summary, heatmap_png = _counterfactual_difference(
            encode(original),
            encode(hidden),
        )

        self.assertTrue(summary["changed"])
        self.assertEqual(summary["changed_area_px"], 8)
        self.assertEqual(summary["became_transparent_area_pct"], 6.25)
        self.assertEqual(summary["repainted_or_revealed_area_pct"], 6.25)
        self.assertEqual(len(summary["components"]), 2)
        self.assertEqual(
            {component["position"] for component in summary["components"]},
            {"top-left", "bottom-right"},
        )
        self.assertGreaterEqual(len(summary["dominant_color_transitions"]), 2)
        heatmap = Image.open(io.BytesIO(heatmap_png)).convert("RGBA")
        self.assertEqual(heatmap.size, (8, 8))
        self.assertEqual(heatmap.getpixel((1, 1))[:3], (255, 59, 48))
        self.assertEqual(heatmap.getpixel((5, 5))[:3], (0, 199, 255))


class QwenCompletionTests(unittest.TestCase):
    def test_reconstruction_candidates_include_nearest_unshortlisted_survivor(self):
        source = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
            '<path fill="#f90" d="M8 20A12 12 0 0 1 32 20H8Z"/>'
            '<circle cx="4" cy="4" r="2" fill="#00f"/>'
            '<rect x="5" y="20" width="30" height="15" fill="#ddd"/>'
            "</svg>"
        )
        scene = {
            node["id"]: node
            for node in build_scene(
                source,
                visual_stats={
                    "n1": {"bbox": [8, 8, 24, 12], "visible": True},
                    "n2": {"bbox": [2, 2, 4, 4], "visible": True},
                    "n3": {"bbox": [5, 20, 30, 15], "visible": True},
                },
            )["nodes"]
        }
        candidates = _completion_reconstruction_candidates(
            source,
            ("n3",),
            {"n3"},
            scene,
        )

        self.assertEqual(candidates[0]["target"], "n1")
        self.assertEqual(
            {candidate["target"] for candidate in candidates},
            {"n1"},
        )

    def test_generated_circle_is_snapped_to_semicircle_endpoints(self):
        existing = list(
            parse_svg(
                '<svg xmlns="http://www.w3.org/2000/svg">'
                '<path d="M8 18A10 10 0 0 1 28 18H8Z"/>'
                "</svg>"
            )
        )[0]
        replacement = list(
            parse_svg(
                '<svg xmlns="http://www.w3.org/2000/svg">'
                '<circle cx="14" cy="18" r="10"/>'
                "</svg>"
            )
        )[0]

        _snap_generated_geometry(existing, replacement)

        self.assertEqual(replacement.attrib["cx"], "18")
        self.assertEqual(replacement.attrib["cy"], "18")
        self.assertEqual(replacement.attrib["r"], "10")

    def test_trigger_requires_transparency_without_revealed_underlayer(self):
        triggered, targets = _completion_triggered(
            [
                {
                    "target": "n2",
                    "difference": {
                        "changed": True,
                        "changed_area_pct": 20.0,
                        "became_transparent_area_pct": 18.0,
                        "revealed_or_repainted_area_pct": 0.0,
                    },
                },
                {
                    "target": "n3",
                    "difference": {
                        "changed": True,
                        "changed_area_pct": 20.0,
                        "became_transparent_area_pct": 4.0,
                        "revealed_or_repainted_area_pct": 16.0,
                    },
                },
            ],
            {"n2", "n3"},
            min_transparent_area_pct=0.25,
            max_revealed_fraction=0.1,
        )
        self.assertTrue(triggered)
        self.assertEqual(targets, ["n2"])

    def test_qwen_generates_only_when_deleted_region_has_no_underlayer(self):
        transparent_png = _png((0, 0, 0, 0))
        model = SequenceModel(
            [
                '{"targets":["n2","n3"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n2"]}]}',
                (
                    '{"target":"n3","element":'
                    '"<circle cx=\\"10\\" cy=\\"10\\" r=\\"8\\" '
                    'fill=\\"#ff0000\\"/>"}'
                ),
            ],
            supports_images=False,
        )
        architecture = SemanticIdPatchArchitecture(
            render_size=64,
            qwen_completion=True,
            completion_max_attempts=1,
        )
        with mock.patch.object(
            architecture,
            "_visual_context",
            return_value=_visual_context(),
        ), mock.patch(
            "svgpatchlab.architectures.semantic.render_svg_png",
            side_effect=[
                transparent_png,
                transparent_png,
                transparent_png,
                _png((0, 0, 255, 255)),
            ],
        ):
            result = architecture.run(_case(), model)

        self.assertIsNone(result.error)
        self.assertEqual(result.model_calls, 3)
        self.assertIn('cx="10"', result.output_svg or "")
        self.assertNotIn("<rect", result.output_svg or "")
        self.assertTrue(result.details["qwen_completion"]["triggered"])
        self.assertTrue(result.details["qwen_completion"]["accepted"])
        self.assertIn(
            "conservative SVG layer-completion generator",
            model.requests[2].prompt,
        )
        self.assertEqual(
            model.requests[2].response_schema_name,
            "svgpatchlab_completion_v1",
        )
        self.assertEqual(
            model.requests[2].response_schema["properties"]["target"]["enum"],
            ["n3"],
        )

    def test_existing_revealed_underlayer_bypasses_qwen_generator(self):
        model = SequenceModel(
            [
                '{"targets":["n2"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n2"]}]}',
            ],
            supports_images=False,
        )
        architecture = SemanticIdPatchArchitecture(
            render_size=64,
            qwen_completion=True,
            completion_max_attempts=1,
        )
        with mock.patch.object(
            architecture,
            "_visual_context",
            return_value=_visual_context(),
        ), mock.patch(
            "svgpatchlab.architectures.semantic.render_svg_png",
            return_value=_png((0, 0, 255, 255)),
        ):
            result = architecture.run(_case(), model)

        self.assertIsNone(result.error)
        self.assertEqual(result.model_calls, 2)
        self.assertFalse(result.details["qwen_completion"]["triggered"])
        self.assertFalse(result.details["qwen_completion"]["accepted"])

    def test_unsafe_qwen_completion_is_rejected_without_losing_deletion(self):
        transparent_png = _png((0, 0, 0, 0))
        unsafe_svg = (
            '{"target":"n3","element":"<script>alert(1)</script>"}'
        )
        model = SequenceModel(
            [
                '{"targets":["n2","n3"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n2"]}]}',
                unsafe_svg,
            ],
            supports_images=False,
        )
        architecture = SemanticIdPatchArchitecture(
            render_size=64,
            qwen_completion=True,
            completion_max_attempts=1,
        )
        with mock.patch.object(
            architecture,
            "_visual_context",
            return_value=_visual_context(),
        ), mock.patch(
            "svgpatchlab.architectures.semantic.render_svg_png",
            side_effect=[transparent_png, transparent_png, transparent_png],
        ):
            result = architecture.run(_case(), model)

        self.assertIsNone(result.error)
        self.assertEqual(result.model_calls, 3)
        self.assertFalse(result.details["qwen_completion"]["accepted"])
        self.assertIn(
            "forbidden element: script",
            result.details["qwen_completion"]["rejection"],
        )
        self.assertNotIn("<script", result.output_svg or "")

    def test_completion_that_restores_deleted_element_is_rejected(self):
        transparent_png = _png((0, 0, 0, 0))
        model = SequenceModel(
            [
                '{"targets":["n2","n3"]}',
                '{"version":2,"operations":[{"op":"remove_element","targets":["n2"]}]}',
                (
                    '{"target":"n3","element":'
                    '"<rect x=\\"0\\" y=\\"0\\" width=\\"10\\" '
                    'height=\\"10\\" fill=\\"#ff0000\\"/>"}'
                ),
            ],
            supports_images=False,
        )
        architecture = SemanticIdPatchArchitecture(
            render_size=64,
            qwen_completion=True,
            completion_max_attempts=1,
        )
        with mock.patch.object(
            architecture,
            "_visual_context",
            return_value=_visual_context(),
        ), mock.patch(
            "svgpatchlab.architectures.semantic.render_svg_png",
            side_effect=[transparent_png, transparent_png, transparent_png],
        ):
            result = architecture.run(_case(), model)

        self.assertIsNone(result.error)
        self.assertEqual(result.model_calls, 3)
        self.assertFalse(result.details["qwen_completion"]["accepted"])
        self.assertIn(
            "restored an element that was deleted",
            result.details["qwen_completion"]["rejection"],
        )
        self.assertNotIn("<rect", result.output_svg or "")


if __name__ == "__main__":
    unittest.main()
