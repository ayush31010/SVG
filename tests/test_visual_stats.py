from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from svgpatchlab.architectures.factory import create_architecture
from svgpatchlab.architectures.patching import (
    SkeletonPatchArchitecture,
    VisualStatsPatchArchitecture,
)
from svgpatchlab.architectures.prompts import patch_prompt
from svgpatchlab.core import build_scene
from svgpatchlab.models.base import ModelAdapter
from svgpatchlab.types import BenchmarkCase, ModelRequest, ModelResponse
from svgpatchlab.eval.render import (
    RendererUnavailable,
    VisualStatsCache,
    _parse_viewbox,
    _position_word,
    _stats_from_diff,
    _stats_from_mask,
    ensure_renderer,
    node_visual_stats,
)
from svgpatchlab.core.xml import parse_svg

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


SIMPLE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">'
    '<rect x="0" y="0" width="36" height="36" fill="#0000ff"/>'
    '<rect x="0" y="0" width="12" height="12" fill="#ff0000"/>'
    "</svg>"
)


def _rgba_canvas(size: int):
    return np.zeros((size, size, 4), dtype=np.float32)


class CapturingModel(ModelAdapter):
    def __init__(self, response: str, *, supports_images: bool = False):
        self.response = response
        self.supports_images = supports_images
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(self.response)


@unittest.skipUnless(np is not None, "numpy not installed")
class StatsFromDiffTests(unittest.TestCase):
    VIEWBOX = (0.0, 0.0, 36.0, 36.0)

    def test_top_left_square(self):
        full = _rgba_canvas(12)
        full[0:4, 0:4] = [1.0, 0.0, 0.0, 1.0]
        without = _rgba_canvas(12)
        stats = _stats_from_diff(full, without, self.VIEWBOX)
        self.assertEqual(stats["bbox"], [0.0, 0.0, 12.0, 12.0])
        self.assertAlmostEqual(stats["area_pct"], 100.0 * 16 / 144, places=2)
        self.assertEqual(stats["position"], "top-left")
        self.assertEqual(stats["color"], "#ff0000")

    def test_center_square(self):
        full = _rgba_canvas(12)
        full[4:8, 4:8] = [0.0, 1.0, 0.0, 1.0]
        without = _rgba_canvas(12)
        stats = _stats_from_diff(full, without, self.VIEWBOX)
        self.assertEqual(stats["position"], "center")
        self.assertEqual(stats["bbox"], [12.0, 12.0, 12.0, 12.0])
        self.assertEqual(stats["color"], "#00ff00")

    def test_identical_renders_mean_invisible(self):
        full = _rgba_canvas(8)
        full[2:5, 2:5] = [0.3, 0.3, 0.3, 1.0]
        stats = _stats_from_diff(full, full.copy(), self.VIEWBOX)
        self.assertEqual(stats, {"visible": False})

    def test_below_threshold_contribution_is_reported_as_not_measurable(self):
        full = _rgba_canvas(8)
        full[2:5, 2:5] = [0.01, 0.01, 0.01, 0.01]
        without = _rgba_canvas(8)

        stats = _stats_from_diff(full, without, self.VIEWBOX)

        self.assertEqual(stats, {"visible": False})

    def test_viewbox_offset_shifts_bbox(self):
        full = _rgba_canvas(10)
        full[0:5, 0:5] = [1.0, 1.0, 1.0, 1.0]
        without = _rgba_canvas(10)
        stats = _stats_from_diff(
            full,
            without,
            (10.0, 20.0, 100.0, 50.0),
            preserve_aspect_ratio="none",
        )
        self.assertEqual(stats["bbox"], [10.0, 20.0, 50.0, 25.0])

    def test_default_aspect_ratio_accounts_for_letterboxing(self):
        full = _rgba_canvas(20)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 0:10] = True
        full[mask] = [1.0, 0.0, 0.0, 1.0]

        stats = _stats_from_mask(full, mask, (0.0, 0.0, 100.0, 50.0))

        self.assertEqual(stats["bbox"], [0.0, 0.0, 50.0, 25.0])
        self.assertEqual(stats["position"], "top-left")

    def test_dominant_color_is_observed_not_cross_channel_median(self):
        full = _rgba_canvas(4)
        mask = np.ones((4, 4), dtype=bool)
        full[:, :2] = [1.0, 0.0, 0.0, 1.0]
        full[:, 2:] = [0.0, 0.0, 1.0, 1.0]

        stats = _stats_from_mask(
            full,
            mask,
            (0.0, 0.0, 4.0, 4.0),
            preserve_aspect_ratio="none",
        )

        self.assertIn(stats["color"], {"#ff0000", "#0000ff"})
        self.assertNotEqual(stats["color"], "#800080")


class PositionWordTests(unittest.TestCase):
    def test_grid_words(self):
        self.assertEqual(_position_word(0.1, 0.1), "top-left")
        self.assertEqual(_position_word(0.5, 0.1), "top")
        self.assertEqual(_position_word(0.9, 0.5), "right")
        self.assertEqual(_position_word(0.5, 0.5), "center")
        self.assertEqual(_position_word(0.9, 0.9), "bottom-right")


class ParseViewboxTests(unittest.TestCase):
    def test_standard_viewbox(self):
        root = parse_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36"/>')
        self.assertEqual(_parse_viewbox(root), (0.0, 0.0, 36.0, 36.0))

    def test_comma_separated_viewbox(self):
        root = parse_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="1,2,3,4"/>')
        self.assertEqual(_parse_viewbox(root), (1.0, 2.0, 3.0, 4.0))

    def test_missing_or_invalid_viewbox(self):
        root = parse_svg('<svg xmlns="http://www.w3.org/2000/svg"/>')
        self.assertIsNone(_parse_viewbox(root))
        root = parse_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 0 36"/>')
        self.assertIsNone(_parse_viewbox(root))


class VisualStatsCacheTests(unittest.TestCase):
    CANNED = {"n1": {"bbox": [0, 0, 12, 12], "area_pct": 11.11, "position": "top-left", "color": "#ff0000"}}

    def test_compute_then_hit_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = VisualStatsCache(tmp)
            with mock.patch(
                "svgpatchlab.eval.render.node_visual_stats", return_value=self.CANNED
            ) as compute:
                first = cache.get_or_compute(SIMPLE_SVG)
            self.assertEqual(first, self.CANNED)
            self.assertEqual(compute.call_count, 1)
            files = list(Path(tmp).glob("*.json"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], VisualStatsCache.FORMAT)

            with mock.patch(
                "svgpatchlab.eval.render.node_visual_stats",
                side_effect=AssertionError("should not recompute"),
            ):
                second = cache.get_or_compute(SIMPLE_SVG)
            self.assertEqual(second, self.CANNED)

    def test_corrupt_cache_file_recomputes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = VisualStatsCache(tmp)
            with mock.patch("svgpatchlab.eval.render.node_visual_stats", return_value=self.CANNED):
                cache.get_or_compute(SIMPLE_SVG)
            for file in Path(tmp).glob("*.json"):
                file.write_text("not json", encoding="utf-8")
            with mock.patch("svgpatchlab.eval.render.node_visual_stats", return_value=self.CANNED):
                self.assertEqual(cache.get_or_compute(SIMPLE_SVG), self.CANNED)

    def test_render_size_changes_cache_key(self):
        cache = VisualStatsCache("unused")
        self.assertNotEqual(cache._path(SIMPLE_SVG, 64), cache._path(SIMPLE_SVG, 128))


class SceneVisualFieldTests(unittest.TestCase):
    def test_visual_stats_attached_to_matching_nodes(self):
        stats = {"n1": {"bbox": [0, 0, 36, 36], "position": "center"}}
        scene = build_scene(SIMPLE_SVG, visual_stats=stats)
        by_id = {node["id"]: node for node in scene["nodes"]}
        self.assertEqual(by_id["n1"]["visual"]["position"], "center")
        self.assertNotIn("visual", by_id["n0"])
        self.assertNotIn("visual", by_id["n2"])

    def test_scene_unchanged_without_stats(self):
        scene = build_scene(SIMPLE_SVG)
        self.assertTrue(all("visual" not in node for node in scene["nodes"]))


class ArchitectureRegistrationTests(unittest.TestCase):
    def test_visual_stats_patch_is_registered(self):
        architecture = create_architecture("visual_stats_patch")
        self.assertEqual(architecture.name, "visual_stats_patch")


class CleanArchitectureABTests(unittest.TestCase):
    def test_standalone_presets_keep_dataset_model_and_evaluator_paired(self):
        config_root = Path("configs/experiments")
        baseline = json.loads(
            (config_root / "skeleton_patch.json").read_text(encoding="utf-8")
        )
        visual = json.loads(
            (config_root / "visual_stats_patch.json").read_text(encoding="utf-8")
        )

        self.assertEqual(visual["dataset"], baseline["dataset"])
        self.assertEqual(visual["model"], baseline["model"])
        for key in ("render", "render_size", "save_outputs"):
            self.assertEqual(
                visual["evaluation"].get(key),
                baseline["evaluation"].get(key),
            )

    def test_only_visual_fields_differ_at_model_boundary(self):
        case = BenchmarkCase(
            task="change_color",
            emoji_id="visual-ab",
            instruction="Change the small red shape in the top-left to green.",
            source_svg=SIMPLE_SVG,
            answer_svg=SIMPLE_SVG,
            query_path=Path("."),
            answer_path=Path("."),
        )
        stats = {
            "n0": {
                "bbox": [0.0, 0.0, 36.0, 36.0],
                "area_pct": 100.0,
                "position": "center",
                "color": "#0000ff",
            },
            "n1": {
                "bbox": [0.0, 0.0, 36.0, 36.0],
                "area_pct": 88.89,
                "position": "center",
                "color": "#0000ff",
            },
            "n2": {
                "bbox": [0.0, 0.0, 12.0, 12.0],
                "area_pct": 11.11,
                "position": "top-left",
                "color": "#ff0000",
            },
        }
        response = (
            '{"version":1,"operations":[{"op":"set_attributes",'
            '"targets":["n2"],"attributes":{"fill":"#00ff00"}}]}'
        )
        baseline = SkeletonPatchArchitecture()
        visual = VisualStatsPatchArchitecture(cache_dir="unused")

        baseline_scene = baseline.scene_for(case)
        with mock.patch.object(
            visual._cache,
            "get_or_compute",
            return_value=stats,
        ) as compute:
            visual_scene = visual.scene_for(case)
        compute.assert_called_once_with(SIMPLE_SVG, size=64)

        stripped_visual_scene = json.loads(json.dumps(visual_scene))
        for node in stripped_visual_scene["nodes"]:
            node.pop("visual", None)
        self.assertEqual(stripped_visual_scene, baseline_scene)
        self.assertTrue(
            all("visual" not in node for node in baseline_scene["nodes"])
        )
        self.assertTrue(
            all("visual" in node for node in visual_scene["nodes"])
        )

        baseline_model = CapturingModel(response, supports_images=True)
        visual_model = CapturingModel(response, supports_images=True)
        baseline_result = baseline.run(case, baseline_model)
        with mock.patch.object(
            visual._cache,
            "get_or_compute",
            return_value=stats,
        ):
            visual_result = visual.run(case, visual_model)

        expected_baseline_prompt = patch_prompt(
            case.instruction,
            "SVG DOM skeleton",
            json.dumps(baseline_scene, indent=2, sort_keys=True),
        )
        expected_visual_prompt = patch_prompt(
            case.instruction,
            "SVG DOM skeleton",
            json.dumps(visual_scene, indent=2, sort_keys=True),
        )
        self.assertEqual(baseline_model.requests[0].prompt, expected_baseline_prompt)
        self.assertEqual(visual_model.requests[0].prompt, expected_visual_prompt)
        self.assertEqual(baseline_model.requests[0].images, ())
        self.assertEqual(visual_model.requests[0].images, ())
        self.assertEqual(
            baseline_model.requests[0].metadata,
            visual_model.requests[0].metadata,
        )
        self.assertNotIn('"visual"', baseline_model.requests[0].prompt)
        self.assertIn('"visual"', visual_model.requests[0].prompt)

        self.assertIsNone(baseline_result.error)
        self.assertIsNone(visual_result.error)
        self.assertEqual(
            baseline_result.patch.to_dict(),
            visual_result.patch.to_dict(),
        )
        self.assertEqual(baseline_result.output_svg, visual_result.output_svg)
        self.assertIs(
            VisualStatsPatchArchitecture.run,
            SkeletonPatchArchitecture.run,
        )
        self.assertIs(
            VisualStatsPatchArchitecture.context,
            SkeletonPatchArchitecture.context,
        )


class EndToEndRenderTests(unittest.TestCase):
    def setUp(self):
        try:
            ensure_renderer()
        except RendererUnavailable:
            self.skipTest("render dependencies not installed")

    def test_stats_on_real_render(self):
        stats = node_visual_stats(SIMPLE_SVG, size=36)
        self.assertEqual(stats["n2"]["color"], "#ff0000")
        self.assertEqual(stats["n2"]["position"], "top-left")
        x, y, w, h = stats["n2"]["bbox"]
        self.assertLessEqual(abs(x), 1.5)
        self.assertLessEqual(abs(y), 1.5)
        self.assertLessEqual(abs(w - 12), 2.0)
        self.assertLessEqual(abs(h - 12), 2.0)

    def test_fully_occluded_node_flagged(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">'
            '<rect x="8" y="8" width="8" height="8" fill="#00ff00"/>'
            '<rect x="0" y="0" width="36" height="36" fill="#0000ff"/>'
            "</svg>"
        )
        stats = node_visual_stats(svg, size=36)
        self.assertEqual(stats["n1"], {"visible": False})
        self.assertEqual(stats["n2"]["color"], "#0000ff")

    def test_identical_underlayer_means_no_unique_pixel_contribution(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">'
            '<rect x="4" y="4" width="12" height="12" fill="#ff0000"/>'
            '<rect x="4" y="4" width="12" height="12" fill="#ff0000"/>'
            "</svg>"
        )

        stats = node_visual_stats(svg, size=36)

        self.assertEqual(stats["n2"], {"visible": False})

    def test_hide_measurement_overrides_existing_important_display(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">'
            '<rect x="4" y="4" width="12" height="12" fill="#ff0000" '
            'style="display:block!important"/>'
            "</svg>"
        )

        stats = node_visual_stats(svg, size=36)

        self.assertEqual(stats["n1"]["color"], "#ff0000")


if __name__ == "__main__":
    unittest.main()
