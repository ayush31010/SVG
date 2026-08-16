"""Analytic geometry must describe nominal extent, and must not fake occlusion."""
import unittest

from svgpatchlab.architectures import create_architecture
from svgpatchlab.core.geometry import node_analytic_stats

SQUARE_TOP_LEFT = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<rect fill="#ff0000" x="0" y="0" width="20" height="20"/>'
    "</svg>"
)
TWO_SHAPES = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<rect fill="#ff0000" x="0" y="0" width="20" height="20"/>'
    '<rect fill="#0000ff" x="80" y="80" width="20" height="20"/>'
    "</svg>"
)
COVERED = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<rect fill="#ff0000" x="10" y="10" width="40" height="40"/>'
    '<rect fill="#00ff00" x="0" y="0" width="100" height="100"/>'
    "</svg>"
)
GROUPED = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<g><rect fill="#ff0000" x="0" y="0" width="10" height="10"/>'
    '<rect fill="#0000ff" x="90" y="90" width="10" height="10"/></g>'
    "</svg>"
)
TRANSFORMED = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<rect fill="#ff0000" x="0" y="0" width="10" height="10" '
    'transform="translate(90 90)"/>'
    "</svg>"
)
DISJOINT_SUBPATHS = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<path fill="#ff0000" '
    'd="M0 0H10V10H0Z M90 90H100V100H90Z"/>'
    "</svg>"
)
EVENODD_HOLE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<path fill="#ff0000" fill-rule="evenodd" '
    'd="M0 0H100V100H0Z M25 25H75V75H25Z"/>'
    "</svg>"
)
NONZERO_HOLE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<path fill="#ff0000" '
    'd="M0 0H100V100H0Z M25 25V75H75V25Z"/>'
    "</svg>"
)
ASYMMETRIC_TRIANGLE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
    '<path fill="#ff0000" d="M0 0H90L0 30Z"/>'
    "</svg>"
)


class GeometryTest(unittest.TestCase):
    def test_bbox_and_area_of_a_simple_rect(self):
        stats = node_analytic_stats(SQUARE_TOP_LEFT)["n1"]
        self.assertEqual(stats["bbox"], [0.0, 0.0, 20.0, 20.0])
        self.assertAlmostEqual(stats["area_pct"], 4.0, places=1)
        self.assertEqual(stats["position"], "top-left")
        self.assertEqual(stats["color"], "#ff0000")

    def test_position_words_span_the_canvas(self):
        stats = node_analytic_stats(TWO_SHAPES)
        self.assertEqual(stats["n1"]["position"], "top-left")
        self.assertEqual(stats["n2"]["position"], "bottom-right")

    def test_transform_is_applied(self):
        stats = node_analytic_stats(TRANSFORMED)["n1"]
        self.assertEqual(stats["bbox"], [90.0, 90.0, 10.0, 10.0])
        self.assertEqual(stats["position"], "bottom-right")

    def test_disjoint_subpaths_do_not_gain_connecting_area(self):
        stats = node_analytic_stats(DISJOINT_SUBPATHS)["n1"]
        self.assertAlmostEqual(stats["area_pct"], 2.0, places=2)

    def test_evenodd_fill_rule_subtracts_a_hole(self):
        stats = node_analytic_stats(EVENODD_HOLE)["n1"]
        self.assertAlmostEqual(stats["area_pct"], 75.0, places=2)

    def test_nonzero_fill_rule_uses_contour_winding_for_a_hole(self):
        stats = node_analytic_stats(NONZERO_HOLE)["n1"]
        self.assertAlmostEqual(stats["area_pct"], 75.0, places=2)

    def test_position_uses_filled_area_centroid_not_bbox_center(self):
        stats = node_analytic_stats(ASYMMETRIC_TRIANGLE)["n1"]
        self.assertEqual(stats["position"], "top-left")

    def test_group_reports_the_union_of_its_children(self):
        stats = node_analytic_stats(GROUPED)
        self.assertEqual(stats["n1"]["bbox"], [0.0, 0.0, 100.0, 100.0])

    def test_never_reports_visible(self):
        """Occlusion is not knowable without rendering; claiming otherwise
        would silently turn the ablation baseline into the treatment."""
        for svg in (SQUARE_TOP_LEFT, TWO_SHAPES, COVERED, GROUPED):
            for stats in node_analytic_stats(svg).values():
                self.assertNotIn("visible", stats)

    def test_a_fully_covered_node_still_reports_its_nominal_extent(self):
        """The whole point of the ablation: analytic geometry cannot tell that
        n1 is invisible, and reports it as if it were on top."""
        stats = node_analytic_stats(COVERED)
        self.assertEqual(stats["n1"]["bbox"], [10.0, 10.0, 40.0, 40.0])
        self.assertGreater(stats["n1"]["area_pct"], 15.0)

    def test_fields_match_the_rendered_block(self):
        rendered_fields = {"bbox", "area_pct", "position", "color"}
        for stats in node_analytic_stats(TWO_SHAPES).values():
            self.assertTrue(set(stats).issubset(rendered_fields))

    def test_malformed_geometry_is_skipped_not_fatal(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            '<path fill="#000" d="not a path"/>'
            '<rect fill="#f00" x="0" y="0" width="2" height="2"/>'
            "</svg>"
        )
        stats = node_analytic_stats(svg)
        self.assertIn("n2", stats)


class ArchitectureTest(unittest.TestCase):
    def test_registered_and_needs_no_renderer(self):
        architecture = create_architecture("analytic_stats_patch")
        self.assertFalse(getattr(architecture, "requires_renderer", False))

    def test_scene_carries_analytic_values_under_the_visual_key(self):
        """Same key as the rendered arm so the prompt shape is identical and
        only the provenance of the numbers differs."""
        from pathlib import Path

        from svgpatchlab.types import BenchmarkCase

        case = BenchmarkCase(
            task="change_color",
            emoji_id="t",
            instruction="i",
            source_svg=TWO_SHAPES,
            answer_svg=TWO_SHAPES,
            query_path=Path("q"),
            answer_path=Path("a"),
        )
        scene = create_architecture("analytic_stats_patch").scene_for(case)
        blocks = [n["visual"] for n in scene["nodes"] if "visual" in n]
        self.assertTrue(blocks)
        self.assertIn("position", blocks[0])

    def test_strict_variant_constrains_output(self):
        architecture = create_architecture("strict_analytic_stats_patch")
        self.assertTrue(architecture.constrain_output)


if __name__ == "__main__":
    unittest.main()
