from __future__ import annotations

import unittest

from scripts.analyze_vector_edits_groups import structural_candidate_groups
from scripts.analyze_vector_edits_routing import complexity_hybrid
from scripts.run_vector_edits_grounding import (
    derive_natural_grounding_case,
    strip_known_svg_doctype,
)
from svgpatchlab.vision import StructuralGroupGrounder, extract_target_reference
from svgpatchlab.vision.graph_features import parse_color
from svgpatchlab.vision.siglip_grounder import _pooled_tensor


def _record(source: str, target: str, instruction: str = "recolor the middle box"):
    return {
        "collection_slug": "unit-test",
        "instruction": instruction,
        "item_1": {"item_id": 1, "item_svg": source},
        "item_2": {"item_id": 2, "item_svg": target},
    }


class InstructionDecompositionTests(unittest.TestCase):
    def test_destination_direction_is_not_part_of_target_reference(self):
        self.assertEqual(
            extract_target_reference("rotate the arrow inside the circle to point left"),
            "the arrow inside the circle",
        )
        self.assertEqual(
            extract_target_reference("Move the needle on the gauge to the right"),
            "the needle on the gauge",
        )

    def test_multiple_source_color_clauses_are_retained(self):
        self.assertEqual(
            extract_target_reference(
                "change the color of the cups from white to blue and "
                "the color of the stems from green to olive green"
            ),
            "the cups with white color and the stems with green color",
        )

    def test_source_spatial_description_is_preserved(self):
        self.assertEqual(
            extract_target_reference("shorten the third bar from the top"),
            "the third bar from the top",
        )
        self.assertEqual(
            extract_target_reference("Remove the bottom right shape from the image"),
            "the bottom right shape",
        )


class VectorEditsExtractionTests(unittest.TestCase):
    def test_aligned_drawable_diff_becomes_grounding_case(self):
        source = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect fill="red"/><rect fill="blue"/><rect fill="green"/></svg>'
        )
        target = source.replace('fill="blue"', 'fill="black"')
        case, reason = derive_natural_grounding_case(_record(source, target), 7)
        self.assertEqual(reason, "accepted")
        self.assertIsNotNone(case)
        self.assertEqual(case.candidate_ids, ("n1", "n2", "n3"))
        self.assertEqual(case.gold_target_ids, ("n2",))
        self.assertEqual(case.changed_attributes, ("fill",))

    def test_topology_change_is_rejected(self):
        source = '<svg><rect/><circle/><path d="M0 0"/></svg>'
        target = '<svg><rect/><circle/><path d="M0 0"/><line/></svg>'
        case, reason = derive_natural_grounding_case(_record(source, target), 0)
        self.assertIsNone(case)
        self.assertEqual(reason, "topology_changed")

    def test_only_known_entity_free_svg_doctype_can_be_stripped(self):
        doctype = (
            '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
            '"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">'
        )
        self.assertEqual(strip_known_svg_doctype(doctype + "<svg/>"), "<svg/>")
        with self.assertRaises(ValueError):
            strip_known_svg_doctype("<!DOCTYPE svg SYSTEM 'other.dtd'><svg/>")
        with self.assertRaises(ValueError):
            strip_known_svg_doctype(doctype + "<!ENTITY x 'bad'><svg/>")


class SiglipCompatibilityTests(unittest.TestCase):
    def test_transformers_model_output_pooler_is_supported(self):
        sentinel = object()

        class Output:
            pooler_output = sentinel

        self.assertIs(_pooled_tensor(Output()), sentinel)


class ComplexityRoutingTests(unittest.TestCase):
    @staticmethod
    def _result(case_id: str, candidates: int, source: str):
        return {
            "case_id": case_id,
            "candidate_ids": [f"n{index}" for index in range(candidates)],
            "source": source,
        }

    def test_threshold_routes_only_complex_cases_to_visual_arm(self):
        structural = [
            self._result("simple", 9, "structural"),
            self._result("complex", 10, "structural"),
        ]
        visual = [
            self._result("simple", 9, "visual"),
            self._result("complex", 10, "visual"),
        ]
        routed = complexity_hybrid(structural, visual, threshold=10)
        self.assertEqual([record["source"] for record in routed], ["structural", "visual"])
        self.assertEqual(
            [record["routed_from_arm"] for record in routed],
            ["structural", "visual"],
        )

    def test_mismatched_cases_are_rejected(self):
        with self.assertRaises(ValueError):
            complexity_hybrid(
                [self._result("a", 3, "structural")],
                [self._result("b", 3, "visual")],
                threshold=3,
            )


class StructuralGroupTests(unittest.TestCase):
    def test_css_rgb_color_is_available_to_graph_and_group_features(self):
        self.assertEqual(parse_color("rgb(174,32,37)"), (174 / 255, 32 / 255, 37 / 255))
        self.assertEqual(parse_color("rgb(50% 0% 100%)"), (0.5, 0.0, 1.0))

    def test_dom_and_paint_groups_are_derived_without_a_target_svg(self):
        source = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<g><path fill="red"/><path fill="red"/></g>'
            '<circle fill="blue"/></svg>'
        )
        groups = structural_candidate_groups(source, ("n2", "n3", "n4"))
        self.assertIn(("n2", "n3"), groups)
        self.assertIn("dom_subtree", groups[("n2", "n3")])
        self.assertIn("shared_paint", groups[("n2", "n3")])

    def test_whole_canvas_candidate_set_is_not_a_group(self):
        source = '<svg><g><path/><path/></g></svg>'
        self.assertEqual(
            structural_candidate_groups(source, ("n2", "n3")),
            {},
        )

    def test_group_grounder_selects_source_paint_set(self):
        source = (
            '<svg viewBox="0 0 100 100">'
            '<path fill="red" d="M0 0h10v10z"/>'
            '<path fill="blue" d="M20 0h10v10z"/>'
            '<path fill="red" d="M40 0h10v10z"/></svg>'
        )
        prediction = StructuralGroupGrounder().predict(
            source, "the red shapes", ("n1", "n2", "n3")
        )
        self.assertEqual(prediction.selected_ids, ("n1", "n3"))
        self.assertEqual(prediction.rule, "source_paint_set")

    def test_group_grounder_maps_color_name_to_nearest_source_palette(self):
        source = (
            '<svg viewBox="0 0 100 100">'
            '<path style="fill:rgb(174,32,37)" d="M0 0h10v10z"/>'
            '<path style="fill:rgb(103,154,69)" d="M20 0h10v10z"/>'
            '<path style="fill:rgb(174,32,37)" d="M40 0h10v10z"/></svg>'
        )
        prediction = StructuralGroupGrounder().predict(
            source, "the red shapes", ("n1", "n2", "n3")
        )
        self.assertEqual(prediction.selected_ids, ("n1", "n3"))

    def test_group_grounder_selects_nodes_inside_named_container(self):
        source = (
            '<svg viewBox="0 0 100 100">'
            '<circle cx="50" cy="50" r="40" fill="none" stroke="black"/>'
            '<path d="M40 40h10"/><path d="M50 40v10"/>'
            '<path d="M95 95h2"/></svg>'
        )
        prediction = StructuralGroupGrounder().predict(
            source, "the arrow inside the circle", ("n1", "n2", "n3", "n4")
        )
        self.assertEqual(prediction.selected_ids, ("n2", "n3"))
        self.assertEqual(prediction.rule, "inside_container_set")


if __name__ == "__main__":
    unittest.main()
