from __future__ import annotations

import unittest

from scripts.analyze_vector_edits_routing import complexity_hybrid
from scripts.run_vector_edits_grounding import (
    derive_natural_grounding_case,
    strip_known_svg_doctype,
)
from svgpatchlab.vision import extract_target_reference
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


if __name__ == "__main__":
    unittest.main()
