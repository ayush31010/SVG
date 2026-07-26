from __future__ import annotations

import hashlib
import unittest

from scripts.run_occlusion_deletion_benchmark import (
    VARIANTS,
    build_cases,
    validate_suite,
)
from svgpatchlab.core.xml import index_tree, parse_svg


class OcclusionDeletionDesignTests(unittest.TestCase):
    def test_suite_pairs_complete_and_missing_underlayers(self):
        cases = build_cases()
        self.assertEqual(len(cases), 2 * len(VARIANTS))
        by_family = {
            family: [item for item in cases if item.family == family]
            for family in {item.family for item in cases}
        }
        self.assertEqual(
            set(by_family),
            {"complete_underlayer", "missing_underlayer"},
        )
        self.assertTrue(
            all(not item.completion_expected for item in by_family["complete_underlayer"])
        )
        self.assertTrue(
            all(item.completion_expected for item in by_family["missing_underlayer"])
        )

    def test_target_and_reconstruction_ids_are_frozen(self):
        for item in build_cases():
            with self.subTest(case=item.case.case_id):
                nodes = index_tree(parse_svg(item.case.source_svg))
                self.assertEqual(nodes[1].node_id, item.reconstruction_target)
                self.assertEqual(nodes[3].node_id, item.expected_removed_target)
                self.assertEqual(
                    nodes[3].element.attrib["fill"].lower(),
                    "#e1e8ed",
                )

    def test_missing_sources_use_partial_paths_but_answers_use_circles(self):
        for item in build_cases():
            source_tag = index_tree(parse_svg(item.case.source_svg))[1].element.tag
            answer_tag = index_tree(parse_svg(item.case.answer_svg))[1].element.tag
            source_tag = source_tag.rsplit("}", 1)[-1]
            answer_tag = answer_tag.rsplit("}", 1)[-1]
            self.assertEqual(answer_tag, "circle")
            self.assertEqual(
                source_tag,
                "path" if item.completion_expected else "circle",
            )

    def test_design_validation_and_sources_are_stable(self):
        cases = build_cases()
        design = validate_suite(cases)
        self.assertEqual(design["cases"], 8)
        hashes = [
            hashlib.sha256(item.case.source_svg.encode("utf-8")).hexdigest()
            for item in cases
        ]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(
            design["cases_by_family"],
            {"complete_underlayer": 4, "missing_underlayer": 4},
        )


if __name__ == "__main__":
    unittest.main()
