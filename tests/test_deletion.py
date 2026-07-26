from __future__ import annotations

import unittest

from svgpatchlab.core import apply_patch, build_scene, derive_patch, validate_patch
from svgpatchlab.core.patch import Patch, PatchError, PatchOperation
from svgpatchlab.core.xml import local_name, parse_svg
from svgpatchlab.eval.metrics import patch_scores, structural_scores


SOURCE = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <path id="first" d="M0 0 L1 1" fill="red"/>
  <path id="middle" d="M2 2 L3 3" fill="green"/>
  <path id="last" d="M4 4 L5 5" fill="blue"/>
</svg>
"""


def _child_ids(svg: str) -> list[str]:
    return [
        child.attrib["id"]
        for child in parse_svg(svg)
        if local_name(child.tag) == "path"
    ]


class DeletionExecutionTests(unittest.TestCase):
    def test_separate_remove_operations_keep_source_node_ids_stable(self):
        patch = Patch(
            (
                PatchOperation("remove_element", ("n1",)),
                PatchOperation("remove_element", ("n3",)),
            ),
            version=2,
        )

        validate_patch(patch, build_scene(SOURCE), task="delete")
        self.assertEqual(_child_ids(apply_patch(SOURCE, patch)), ["middle"])

    def test_ancestor_and_descendant_targets_have_order_independent_result(self):
        source = """\
<svg xmlns="http://www.w3.org/2000/svg">
  <g id="object"><path id="detail" d="M0 0 L1 1"/></g>
  <path id="keep" d="M2 2 L3 3"/>
</svg>
"""
        descendant_first = Patch(
            (PatchOperation("remove_element", ("n2", "n1")),),
            version=2,
        )
        ancestor_first = Patch(
            (PatchOperation("remove_element", ("n1", "n2")),),
            version=2,
        )

        self.assertEqual(
            _child_ids(apply_patch(source, descendant_first)),
            _child_ids(apply_patch(source, ancestor_first)),
        )
        self.assertEqual(_child_ids(apply_patch(source, ancestor_first)), ["keep"])


class DeletionPatchTests(unittest.TestCase):
    def test_derive_patch_identifies_a_deleted_middle_sibling(self):
        answer = SOURCE.replace(
            '  <path id="middle" d="M2 2 L3 3" fill="green"/>\n',
            "",
        )

        patch = derive_patch(SOURCE, answer)

        self.assertEqual(
            patch,
            Patch((PatchOperation("remove_element", ("n2",)),), version=2),
        )
        self.assertEqual(_child_ids(apply_patch(SOURCE, patch)), ["first", "last"])

    def test_patch_scores_distinguish_remove_targets(self):
        gold = Patch((PatchOperation("remove_element", ("n2",)),), version=2)
        correct = Patch((PatchOperation("remove_element", ("n2",)),), version=2)
        wrong = Patch((PatchOperation("remove_element", ("n1",)),), version=2)

        self.assertEqual(
            patch_scores(correct, gold),
            {
                "gold_patch_exact": True,
                "patch_precision": 1.0,
                "patch_recall": 1.0,
            },
        )
        self.assertEqual(
            patch_scores(wrong, gold),
            {
                "gold_patch_exact": False,
                "patch_precision": 0.0,
                "patch_recall": 0.0,
            },
        )

    def test_remove_requires_version_two_nonempty_unique_targets(self):
        scene = build_scene(SOURCE)
        invalid = (
            Patch((PatchOperation("remove_element", ("n1",)),), version=1),
            Patch((PatchOperation("remove_element"),), version=2),
            Patch((PatchOperation("remove_element", ("n1", "n1")),), version=2),
            Patch(
                (
                    PatchOperation("remove_element", ("n1",)),
                    PatchOperation("remove_element", ("n1",)),
                ),
                version=2,
            ),
        )

        for patch in invalid:
            with self.subTest(patch=patch):
                with self.assertRaises(PatchError):
                    validate_patch(patch, scene, task="delete")


class DeletionMetricTests(unittest.TestCase):
    def test_correct_deletion_preserves_remaining_protected_geometry(self):
        answer = SOURCE.replace(
            '  <path id="middle" d="M2 2 L3 3" fill="green"/>\n',
            "",
        )

        scores = structural_scores(SOURCE, answer, answer)

        self.assertEqual(scores["changed_nodes"], 1)
        self.assertTrue(scores["protected_geometry_preserved"])
        self.assertTrue(scores["reference_structure_match"])

    def test_mutating_a_surviving_path_is_not_treated_as_deletion(self):
        answer = SOURCE.replace(
            '  <path id="middle" d="M2 2 L3 3" fill="green"/>\n',
            "",
        )
        mutated = answer.replace("M4 4 L5 5", "M4 4 L9 9")

        scores = structural_scores(SOURCE, mutated, answer)

        self.assertFalse(scores["protected_geometry_preserved"])
        self.assertFalse(scores["reference_structure_match"])


if __name__ == "__main__":
    unittest.main()
