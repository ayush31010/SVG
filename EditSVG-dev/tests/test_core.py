from __future__ import annotations

import json
import unittest

from svgpatchlab.architectures.prompts import (
    PATCH_EXAMPLES,
    PATCH_PROMPT_TEMPLATE_DIR,
    PATCH_PROMPT_VERSION,
    patch_prompt,
)
from svgpatchlab.core import apply_patch, build_scene, derive_patch, validate_patch
from svgpatchlab.core.patch import Patch, PatchError, PatchOperation, parse_patch
from svgpatchlab.core.xml import normalized_tree, parse_svg, protected_geometry
from svgpatchlab.data import SVGEditBench


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = SVGEditBench("SVGEditBench")

    def test_scene_hides_path_coordinates(self):
        case = next(self.benchmark.iter_cases())
        root = parse_svg(case.source_svg)
        raw_path = next(element.attrib["d"] for element in root.iter() if "d" in element.attrib)
        scene_text = json.dumps(build_scene(case.source_svg))
        self.assertNotIn(raw_path, scene_text)
        self.assertIn("sha256", scene_text)

    def test_protected_path_attribute_is_rejected(self):
        case = next(self.benchmark.iter_cases())
        patch = Patch((PatchOperation("set_attributes", ("n1",), (("d", "M0 0"),)),))
        with self.assertRaises(PatchError):
            validate_patch(patch, build_scene(case.source_svg), task=case.task)

    def test_unknown_patch_fields_are_rejected(self):
        with self.assertRaises(PatchError):
            parse_patch('{"version":1,"operations":[],"surprise":true}')

    def test_unsafe_attribute_values_are_rejected(self):
        case = next(self.benchmark.iter_cases())
        patch = Patch(
            (PatchOperation("set_attributes", ("n1",), (("fill", "url(https://bad)"),)),)
        )
        with self.assertRaises(PatchError):
            validate_patch(patch, build_scene(case.source_svg), task=case.task)

    def test_all_gold_patches_reproduce_answers(self):
        for case in self.benchmark.iter_cases():
            with self.subTest(case=case.case_id):
                patch = derive_patch(case.source_svg, case.answer_svg)
                validate_patch(patch, build_scene(case.source_svg), task=case.task)
                output = apply_patch(case.source_svg, patch)
                self.assertEqual(
                    normalized_tree(parse_svg(output)),
                    normalized_tree(parse_svg(case.answer_svg)),
                )
                self.assertEqual(
                    protected_geometry(parse_svg(output)),
                    protected_geometry(parse_svg(case.source_svg)),
                )


class PromptTests(unittest.TestCase):
    def test_patch_prompt_template_is_the_active_skeleton_prompt(self):
        path = PATCH_PROMPT_TEMPLATE_DIR / f"patch_v{PATCH_PROMPT_VERSION}.txt"
        self.assertTrue(path.exists())
        self.assertIn("$instruction", path.read_text())

    def test_patch_prompt_v3_is_generic_and_rule_driven(self):
        prompt = patch_prompt(
            "Make the drawing half transparent.",
            "SVG DOM skeleton",
            '{"root_id":"n0","nodes":[{"id":"n0","tag":"svg"}]}',
        )
        self.assertEqual(PATCH_PROMPT_VERSION, 3)
        self.assertIn(f"svgpatchlab.patch.v{PATCH_PROMPT_VERSION}", prompt)
        for attribute in ("stroke-width", "opacity", "transform", "viewBox"):
            self.assertIn(attribute, prompt)
        self.assertIn("Generic SVG edit rules:", prompt)
        self.assertIn("Compute every target, color, number, and attribute value", prompt)
        self.assertNotIn("insert_primitive", prompt)
        self.assertNotIn("remove_attributes", prompt)
        self.assertNotIn("NODE_ID", prompt)
        self.assertNotIn("ATTRIBUTE", prompt)
        self.assertNotIn('"VALUE"', prompt)
        self.assertIn("Actual edit instruction:", prompt)
        self.assertIn("Output the JSON patch object now.", prompt)

    def test_patch_prompt_v3_has_no_few_shot_examples(self):
        self.assertEqual(PATCH_EXAMPLES, ())
        self.assertNotIn("Example 1", patch_prompt("Flip it.", "SVG DOM skeleton", "{}"))

    def test_patch_prompt_v3_does_not_include_copyable_benchmark_constants(self):
        prompt = patch_prompt(
            "Please trim the right half and keep the left half.",
            "SVG DOM skeleton",
            '{"root_id":"n0","nodes":[{"id":"n0","tag":"svg","attributes":{"viewBox":"0 0 36 36"}}]}',
        )
        examples = json.dumps(PATCH_EXAMPLES)
        self.assertNotIn("translate(0,36) scale(1,-1)", examples)
        self.assertNotIn('"opacity": "0.5"', examples)
        self.assertNotIn('"viewBox": "0 0 18 36"', examples)
        self.assertNotIn("10 5 40 40", prompt)
        self.assertNotIn("0 0 18 36", prompt)


if __name__ == "__main__":
    unittest.main()
