from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.run_spatial_grounding_holdout import (
    DEFAULT_SEED,
    HOLDOUT_PATCH_PROMPT_VERSION,
    POSITION_CENTERS,
    SIZE_LEVELS,
    SUITE_FORMAT,
    _manifest,
    build_cases,
    build_parser,
    paired_summary,
    reserve_output_root,
    validate_suite,
)
from svgpatchlab.core.patch import derive_patch
from svgpatchlab.core.scene import build_scene


class HoldoutDesignTests(unittest.TestCase):
    def test_default_suite_is_deterministic_and_expands_the_pilot(self):
        first = build_cases()
        second = build_cases()

        self.assertEqual(first, second)
        self.assertEqual(len(first), 39)
        self.assertEqual(
            sum(item.family == "position" for item in first),
            27,
        )
        self.assertEqual(
            sum(item.family == "size" for item in first),
            12,
        )
        self.assertEqual(
            {item.descriptor for item in first if item.family == "position"},
            set(POSITION_CENTERS),
        )
        self.assertEqual(
            {item.descriptor for item in first if item.family == "size"},
            set(SIZE_LEVELS),
        )
        self.assertTrue(
            {"top", "left", "center", "right", "bottom"}
            <= set(POSITION_CENTERS)
        )
        self.assertEqual(
            len({item.case.source_svg for item in first}),
            len(first),
        )
        self.assertEqual(
            len(
                {
                    item.dom_order
                    for item in first
                    if item.family == "position"
                }
            ),
            27,
        )
        self.assertEqual(
            len(
                {
                    item.dom_order
                    for item in first
                    if item.family == "size"
                }
            ),
            12,
        )
        self.assertEqual(
            Counter(
                item.target_node_id
                for item in first
                if item.family == "position"
            ),
            Counter({f"n{index}": 3 for index in range(1, 10)}),
        )
        self.assertEqual(
            Counter(
                item.target_node_id
                for item in first
                if item.family == "size"
            ),
            Counter({f"n{index}": 3 for index in range(1, 5)}),
        )

        different_seed = build_cases(DEFAULT_SEED + 1)
        self.assertNotEqual(
            [item.case.source_svg for item in first],
            [item.case.source_svg for item in different_seed],
        )

    def test_model_skeleton_keeps_path_candidates_intentionally_ambiguous(self):
        cases = build_cases()
        design = validate_suite(cases)

        self.assertEqual(design["cases"], 39)
        self.assertEqual(design["unique_source_svgs"], 39)
        self.assertTrue(design["one_unique_source_per_case"])
        self.assertTrue(design["deterministic_case_specific_dom_order"])
        self.assertTrue(design["unique_dom_order_within_each_family"])
        self.assertTrue(design["balanced_target_node_indices"])
        self.assertTrue(design["within_position_bin_jitter_verified"])
        self.assertTrue(design["relative_size_ranking_verified"])
        self.assertTrue(design["aspect_variation_present"])
        self.assertTrue(design["raw_path_coordinates_hidden_from_model"])
        self.assertTrue(design["protected_path_sha256_only"])
        self.assertTrue(design["equal_path_character_counts"])

        for source in {item.case.source_svg for item in cases}:
            scene = build_scene(source)
            candidates = scene["nodes"][1:]
            self.assertTrue(candidates)
            self.assertEqual({node["tag"] for node in candidates}, {"path"})
            self.assertEqual(
                {json.dumps(node["attributes"], sort_keys=True) for node in candidates},
                {'{"fill": "#ff0000"}'},
            )
            characters = {
                node["protected_geometry"]["d"]["characters"]
                for node in candidates
            }
            hashes = {
                node["protected_geometry"]["d"]["sha256"]
                for node in candidates
            }
            self.assertEqual(len(characters), 1)
            self.assertEqual(len(hashes), len(candidates))
            self.assertTrue(
                all("d" not in node["attributes"] for node in candidates)
            )

    def test_every_answer_has_exactly_the_declared_single_node_target(self):
        for item in build_cases():
            patch = derive_patch(item.case.source_svg, item.case.answer_svg)
            self.assertEqual(len(patch.operations), 1)
            operation = patch.operations[0]
            self.assertEqual(operation.op, "set_attributes")
            self.assertEqual(operation.targets, (item.target_node_id,))
            self.assertEqual(
                operation.attributes_dict,
                {"stroke": "#000000", "stroke-width": "2"},
            )

    def test_manifest_is_stable_and_contains_content_hashes(self):
        cases = build_cases()
        first = _manifest(cases, DEFAULT_SEED)
        second = _manifest(build_cases(), DEFAULT_SEED)

        self.assertEqual(first, second)
        self.assertEqual(first["format"], SUITE_FORMAT)
        self.assertEqual(first["seed"], DEFAULT_SEED)
        self.assertEqual(
            first["patch_prompt"]["version"],
            HOLDOUT_PATCH_PROMPT_VERSION,
        )
        self.assertEqual(
            first["patch_prompt"]["identifier"],
            "svgpatchlab.patch.v4",
        )
        self.assertEqual(
            len(first["patch_prompt"]["template_sha256"]),
            64,
        )
        self.assertEqual(first["case_count"], 39)
        self.assertEqual(first["unique_source_count"], 39)
        self.assertTrue(first["one_unique_source_per_case"])
        self.assertEqual(len(first["cases"]), 39)
        self.assertEqual(
            len({item["source_sha256"] for item in first["cases"]}),
            39,
        )
        self.assertTrue(
            all(len(item["source_sha256"]) == 64 for item in first["cases"])
        )
        self.assertTrue(
            all(len(item["answer_sha256"]) == 64 for item in first["cases"])
        )
        self.assertTrue(
            all(
                len(item["dom_order_sha256"]) == 64
                for item in first["cases"]
            )
        )

    def test_default_output_root_is_separate_from_pilot(self):
        args = build_parser().parse_args([])
        self.assertIn("holdout-v1", args.output_root)
        self.assertNotIn("probe-v1", args.output_root)

        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "already-there"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                reserve_output_root(existing)


def _record(
    case_id: str,
    family: str,
    *,
    target: bool,
    patch: bool,
    mse: float | None,
) -> dict:
    return {
        "case_id": case_id,
        "family": family,
        "target_exact": target,
        "metrics": {
            "gold_patch_exact": patch,
            "failure_aware_mse": mse,
        },
    }


class PairedSummaryTests(unittest.TestCase):
    def test_reports_directional_paired_outcomes_overall_and_by_family(self):
        baseline = [
            _record("a", "position", target=False, patch=False, mse=0.4),
            _record("b", "position", target=True, patch=True, mse=0.1),
            _record("c", "size", target=True, patch=False, mse=0.2),
            _record("d", "size", target=False, patch=False, mse=None),
        ]
        visual = [
            _record("a", "position", target=True, patch=True, mse=0.1),
            _record("b", "position", target=True, patch=False, mse=0.1),
            _record("c", "size", target=False, patch=True, mse=0.3),
            _record("d", "size", target=False, patch=False, mse=0.2),
        ]

        result = paired_summary(baseline, visual)

        target = result["overall"]["target_exact"]
        self.assertEqual(target["visual_only_wins"], 1)
        self.assertEqual(target["skeleton_only_wins"], 1)
        self.assertEqual(target["both_correct"], 1)
        self.assertEqual(target["neither_correct"], 1)

        patch = result["overall"]["gold_patch_exact"]
        self.assertEqual(patch["visual_only_wins"], 2)
        self.assertEqual(patch["skeleton_only_wins"], 1)
        self.assertEqual(patch["neither_correct"], 1)

        mse = result["overall"]["failure_aware_mse"]
        self.assertEqual(mse["visual_lower"], 1)
        self.assertEqual(mse["skeleton_lower"], 1)
        self.assertEqual(mse["ties"], 1)
        self.assertEqual(mse["missing_pairs"], 1)
        self.assertAlmostEqual(mse["mean_visual_minus_skeleton"], -0.2 / 3)
        self.assertEqual(result["by_family"]["position"]["cases"], 2)
        self.assertEqual(result["by_family"]["size"]["cases"], 2)

    def test_rejects_mismatched_cases(self):
        baseline = [
            _record("a", "position", target=False, patch=False, mse=0.4)
        ]
        visual = [
            _record("b", "position", target=True, patch=True, mse=0.1)
        ]

        with self.assertRaisesRegex(ValueError, "different cases"):
            paired_summary(baseline, visual)


if __name__ == "__main__":
    unittest.main()
