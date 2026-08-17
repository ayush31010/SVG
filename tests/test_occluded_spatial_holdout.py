from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.run_occluded_spatial_holdout import (
    ARMS,
    DEFAULT_PER_FAMILY,
    DEFAULT_SEED,
    EVALUATION_RENDER_SIZE,
    HOLDOUT_PATCH_PROMPT_VERSION,
    SUITE_FORMAT,
    VISUAL_STATS_RENDER_SIZE,
    AuditRecordingModelAdapter,
    _overlap_area,
    build_cases,
    build_manifest,
    build_parser,
    paired_summary,
    reserve_output_root,
    run_arm,
    validate_static_suite,
    verify_construction,
)
from svgpatchlab.core.patch import derive_patch
from svgpatchlab.core.scene import build_scene
from svgpatchlab.models.base import ModelAdapter
from svgpatchlab.types import (
    ArchitectureResult,
    ModelRequest,
    ModelResponse,
)


class OccludedHoldoutDesignTests(unittest.TestCase):
    def test_frozen_suite_is_deterministic_unique_and_balanced(self):
        first = build_cases()
        second = build_cases()

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2 * DEFAULT_PER_FAMILY)
        self.assertEqual(
            Counter(item.family for item in first),
            Counter({"occluded": 24, "clean": 24}),
        )
        self.assertEqual(
            len({item.case.source_svg for item in first}),
            len(first),
        )
        self.assertEqual(
            len({item.case.case_id for item in first}),
            len(first),
        )

        expected = Counter({f"n{index}": 6 for index in range(1, 5)})
        for family in ("occluded", "clean"):
            rows = [item for item in first if item.family == family]
            self.assertEqual(
                Counter(item.target_node_id for item in rows),
                expected,
            )
        self.assertEqual(
            Counter(
                item.decoy_node_id
                for item in first
                if item.family == "occluded"
            ),
            expected,
        )

        for family in ("occluded", "clean"):
            rows = [item for item in first if item.family == family]
            for named_cell in {item.named_cell for item in rows}:
                label_rows = [
                    item for item in rows if item.named_cell == named_cell
                ]
                self.assertEqual(len(label_rows), 4)
                self.assertEqual(
                    {item.target_node_id for item in label_rows},
                    {"n1", "n2", "n3", "n4"},
                )
                if family == "occluded":
                    self.assertEqual(
                        {item.decoy_node_id for item in label_rows},
                        {"n1", "n2", "n3", "n4"},
                    )

        different_seed = build_cases(DEFAULT_SEED + 1)
        self.assertNotEqual(
            [item.case.source_svg for item in first],
            [item.case.source_svg for item in different_seed],
        )

    def test_frozen_suite_rejects_a_different_case_count(self):
        with self.assertRaisesRegex(ValueError, "requires --per-family 24"):
            build_cases(per_family=12)

    def test_static_validator_closes_geometry_leaks_and_overlap(self):
        cases = build_cases()
        design = validate_static_suite(cases)

        self.assertEqual(design["cases"], 48)
        self.assertTrue(design["one_unique_source_per_case"])
        self.assertTrue(design["fixed_width_coordinates"])
        self.assertTrue(design["equal_red_path_character_counts"])
        self.assertTrue(design["equal_occluder_path_character_counts"])
        self.assertTrue(design["balanced_target_node_ids"])
        self.assertTrue(design["balanced_decoy_node_ids"])
        self.assertTrue(design["balanced_ids_within_each_label"])
        self.assertTrue(design["candidate_pairwise_non_overlap"])
        self.assertTrue(design["clean_pairwise_non_overlap"])
        self.assertTrue(design["clean_has_no_occluder"])
        self.assertEqual(design["red_path_characters"], 21)
        self.assertEqual(design["occluder_path_characters"], 42)

        red_lengths: set[int] = set()
        occluder_lengths: set[int] = set()
        for item in cases:
            for left_index, left in enumerate(item.candidate_boxes):
                for right in item.candidate_boxes[left_index + 1 :]:
                    self.assertEqual(_overlap_area(left, right), 0)

            scene = build_scene(item.case.source_svg)
            candidates = [
                node
                for node in scene["nodes"]
                if node["id"] in {"n1", "n2", "n3", "n4"}
            ]
            self.assertEqual(len(candidates), 4)
            for node in candidates:
                self.assertEqual(node["attributes"], {"fill": "#d32f2f"})
                self.assertNotIn("d", node["attributes"])
                protected = node["protected_geometry"]["d"]
                self.assertEqual(
                    set(protected),
                    {"sha256", "characters"},
                )
                red_lengths.add(protected["characters"])

            if item.family == "occluded":
                occluder = next(
                    node
                    for node in scene["nodes"]
                    if node["id"] == "n5"
                )
                occluder_lengths.add(
                    occluder["protected_geometry"]["d"]["characters"]
                )
            else:
                self.assertEqual(len(scene["nodes"]), 5)

        self.assertEqual(red_lengths, {21})
        self.assertEqual(occluder_lengths, {42})

    def test_every_answer_is_the_declared_single_target_patch(self):
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

    def test_renderer_verifier_requires_the_unique_named_candidate(self):
        cases = build_cases()
        by_source = {item.case.source_svg: item for item in cases}

        def analytic(svg: str):
            item = by_source[svg]
            stats = {
                f"n{index}": {"position": "center"}
                for index in range(1, 5)
            }
            expected = (
                item.decoy_node_id
                if item.family == "occluded"
                else item.target_node_id
            )
            stats[expected]["position"] = item.named_cell
            return stats

        def rendered(svg: str):
            item = by_source[svg]
            stats = {
                f"n{index}": {"position": "center", "visible": True}
                for index in range(1, 5)
            }
            stats[item.target_node_id]["position"] = item.named_cell
            if item.decoy_node_id is not None:
                stats[item.decoy_node_id] = {"visible": False}
            return stats

        report = verify_construction(
            cases,
            analytic_provider=analytic,
            visual_provider=rendered,
        )
        self.assertEqual(report["occluded_verified"], 24)
        self.assertEqual(report["clean_verified"], 24)
        self.assertEqual(report["problems"], [])

        broken_source = next(
            item.case.source_svg for item in cases if item.family == "occluded"
        )

        def broken_rendered(svg: str):
            stats = rendered(svg)
            item = by_source[svg]
            if svg == broken_source:
                stats[item.decoy_node_id] = {
                    "position": item.named_cell,
                    "visible": True,
                }
            return stats

        broken = verify_construction(
            cases,
            analytic_provider=analytic,
            visual_provider=broken_rendered,
        )
        self.assertEqual(len(broken["problems"]), 1)
        self.assertEqual(broken["occluded_verified"], 23)

    def test_manifest_freezes_model_prompt_render_and_case_provenance(self):
        cases = build_cases()
        design = validate_static_suite(cases)
        verification = {
            "occluded_verified": 24,
            "clean_verified": 24,
            "problems": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            model_payload = {
                "adapter": "openai_compatible",
                "model": "test-model",
                "api_key": "must-not-leak",
            }
            model_path.write_text(json.dumps(model_payload), encoding="utf-8")
            manifest = build_manifest(
                cases,
                seed=DEFAULT_SEED,
                model_config_path=model_path,
                model_config=model_payload,
                arms=ARMS,
                design=design,
                verification=verification,
                created_at="2026-07-29T00:00:00+00:00",
            )

        self.assertEqual(manifest["format"], SUITE_FORMAT)
        self.assertEqual(manifest["seed"], DEFAULT_SEED)
        self.assertEqual(manifest["arm_order"], list(ARMS))
        self.assertEqual(manifest["case_count"], 48)
        self.assertEqual(manifest["unique_source_count"], 48)
        self.assertEqual(
            manifest["model_config"]["sha256"],
            hashlib.sha256(json.dumps(model_payload).encode()).hexdigest(),
        )
        self.assertEqual(
            manifest["model_config"]["resolved_redacted"]["api_key"],
            "<redacted>",
        )
        self.assertEqual(
            manifest["patch_prompt"]["version"],
            HOLDOUT_PATCH_PROMPT_VERSION,
        )
        self.assertEqual(
            manifest["rendering"]["visual_stats_render_size"],
            VISUAL_STATS_RENDER_SIZE,
        )
        self.assertEqual(
            manifest["rendering"]["evaluation_render_size"],
            EVALUATION_RENDER_SIZE,
        )
        self.assertEqual(len(manifest["cases"]), 48)
        self.assertEqual(
            len({item["source_sha256"] for item in manifest["cases"]}),
            48,
        )
        self.assertTrue(
            all(item["source_svg"] for item in manifest["cases"])
        )
        self.assertTrue(
            all(item["answer_svg"] for item in manifest["cases"])
        )
        self.assertTrue(
            all(item["gold_patch"] for item in manifest["cases"])
        )

    def test_default_root_is_v2_and_existing_roots_are_immutable(self):
        args = build_parser().parse_args([])
        self.assertIn("holdout-v2", args.output_root)
        self.assertNotIn("holdout-v1", args.output_root)
        self.assertFalse(args.verify_only)
        self.assertEqual(args.per_family, 24)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh"
            reserve_output_root(root)
            self.assertTrue(root.is_dir())
            with self.assertRaises(FileExistsError):
                reserve_output_root(root)


class _FakeModel(ModelAdapter):
    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            text='{"version":1,"operations":[]}',
            metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 4}},
        )


class _FakeArchitecture:
    def run(
        self,
        case,
        model: ModelAdapter,
    ) -> ArchitectureResult:
        response = model.generate(
            ModelRequest(
                prompt=f"audit prompt for {case.case_id}",
                metadata={"request_id": case.case_id, "task": case.task},
                response_schema={"type": "object"},
                response_schema_name="svg_patch",
            )
        )
        patch = derive_patch(case.source_svg, case.answer_svg)
        return ArchitectureResult(
            output_svg=case.answer_svg,
            patch=patch,
            raw_responses=[response.text],
            details={"audit": True},
            model_calls=1,
        )


class OccludedHoldoutAuditTests(unittest.TestCase):
    def test_audit_adapter_keeps_reconstructable_call_metadata(self):
        adapter = AuditRecordingModelAdapter(_FakeModel())
        response = adapter.generate(
            ModelRequest(
                prompt="raw prompt",
                metadata={"request_id": "case-1", "custom": 7},
                response_schema={"type": "object"},
                response_schema_name="patch",
            )
        )
        self.assertTrue(response.text)
        self.assertEqual(len(adapter.records), 1)
        record = adapter.records[0]
        self.assertEqual(record["prompt"], "raw prompt")
        self.assertEqual(record["request_metadata"]["custom"], 7)
        self.assertEqual(record["response_schema"], {"type": "object"})
        self.assertEqual(record["response_schema_name"], "patch")
        self.assertTrue(record["ok"])
        self.assertEqual(
            record["response_metadata"]["usage"]["prompt_tokens"],
            10,
        )

    def test_run_arm_streams_raw_audit_fields_and_distinct_exact_metrics(self):
        cases = build_cases()[:2]

        def evaluator(source, answer, output, patch, **kwargs):
            self.assertEqual(kwargs["render_size"], EVALUATION_RENDER_SIZE)
            return {
                "valid_output": True,
                "gold_patch_exact": True,
                "failure_aware_mse": 0.0,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            reserve_output_root(root)
            records, summary = run_arm(
                "audit_arm",
                cases,
                {"adapter": "fake"},
                root,
                architecture_factory=lambda name, output: _FakeArchitecture(),
                model_factory=lambda config: _FakeModel(),
                evaluator=evaluator,
            )
            result_path = root / "audit_arm" / "results.jsonl"
            persisted = [
                json.loads(line)
                for line in result_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(records), 2)
            self.assertEqual(len(persisted), 2)
            for record in persisted:
                self.assertTrue(record["target_exact"])
                self.assertTrue(record["gold_patch_exact"])
                self.assertIsInstance(record["patch"], dict)
                self.assertIsInstance(record["gold_patch"], dict)
                self.assertTrue(record["raw_responses"])
                self.assertEqual(record["architecture_details"], {"audit": True})
                self.assertEqual(record["metrics"]["gold_patch_exact"], True)
                self.assertEqual(len(record["model_call_details"]), 1)
                self.assertIn(
                    "audit prompt",
                    record["model_call_details"][0]["prompt"],
                )
                self.assertTrue(record["output_svg"])
                self.assertTrue(record["output_svg_sha256"])
                self.assertTrue(
                    (
                        root
                        / "audit_arm"
                        / record["output_svg_path"]
                    ).is_file()
                )

            self.assertEqual(
                summary["overall"]["target_exact_rate"],
                1.0,
            )
            self.assertEqual(
                summary["overall"]["gold_patch_exact_rate"],
                1.0,
            )
            self.assertTrue((root / "audit_arm" / "summary.json").is_file())

    def test_paired_summary_names_both_metrics_and_exact_sign_test(self):
        left = [
            {
                "case_id": "a",
                "family": "occluded",
                "target_exact": False,
                "gold_patch_exact": False,
            },
            {
                "case_id": "b",
                "family": "occluded",
                "target_exact": False,
                "gold_patch_exact": True,
            },
        ]
        right = [
            {
                "case_id": "a",
                "family": "occluded",
                "target_exact": True,
                "gold_patch_exact": True,
            },
            {
                "case_id": "b",
                "family": "occluded",
                "target_exact": False,
                "gold_patch_exact": False,
            },
        ]
        result = paired_summary(
            left,
            right,
            left_arm="analytic",
            right_arm="rendered",
        )
        overall = result["overall"]
        self.assertEqual(overall["left_arm"], "analytic")
        self.assertEqual(overall["right_arm"], "rendered")
        self.assertEqual(
            overall["target_exact"]["right_only_wins"],
            1,
        )
        self.assertEqual(
            overall["gold_patch_exact"]["left_only_wins"],
            1,
        )
        self.assertEqual(
            overall["gold_patch_exact"]["right_only_wins"],
            1,
        )
        self.assertEqual(
            overall["gold_patch_exact"]["two_sided_sign_test_p"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
