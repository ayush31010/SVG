"""Focused tests for the measurement-disagreement analyzer."""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from scripts import analyze_occlusion_split as analyzer


def _record(case_id: str, task: str = "change_color") -> dict:
    return {
        "case_id": case_id,
        "task": task,
        "patch": {"operations": [{"targets": ["n1"]}]},
    }


class CaseCoverageTest(TestCase):
    def test_requires_exact_case_id_equality_across_every_arm(self):
        records = {
            "control": {"case-a": _record("case-a")},
            "analytic": {
                "case-a": _record("case-a"),
                "case-b": _record("case-b"),
            },
            "rendered": {"case-b": _record("case-b")},
        }
        cases = {
            case_id: SimpleNamespace(task="change_color")
            for case_id in ("case-a", "case-b")
        }
        gold = {"case-a": {"n1"}, "case-b": {"n1"}}

        with self.assertRaisesRegex(ValueError, "arm case-ID mismatch") as caught:
            analyzer._validated_case_ids(records, cases, gold)

        message = str(caught.exception)
        self.assertIn("case-a", message)
        self.assertIn("case-b", message)

    def test_requires_every_analyzed_id_in_dataset_and_gold_indexes(self):
        records = {
            "control": {
                "covered": _record("covered"),
                "missing-dataset": _record("missing-dataset"),
                "missing-gold": _record("missing-gold"),
            },
            "rendered": {
                "covered": _record("covered"),
                "missing-dataset": _record("missing-dataset"),
                "missing-gold": _record("missing-gold"),
            },
        }
        cases = {
            "covered": SimpleNamespace(task="change_color"),
            "missing-gold": SimpleNamespace(task="change_color"),
        }
        gold = {"covered": {"n1"}, "missing-dataset": {"n1"}}

        with self.assertRaisesRegex(
            ValueError, "analysis case coverage mismatch"
        ) as caught:
            analyzer._validated_case_ids(records, cases, gold)

        message = str(caught.exception)
        self.assertIn("missing-dataset", message)
        self.assertIn("missing-gold", message)

    def test_subset_runs_are_valid_and_returned_in_stable_order(self):
        records = {
            "left": {
                "z": _record("z"),
                "a": _record("a"),
            },
            "right": {
                "a": _record("a"),
                "z": _record("z"),
            },
        }
        cases = {
            case_id: SimpleNamespace(task="change_color")
            for case_id in ("a", "omitted", "z")
        }
        gold = {case_id: {"n1"} for case_id in cases}

        self.assertEqual(
            analyzer._validated_case_ids(records, cases, gold),
            ["a", "z"],
        )


class MeasurementDisagreementTest(TestCase):
    def test_area_gap_is_symmetric_and_never_described_as_hidden_fraction(self):
        analytic = {
            "smaller-render": {"area_pct": 10.0},
            "larger-render": {"area_pct": 10.0},
            "no-contribution": {"area_pct": 10.0},
            "missing-render": {"area_pct": 10.0},
            "zero-analytic": {"area_pct": 0.0},
        }
        rendered = {
            "smaller-render": {"area_pct": 5.0},
            "larger-render": {"area_pct": 20.0},
            "no-contribution": {"visible": False},
            "zero-analytic": {"area_pct": 5.0},
        }

        scores = analyzer.area_measurement_disagreement_scores_from_stats(
            analytic,
            rendered,
        )

        self.assertEqual(scores["smaller-render"], 0.5)
        self.assertEqual(scores["larger-render"], 0.5)
        self.assertEqual(scores["no-contribution"], 1.0)
        self.assertNotIn("missing-render", scores)
        self.assertNotIn("zero-analytic", scores)

    def test_legacy_function_warns_that_it_does_not_measure_occlusion(self):
        with (
            patch.object(
                analyzer,
                "area_measurement_disagreement_scores",
                return_value={"n1": 0.25},
            ),
            self.assertWarnsRegex(DeprecationWarning, "do not prove occlusion"),
        ):
            result = analyzer.occlusion_scores("<svg/>", {})

        self.assertEqual(result, {"n1": 0.25})


class PositionDescriptorAuditTest(TestCase):
    def test_audit_is_reproducible_and_does_not_claim_occlusion(self):
        cases = {
            "z-case": SimpleNamespace(task="change_color"),
            "a-case": SimpleNamespace(task="compression"),
        }
        gold = {
            "z-case": {"n2", "n1"},
            "a-case": {"n3"},
        }
        analytic = {
            "z-case": {
                "n1": {"position": "left"},
                "n2": {"position": "center"},
            },
            "a-case": {"n3": {"position": "top"}},
        }
        rendered = {
            "z-case": {
                "n1": {"position": "right", "area_pct": 2.0},
                "n2": {"visible": False},
            },
            "a-case": {"n3": {"position": "top", "area_pct": 1.0}},
        }
        scores = {
            "z-case": {"n1": 0.5, "n2": 1.0},
            "a-case": {"n3": 0.0},
        }

        audit = analyzer.position_descriptor_disagreement_audit(
            ["z-case", "a-case"],
            cases,
            gold,
            analytic,
            rendered,
            scores,
            render_size=96,
        )

        ordered_keys = [
            (row["case_id"], row["target_id"])
            for row in audit["observations"]
        ]
        self.assertEqual(
            ordered_keys,
            [("a-case", "n3"), ("z-case", "n1"), ("z-case", "n2")],
        )
        self.assertEqual(
            audit["summary"]["position_descriptor_disagreements"],
            1,
        )
        self.assertEqual(
            audit["summary"]["position_descriptor_agreements"],
            1,
        )
        self.assertEqual(
            audit["summary"]["no_rendered_visible_contribution"],
            1,
        )
        self.assertEqual(audit["summary"]["proven_occlusions"], 0)
        self.assertTrue(
            all(not row["occlusion_proven"] for row in audit["observations"])
        )
        self.assertIn("does not establish", audit["interpretation"])

    def test_grounding_groups_use_dataset_task_not_case_id_prefix(self):
        cases = {
            "misleading-prefix/one": SimpleNamespace(task="change_color"),
            "change_color/two": SimpleNamespace(task="compression"),
        }
        classifications = {
            "misleading-prefix/one": "high",
            "change_color/two": "high",
        }

        groups = analyzer._groups(
            ["change_color/two", "misleading-prefix/one"],
            cases,
            classifications,
        )

        self.assertEqual(
            groups["grounding_high_area_disagreement"],
            ["misleading-prefix/one"],
        )


class AnalyzeIntegrationTest(TestCase):
    def test_report_names_disagreement_and_records_subset_coverage(self):
        analyzed = SimpleNamespace(
            case_id="not-a-task-prefix/case",
            task="change_color",
            source_svg="<svg id='analyzed'/>",
        )
        omitted = SimpleNamespace(
            case_id="compression/omitted",
            task="compression",
            source_svg="<svg id='omitted'/>",
        )
        records = {
            arm: {
                analyzed.case_id: _record(analyzed.case_id, analyzed.task)
            }
            for arm in ("control", "analytic", "rendered")
        }

        class FakeDataset:
            def __init__(self, _root):
                pass

            def iter_cases(self):
                return iter((analyzed, omitted))

        class FakeCache:
            def get_or_compute(self, _svg, size):
                self.size = size
                return {
                    "n1": {
                        "area_pct": 8.0,
                        "position": "right",
                    }
                }

        def read_records(path: Path):
            return records[path.parent.name]

        with (
            patch.object(analyzer, "SVGEditBench", FakeDataset),
            patch.object(
                analyzer,
                "_gold_targets",
                return_value={
                    analyzed.case_id: {"n1"},
                    omitted.case_id: {"n1"},
                },
            ),
            patch.object(analyzer, "_read_records", side_effect=read_records),
            patch.object(
                analyzer,
                "node_analytic_stats",
                return_value={
                    "n1": {
                        "area_pct": 10.0,
                        "position": "left",
                    }
                },
            ),
        ):
            report = analyzer.analyze(
                Path("unused"),
                "unused-dataset",
                ("control", "analytic", "rendered"),
                FakeCache(),
                render_size=80,
                area_disagreement_threshold=0.25,
            )

        self.assertEqual(
            report["format"],
            "svgpatchlab.measurement_disagreement_analysis.v1",
        )
        self.assertEqual(
            report["coverage"]["dataset_case_ids_not_analyzed"],
            [omitted.case_id],
        )
        self.assertEqual(
            report["groups"]["grounding_low_area_disagreement"]["cases"],
            1,
        )
        self.assertNotIn("occluded_target", report["groups"])
        observation = report["position_descriptor_audit"]["observations"][0]
        self.assertTrue(observation["position_descriptor_disagreement"])
        self.assertFalse(observation["occlusion_proven"])
