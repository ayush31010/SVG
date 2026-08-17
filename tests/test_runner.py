from __future__ import annotations

import tempfile
import unittest
import json
import contextlib
import io
from pathlib import Path
from unittest import mock

from svgpatchlab.cli import main as cli_main
from svgpatchlab.architectures.factory import ARCHITECTURES
from svgpatchlab.eval.runner import run_evaluation
from svgpatchlab.core import derive_patch
from svgpatchlab.data import SVGEditBench
from svgpatchlab.types import ArchitectureResult


class RunnerTests(unittest.TestCase):
    def test_oracle_smoke_run_without_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "dataset": {"root": "SVGEditBench", "limit": 12},
                "architecture": {"name": "oracle_patch"},
                "model": {"adapter": "none"},
                "evaluation": {
                    "render": False,
                    "output_dir": directory,
                    "save_outputs": True,
                },
            }
            summary = run_evaluation(config)
            self.assertEqual(summary["overall"]["cases"], 12)
            self.assertEqual(summary["overall"]["valid_output_rate"], 1.0)
            self.assertEqual(summary["overall"]["reference_structure_match_rate"], 1.0)
            self.assertEqual(summary["overall"]["protected_geometry_rate"], 1.0)
            self.assertTrue((Path(directory) / "summary.json").exists())

    def test_replay_adapter_drives_skeleton_architecture(self):
        case = next(SVGEditBench("SVGEditBench").iter_cases(tasks=["change_color"]))
        patch = derive_patch(case.source_svg, case.answer_svg)
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "responses.jsonl"
            replay.write_text(
                json.dumps({"request_id": case.case_id, "response": patch.to_json()}) + "\n"
            )
            summary = run_evaluation(
                {
                    "dataset": {
                        "root": "SVGEditBench",
                        "tasks": ["change_color"],
                        "limit": 1,
                    },
                    "architecture": {"name": "skeleton_patch"},
                    "model": {"adapter": "replay", "path": str(replay)},
                    "evaluation": {"render": False, "output_dir": directory},
                }
            )
            self.assertEqual(summary["overall"]["gold_patch_exact_rate"], 1.0)
            self.assertEqual(summary["overall"]["protected_geometry_rate"], 1.0)

    def test_chain_cli_writes_plan_c_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_config = root / "none-model.json"
            model_config.write_text(json.dumps({"adapter": "none"}))
            chain_config = root / "chain.json"
            output_dir = root / "chain-output"
            chain_config.write_text(
                json.dumps(
                    {
                        "decomposer": {
                            "model_config": str(model_config),
                            "max_steps": 1,
                        },
                        "patch_architecture": "skeleton_patch",
                        "patch_model_config": str(model_config),
                        "dataset": {"root": "SVGEditBench", "limit": 1},
                        "evaluation": {"render": False, "output_dir": str(output_dir)},
                    }
                )
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = cli_main(["chain", "--config", str(chain_config)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "chain_summary.json").exists())
            self.assertTrue((output_dir / "chain_results.jsonl").exists())

    def test_architecture_configuration_fields_reach_constructor(self):
        captured = {}

        class ConfiguredArchitecture:
            name = "configured_test"
            requires_model = False
            requires_renderer = False

            def __init__(self, marker):
                captured["marker"] = marker

            def run(self, case, model):
                patch = derive_patch(case.source_svg, case.answer_svg)
                return ArchitectureResult(
                    output_svg=case.answer_svg,
                    patch=patch,
                    details={"captured_context": True},
                )

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            ARCHITECTURES,
            {"configured_test": ConfiguredArchitecture},
        ):
            summary = run_evaluation(
                {
                    "dataset": {
                        "root": "SVGEditBench",
                        "tasks": ["change_color"],
                        "limit": 1,
                    },
                    "architecture": {
                        "name": "configured_test",
                        "marker": "passed-through",
                    },
                    "model": {"adapter": "none"},
                    "evaluation": {
                        "render": False,
                        "output_dir": directory,
                    },
                }
            )
            record = json.loads((Path(directory) / "results.jsonl").read_text())

        self.assertEqual(captured["marker"], "passed-through")
        self.assertEqual(summary["overall"]["cases"], 1)
        self.assertEqual(record["architecture_details"], {"captured_context": True})


if __name__ == "__main__":
    unittest.main()
