from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from svgpatchlab.eval.runner import run_evaluation
from svgpatchlab.core import derive_patch
from svgpatchlab.data import SVGEditBench


class RunnerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
