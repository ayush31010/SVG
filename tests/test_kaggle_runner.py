from __future__ import annotations

import unittest

from scripts.run_kaggle_plans import PLAN_A_ARCHITECTURES, build_parser


class KaggleRunnerTests(unittest.TestCase):
    def test_default_run_has_no_case_limit(self):
        args = build_parser().parse_args([])

        self.assertIsNone(args.limit)
        self.assertIsNone(args.limit_per_task)

    def test_plan_a_includes_clean_visual_stats_baseline(self):
        self.assertIn("skeleton_patch", PLAN_A_ARCHITECTURES)
        self.assertIn("visual_stats_patch", PLAN_A_ARCHITECTURES)
        self.assertLess(
            PLAN_A_ARCHITECTURES.index("skeleton_patch"),
            PLAN_A_ARCHITECTURES.index("visual_stats_patch"),
        )


if __name__ == "__main__":
    unittest.main()
