from __future__ import annotations

import unittest

from svgpatchlab.data import SVGEditBench


class DatasetTests(unittest.TestCase):
    def test_official_dataset_shape(self):
        summary = SVGEditBench("SVGEditBench").summary()
        self.assertEqual(summary["total_cases"], 600)
        self.assertEqual(summary["unique_emojis"], 100)
        self.assertEqual(set(summary["tasks"].values()), {100})

    def test_limit_per_task_samples_each_task(self):
        cases = list(SVGEditBench("SVGEditBench").iter_cases(limit_per_task=2))
        self.assertEqual(len(cases), 12)
        self.assertEqual({task: sum(case.task == task for case in cases) for task in SVGEditBench("SVGEditBench").tasks}, {task: 2 for task in SVGEditBench("SVGEditBench").tasks})


if __name__ == "__main__":
    unittest.main()
