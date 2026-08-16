from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_node_grounding_runs import (
    assert_matched,
    build_comparison,
    identity_digest,
    two_sided_sign_test,
)
from train.node_grounding_sft import GroundingDataError


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["choices"],
    "properties": {
        "choices": {
            "type": "array",
            "items": {"type": "string", "enum": ["A", "B"]},
            "minItems": 1,
            "maxItems": 2,
            "uniqueItems": True,
        }
    },
}


def _row(base_id: str, permutation: int, gold: str, prediction: str) -> dict:
    return {
        "ablation": "none",
        "id": f"{base_id}@p{permutation:03d}",
        "base_id": base_id,
        "choice_to_node": {"A": "n1", "B": "n2"},
        "gold": json.dumps({"choices": [gold]}),
        "target_choices": [gold],
        "source_family": "synthetic_context",
        "image": f"images/test/{base_id}.png",
        "instruction": "Recolor the left eye.",
        "label_permutation_index": permutation,
        "max_selections": 2,
        "response_schema": SCHEMA,
        "source_sha256": f"sha-{base_id}",
        "prediction": prediction,
    }


def _write(directory: Path, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "test_predictions.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class SignTestTests(unittest.TestCase):
    def test_matches_published_context_v3_p_values(self):
        # 19 improvements / 0 regressions and 50 / 0 are the two published
        # discordant splits; both must come back bit-identical.
        self.assertAlmostEqual(two_sided_sign_test(19, 0), 3.814697265625e-06, places=18)
        self.assertAlmostEqual(two_sided_sign_test(50, 0), 1.7763568394002505e-15, places=20)

    def test_is_symmetric_and_bounded(self):
        self.assertEqual(two_sided_sign_test(0, 0), 1.0)
        self.assertEqual(two_sided_sign_test(7, 3), two_sided_sign_test(3, 7))
        self.assertEqual(two_sided_sign_test(5, 5), 1.0)
        for wins in range(0, 12):
            for losses in range(0, 12):
                self.assertLessEqual(two_sided_sign_test(wins, losses), 1.0)


class MatchingTests(unittest.TestCase):
    def test_rejects_runs_whose_inputs_differ(self):
        left = [_row("case-a", 0, "A", '{"choices":["A"]}')]
        right = [_row("case-a", 0, "B", '{"choices":["A"]}')]
        with self.assertRaisesRegex(GroundingDataError, "not comparable"):
            assert_matched(left, right)

    def test_rejects_differing_row_counts(self):
        left = [_row("case-a", 0, "A", '{"choices":["A"]}')]
        right = left + [_row("case-b", 0, "A", '{"choices":["A"]}')]
        with self.assertRaisesRegex(GroundingDataError, "row counts differ"):
            assert_matched(left, right)

    def test_requires_identity_fields(self):
        stripped = _row("case-a", 0, "A", '{"choices":["A"]}')
        del stripped["source_sha256"]
        with self.assertRaisesRegex(GroundingDataError, "identity fields"):
            assert_matched([stripped], [stripped])

    def test_digest_ignores_predictions_but_tracks_inputs(self):
        correct = _row("case-a", 0, "A", '{"choices":["A"]}')
        wrong = dict(correct, prediction='{"choices":["B"]}')
        self.assertEqual(identity_digest([correct]), identity_digest([wrong]))
        moved = dict(correct, instruction="Recolor the right eye.")
        self.assertNotEqual(identity_digest([correct]), identity_digest([moved]))


class BuildComparisonTests(unittest.TestCase):
    def test_paired_counts_use_all_permutations_per_base_case(self):
        # case-a: trained fixes both permutations.  case-b: trained is right on
        # one permutation only, so the base case still does not count.
        base_rows = [
            _row("case-a", 0, "A", '{"choices":["B"]}'),
            _row("case-a", 1, "A", '{"choices":["B"]}'),
            _row("case-b", 0, "A", '{"choices":["B"]}'),
            _row("case-b", 1, "A", '{"choices":["B"]}'),
        ]
        trained_rows = [
            _row("case-a", 0, "A", '{"choices":["A"]}'),
            _row("case-a", 1, "A", '{"choices":["A"]}'),
            _row("case-b", 0, "A", '{"choices":["A"]}'),
            _row("case-b", 1, "A", '{"choices":["B"]}'),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write(root / "base", base_rows)
            _write(root / "trained", trained_rows)
            comparison = build_comparison(root / "base", root / "trained")

        self.assertEqual(comparison["evaluation_definition"]["rows"], 4)
        self.assertEqual(comparison["evaluation_definition"]["base_cases"], 2)
        self.assertIsNone(comparison["evaluation_definition"]["max_eval_samples"])
        self.assertEqual(comparison["base"]["exact_set_match"], 0)
        self.assertEqual(comparison["trained"]["exact_set_match"], 3)

        cases = comparison["paired"]["base_case_all_permutations_exact"]
        self.assertEqual(cases["trained_only_correct"], 1)
        self.assertEqual(cases["base_only_correct"], 0)
        self.assertEqual(cases["neither_correct"], 1)

    def test_unparseable_prediction_counts_as_wrong_not_crash(self):
        base_rows = [_row("case-a", 0, "A", "I think the answer is A.")]
        trained_rows = [_row("case-a", 0, "A", '{"choices":["A"]}')]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write(root / "base", base_rows)
            _write(root / "trained", trained_rows)
            comparison = build_comparison(root / "base", root / "trained")

        self.assertEqual(comparison["base"]["valid_json"], 0)
        self.assertEqual(comparison["base"]["exact_set_match"], 0)
        self.assertEqual(comparison["trained"]["exact_set_match"], 1)
        self.assertEqual(
            comparison["paired"]["row_exact_set"]["trained_only_correct"], 1
        )


if __name__ == "__main__":
    unittest.main()
