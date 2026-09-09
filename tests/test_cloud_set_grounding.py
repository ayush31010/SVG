import copy
import unittest
from types import SimpleNamespace

from scripts.run_cloud_set_grounding import paired_sets, validate_cached


class CloudSetGroundingTests(unittest.TestCase):
    def test_cached_visual_scores_reject_stale_or_duplicate_cases(self):
        case = SimpleNamespace(case_id='one', instruction='the red shape',
                               candidate_ids=('n1', 'n2'), gold_target_ids=('n1',))
        record = dict(case_id='one', instruction=case.instruction,
                      candidate_ids=['n1', 'n2'], gold_targets=['n1'], error=None)
        validate_cached([case], [record])
        for field, value in [('instruction', 'the blue shape'),
                             ('candidate_ids', ['n2', 'n1']),
                             ('gold_targets', ['n2']), ('error', 'failed')]:
            changed = copy.deepcopy(record)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_cached([case], [changed])
        with self.assertRaises(ValueError):
            validate_cached([case], [record, record])

    def test_paired_test_uses_deployed_sets_and_requires_same_cases(self):
        left = [dict(case_id=str(i), set_exact=True) for i in range(6)]
        right = [dict(case_id=str(i), set_exact=False) for i in range(6)]
        result = paired_sets(left, right)
        self.assertEqual(result['left_only_correct'], 6)
        self.assertEqual(result['right_only_correct'], 0)
        self.assertEqual(result['exact_mcnemar_p'], 0.03125)
        with self.assertRaises(ValueError):
            paired_sets(left, right[:-1])
