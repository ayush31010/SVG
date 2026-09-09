import copy
import unittest
from types import SimpleNamespace

from scripts.run_cloud_set_grounding import paired_sets, validate_cached
from scripts.analyze_cloud_cardinality_fusion import transfer_cardinality


class CloudSetGroundingTests(unittest.TestCase):
    def test_visual_fusion_uses_predicted_count_without_gold_labels(self):
        visual = [dict(case_id='one', ranking=['n3', 'n1', 'n2'])]
        scalar = [dict(case_id='one', predicted_cardinality=2)]
        result = transfer_cardinality(visual, scalar)
        self.assertEqual(result[0]['cardinality_targets'], ['n3', 'n1'])
        scalar[0]['gold_targets'] = ['n2']
        self.assertEqual(transfer_cardinality(visual, scalar), result)
        scalar[0]['predicted_cardinality'] = 0
        with self.assertRaises(ValueError):
            transfer_cardinality(visual, scalar)

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


class SeedReplicationTests(unittest.TestCase):
    def _seed(self, gnn, mlp, gnn_val=0.95, mlp_val=0.94):
        return {'gnn': dict(set_exact_rate=gnn, validation_exact=gnn_val),
                'mlp': dict(set_exact_rate=mlp, validation_exact=mlp_val)}

    def test_aggregate_counts_seed_level_direction_and_transfer_gap(self):
        from scripts.replicate_set_grounding_seeds import aggregate

        result = aggregate([self._seed(0.1, 0.3), self._seed(0.2, 0.4),
                            self._seed(0.25, 0.25)])
        self.assertEqual(result['mlp_better_seeds'], 2)
        self.assertEqual(result['gnn_better_seeds'], 0)
        self.assertEqual(result['tied_seeds'], 1)
        self.assertEqual(result['seed_level_sign_p'], 0.5)
        # A tie must not count for either arm, and the gap uses signed means.
        self.assertAlmostEqual(result['mean_natural_set_exact']['mlp'], 0.31666666, places=6)
        self.assertAlmostEqual(result['transfer_gap']['gnn'], 0.95 - 0.18333333, places=6)

    def test_aggregate_reports_no_significance_when_every_seed_ties(self):
        from scripts.replicate_set_grounding_seeds import aggregate

        result = aggregate([self._seed(0.3, 0.3)])
        self.assertIsNone(result['seed_level_sign_p'])


class SignFlipTests(unittest.TestCase):
    def test_signflip_matches_the_sign_test_when_magnitudes_are_equal(self):
        from scripts.replicate_set_grounding_seeds import sign_test, signflip_test

        # With identical magnitudes the permutation test reduces to the sign test.
        self.assertAlmostEqual(signflip_test([0.1] * 5), sign_test(5, 0))
        self.assertAlmostEqual(signflip_test([0.1] * 4 + [-0.1]), sign_test(4, 1))

    def test_signflip_keeps_magnitude_and_ignores_exact_ties(self):
        from scripts.replicate_set_grounding_seeds import signflip_test

        # One large reversal outweighs three small wins; the sign test cannot see this.
        self.assertGreater(signflip_test([0.01, 0.01, 0.01, -0.9]), 0.4)
        self.assertEqual(signflip_test([0.2, 0.0, 0.2]), signflip_test([0.2, 0.2]))
        self.assertIsNone(signflip_test([0.0, 0.0]))
