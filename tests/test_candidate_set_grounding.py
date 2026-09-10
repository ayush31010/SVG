import unittest

from scripts.train_candidate_set_grounding import candidate_sets, categorize, training_targets, source_bootstrap, drawable_ids


class CandidateSetTests(unittest.TestCase):
    def test_training_uses_drawables_without_container_nodes(self):
        from train.graph_moe_grounding import _set_cases
        from train.node_grounding_sft import gold_target_ids
        case = _set_cases(2, 20260909)[1]
        ids = drawable_ids(case.source_svg)
        gold = gold_target_ids(case.source_svg, case.answer_svg)
        groups = candidate_sets(case.source_svg, ids, dict.fromkeys(ids, .5), 'groups')
        self.assertIn(tuple(gold), groups)
        # Four cluster containers are graph features, never editable candidates.
        self.assertNotIn('n1', ids)
        self.assertTrue(all(set(c) <= set(ids) for c in groups))

    def test_structural_groups_add_source_subtree_missing_from_score_prefixes(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><g><circle id="a" cx="10" cy="10" r="2"/><rect id="b" x="12" y="12" width="3" height="3"/></g><circle id="c" cx="70" cy="70" r="3"/></svg>'
        ids, scores = ['n2', 'n3', 'n4'], {'n2': .9, 'n4': .8, 'n3': .1}
        prefix = candidate_sets(svg, ids, scores, 'prefix')
        groups = candidate_sets(svg, ids, scores, 'groups')
        self.assertNotIn(('n2', 'n3'), prefix)
        self.assertIn(('n2', 'n3'), groups)
        self.assertTrue(set(prefix) <= set(groups))
        self.assertEqual(len(groups), len(set(groups)))
        self.assertTrue(all(set(c) <= set(ids) for c in groups))

    def test_supervision_handles_unrepresentable_sets_without_oracle_injection(self):
        candidates = [('a',), ('b',), ('a', 'c')]
        targets = training_targets(candidates, ['a', 'b'])
        self.assertAlmostEqual(float(targets.sum()), 1)
        self.assertEqual(list(targets), [.5, .5, 0])
        self.assertNotIn(('a', 'b'), candidates)

    def test_failure_categories_are_disjoint(self):
        gold = ['a', 'b']
        expected = [(['a', 'b'], 'exact'), (['a'], 'missing_parts'),
                    (['a', 'b', 'c'], 'extra_parts'), (['c'], 'wrong_object'),
                    (['a', 'c'], 'mixed_missing_and_extra')]
        for selected, category in expected:
            self.assertEqual(categorize(selected, gold), category)

    def test_uncertainty_resamples_sources_with_both_instructions(self):
        left = [dict(case_id=str(i), source_id=str(i//2), set_exact=bool(i % 2))
                for i in range(6)]
        right = [{**r, 'set_exact': not r['set_exact']} for r in left]
        result = source_bootstrap(left, right, draws=100)
        # Every source ties even though individual instructions differ.
        self.assertEqual(result['sources'], 3)
        self.assertEqual(result['percentile_95_interval'], [0, 0])
        with self.assertRaises(ValueError):
            source_bootstrap(left, right[:-1], draws=10)

    def test_frozen_holdout_and_checkpoint_match_declared_hashes(self):
        import hashlib
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'configs/train/candidate_set_v1.json').read_text())
        for path_key, hash_key in [('baseline_checkpoint', 'baseline_sha256'),
                                   ('holdout_manifest', 'holdout_sha256')]:
            self.assertEqual(hashlib.sha256((root/config[path_key]).read_bytes()).hexdigest(),
                             config[hash_key])
