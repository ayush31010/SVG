import unittest

from scripts.train_candidate_set_grounding import candidate_sets, categorize, training_targets


class CandidateSetTests(unittest.TestCase):
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
