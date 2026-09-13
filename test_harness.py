import unittest
from unittest.mock import patch
from harness import BASELINE, compare, evaluate, search, validate


class HarnessTests(unittest.TestCase):
    def test_equal_objective_call_budget(self):
        for policy in (BASELINE, dict(BASELINE, explore=0, restart=5, temperature=.1)):
            calls = []
            def score(bits):
                calls.append(tuple(bits))
                return .5
            with patch('harness.objective', return_value=score):
                search(policy, 'linear', 9, budget=120)
            self.assertEqual(len(calls), 120)

    def test_identical_policy_never_adopted(self):
        scores = evaluate(BASELINE, range(5))
        result = compare(scores, scores)
        self.assertEqual(result['mean_delta'], 0)
        self.assertFalse(result['accepted'])

    def test_regression_rejected(self):
        old = {c: [.8] * 10 for c in ('linear', 'trap', 'rugged')}
        new = dict(old, linear=[1.] * 10, trap=[.7] * 10)
        self.assertFalse(compare(old, new)['accepted'])

    def test_real_gain_accepted(self):
        old = {c: [.6] * 10 for c in ('linear', 'trap', 'rugged')}
        new = {c: [.7] * 10 for c in old}
        self.assertTrue(compare(old, new)['accepted'])

    def test_invalid_policy_rejected(self):
        for update in ({'explore': float('nan')}, {'restart': True}, {'exec': 'anything'}, {'flip': 0}):
            with self.assertRaises(ValueError):
                validate(dict(BASELINE, **update))

    def test_seed_reproducibility_and_bounds(self):
        for c in ('linear', 'trap', 'rugged'):
            a = search(BASELINE, c, 6)
            self.assertEqual(a, search(BASELINE, c, 6))
            self.assertTrue(0 <= a <= 1)


if __name__ == '__main__':
    unittest.main()
