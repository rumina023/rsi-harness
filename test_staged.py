import unittest
from staged import difference, select, passes

class StagedTests(unittest.TestCase):
    def test_uncertain_candidate_retained(self):
        rows = [{'id':i,'difference':{'delta':d,'interval':[0,u]}} for i,d,u in [(0,.1,.11),(1,.09,.1),(2,.01,.3),(3,-.1,-.01)]]
        self.assertEqual([r['id'] for r in select(rows,2)],[0,2])

    def test_identical_does_not_pass(self):
        values={c:[.5]*8 for c in ('linear','trap','rugged')}
        self.assertFalse(passes(difference(values,values)))

    def test_category_regression_rejected(self):
        self.assertFalse(passes({'delta':.1,'interval':[.05,.15],'categories':{'a':.2,'b':-.03}}))

    def test_common_seed_gain(self):
        a={c:[.5]*8 for c in ('linear','trap','rugged')}
        b={c:[.6]*8 for c in a}
        self.assertTrue(passes(difference(a,b)))

if __name__=='__main__':
    unittest.main()
