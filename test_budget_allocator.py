import unittest
from budget_allocator import allocate

class BudgetTests(unittest.TestCase):
    def test_uncertain_branch_survives(self):
        r=allocate([{'branch':'gain','difference':{'delta':.1,'interval':[.08,.12]}},{'branch':'unknown','difference':{'delta':0,'interval':[-.2,.2]},'novelty':1}],20)
        self.assertEqual(sum(x['budget'] for x in r['allocations']),20)
        self.assertTrue(all(x['budget']>=1 for x in r['allocations']))
    def test_gain_gets_more(self):
        r=allocate([{'branch':'gain','difference':{'delta':.2,'interval':[.19,.21]}},{'branch':'flat','difference':{'delta':0,'interval':[0,0]}}],20)
        self.assertGreater(r['allocations'][0]['budget'],r['allocations'][1]['budget'])
if __name__=='__main__': unittest.main()
