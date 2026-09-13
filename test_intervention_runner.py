import unittest
from intervention_runner import run_interventions, run_policy_interventions

class InterventionTests(unittest.TestCase):
    def test_executes_attribute_interventions(self):
        r=run_interventions('私 は 赤い リンゴ が 好き だ')
        self.assertTrue(r['all_tests_executed'])
        self.assertTrue(any(x['changed'] for x in r['interventions']))

    def test_executes_policy_interventions(self):
        r=run_policy_interventions()
        self.assertEqual(len(r['interventions']),4)
        self.assertTrue(all('difference' in x for x in r['interventions']))

if __name__=='__main__': unittest.main()
