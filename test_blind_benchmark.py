import unittest
from blind_benchmark import run

class BlindTests(unittest.TestCase):
    def test_benchmark_is_explicit_about_limit(self):
        r=run()
        self.assertEqual(len(r['cases']),4)
        self.assertGreaterEqual(r['accuracy'],.75)
        self.assertIn('実世界',r['warning'])
if __name__=='__main__': unittest.main()
