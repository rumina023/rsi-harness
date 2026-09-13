import unittest
from anchor_verifier import AnchorVerifier, verifier_benchmark

class AnchorTests(unittest.TestCase):
    def test_unknowns_are_preserved(self):
        r=AnchorVerifier().verify('私は赤いリンゴが大好きだ')
        self.assertGreater(r['consistency']['unknown_count'], 0)
        self.assertTrue(r['falsification_tests'])

    def test_consistency_is_not_truth(self):
        r=AnchorVerifier([{'claim':'リンゴは青い','supports':True}]).verify('リンゴは青い')
        self.assertIn(r['status'], ('未反証','証拠不足'))

    def test_contradiction_wins(self):
        r=AnchorVerifier([{'claim':'リンゴは青い','supports':False}]).verify('リンゴは青い')
        self.assertEqual(r['status'],'反証あり')

    def test_interventions_and_benchmark(self):
        r=AnchorVerifier().verify('私 は 赤い リンゴ が 好き だ')
        self.assertTrue(r['interventions'])
        self.assertEqual(verifier_benchmark()['pass_rate'],1.0)

if __name__=='__main__': unittest.main()
