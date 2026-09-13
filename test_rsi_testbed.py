import unittest

from rsi_testbed import (
    BASELINE, Candidate, Critic, DEVELOPMENT_JUDGE, MUTABLE_REGIONS,
    UNKNOWN_DOMAIN_EVALUATOR, boundary_digest, run_arm, run_experiment,
)


class RSITestbedTests(unittest.TestCase):
    def test_critic_rejects_boundary_and_policy_escape(self):
        critic = Critic()
        boundary_escape = Candidate(dict(BASELINE), ("security_boundary",), "balanced", 1, "bad")
        policy_escape = Candidate(dict(BASELINE, evaluator="changed"), ("policy",), "balanced", 1, "bad")
        self.assertFalse(critic.accept(boundary_escape))
        self.assertFalse(critic.accept(policy_escape))
        self.assertNotIn("security_boundary", MUTABLE_REGIONS)

    def test_judge_is_deterministic_and_external(self):
        candidate = Candidate(dict(BASELINE), ("policy",), "balanced", 1, "baseline")
        self.assertEqual(DEVELOPMENT_JUDGE.evaluate(candidate), DEVELOPMENT_JUDGE.evaluate(candidate))
        self.assertEqual(boundary_digest(), boundary_digest())

    def test_frozen_arm_retains_no_research_memory(self):
        recursive = run_arm(seed=3, generations=2, recursive=True)
        frozen = run_arm(seed=3, generations=2, recursive=False)
        self.assertEqual(recursive["memory_records_retained"], 2)
        self.assertEqual(frozen["memory_records_retained"], 0)
        self.assertEqual(len(recursive["improvements"]), 2)
        self.assertEqual(len(frozen["improvements"]), 2)
        self.assertEqual(recursive["boundary_digest"], frozen["boundary_digest"])

    def test_memory_and_meta_ablations_are_separated(self):
        memory_only = run_arm(seed=13, generations=2, mode="memory-only")
        meta_only = run_arm(seed=13, generations=2, mode="meta-only")
        algorithm_only = run_arm(seed=13, generations=2, mode="algorithm-only")
        self.assertEqual(memory_only["memory_records_retained"], 2)
        self.assertEqual(meta_only["memory_records_retained"], 2)
        self.assertEqual(algorithm_only["memory_records_retained"], 0)
        self.assertEqual([row["schedule"] for row in algorithm_only["records"]], ["coordinate", "balanced"])
        self.assertEqual([row["schedule"] for row in meta_only["records"]], ["coordinate", "balanced"])

    def test_unknown_domain_is_not_the_development_evaluator(self):
        candidate = Candidate(dict(BASELINE), ("policy",), "balanced", 1, "baseline")
        self.assertEqual(UNKNOWN_DOMAIN_EVALUATOR.evaluate(candidate), UNKNOWN_DOMAIN_EVALUATOR.evaluate(candidate))
        arm = run_arm(seed=9, generations=2, recursive=True)
        self.assertIn("unknown_domain_evaluation", arm)
        self.assertEqual(set(arm["unknown_domain_evaluation"]), set(arm["final_external_evaluation"]))

    def test_paired_experiment_reports_all_required_measures(self):
        result = run_experiment(seeds=2, generations=2, seed=7)
        self.assertEqual(len(result["recursive"]), 2)
        self.assertEqual(len(result["frozen"]), 2)
        stats = result["statistics"]
        self.assertIn("mean_improvement_discovery_delta", stats)
        self.assertIn("bootstrap_interval_95", stats)
        for arm in result["recursive"] + result["frozen"]:
            self.assertIn("accelerations", arm)
            final = arm["final_external_evaluation"]
            self.assertEqual(set(final), {"performance", "robustness", "falsification", "generalization", "cost_efficiency", "safe"})
            self.assertIn("unknown_domain_evaluation", arm)


if __name__ == "__main__":
    unittest.main()
