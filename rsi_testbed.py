"""Offline, bounded testbed for measuring *improvement discovery*.

This module is deliberately not an autonomous self-modification system.  Agents
can propose only data in MUTABLE_REGIONS.  The security boundary, resource
budget, development judge, final evaluator, and acceptance rule are ordinary
Python code outside that data boundary.  The testbed uses the synthetic search
tasks in ``harness.py`` and requires no model or network connection.

It separates three scopes:
  1. parameter self-optimization (the candidate search policy),
  2. algorithm self-improvement (the candidate-generation schedule), and
  3. meta-optimization (which schedule is allocated next generation).

The primary experiment is a paired recursive-vs-frozen control.  The recursive
arm retains only experiment records; the frozen arm receives the same initial
configuration and budget but an empty memory at every generation.  A result is
only a falsifiable synthetic finding, never evidence of AGI or a deployment
decision.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable

from harness import BASELINE, CATEGORIES, search, summary, validate


ROOT = Path(__file__).resolve().parent
MUTABLE_REGIONS = frozenset({"policy", "prompt", "memory", "tools", "algorithm", "agent_topology"})
IMMUTABLE_BOUNDARIES = frozenset({
    "security_boundary", "resource_limit", "development_judge",
    "final_external_evaluator", "unknown_domain_evaluator", "acceptance_rule",
})
DEFAULT_BUDGET = 6  # Candidate proposals per generation; identical in both arms.
POLICY_FIELDS = ("explore", "flip", "temperature", "restart")
SCHEDULES = ("coordinate", "balanced", "robustness")
INHERITANCE_MODES = ("recursive", "frozen", "memory-only", "meta-only", "algorithm-only")


@dataclass(frozen=True)
class Candidate:
    """Data-only proposal; nothing in this object can alter a boundary."""
    policy: dict[str, Any]
    regions: tuple[str, ...]
    schedule: str
    proposal_cost: int
    rationale: str


@dataclass(frozen=True)
class Evaluation:
    performance: float
    robustness: float
    falsification: float
    generalization: float
    cost_efficiency: float
    safe: bool

    @property
    def utility(self) -> float:
        """A reported metric only; it is not the adoption rule."""
        return statistics.mean((self.performance, self.robustness,
                                self.falsification, self.generalization,
                                self.cost_efficiency))


def _mean(values: Iterable[float]) -> float:
    return statistics.mean(values)


def _evaluate_fixed_budget(policy: dict[str, Any], seeds: tuple[int, ...], budget: int) -> dict[str, list[float]]:
    """Evaluator-owned task budget; candidate data cannot override it."""
    return {category: [search(policy, category, seed, budget=budget) for seed in seeds]
            for category in CATEGORIES}


class FixedJudge:
    """Non-mutable evaluator for development, adversarial, and final splits."""

    def __init__(self, label: str, seeds: range, budget: int = 120):
        self.label, self.seeds, self.budget = label, tuple(seeds), budget
        # Memoization is evaluator-owned and keyed only by candidate data.  It
        # changes neither task outcomes nor the declared resource budget.
        self._cache: dict[tuple[str, int], Evaluation] = {}

    def evaluate(self, candidate: Candidate) -> Evaluation:
        # validate is independent of the agents and rejects out-of-scope policy
        # fields and numerical values before any objective is evaluated.
        try:
            validate(candidate.policy)
        except ValueError:
            return Evaluation(0, 0, 0, 0, 0, False)
        cache_key = (json.dumps(candidate.policy, sort_keys=True, ensure_ascii=False), candidate.proposal_cost)
        if cache_key in self._cache:
            return self._cache[cache_key]
        scores = _evaluate_fixed_budget(candidate.policy, self.seeds, self.budget)
        category_means = summary(scores)
        performance = _mean(category_means.values())
        robustness = min(category_means.values())
        # Fixed counterexamples: one category must not lag the mean by > .06.
        # The value is a separate falsification dimension, not a model verdict.
        falsification = 1.0 if all(performance - value <= .06 for value in category_means.values()) else 0.0
        # A disjoint deterministic transformation of the supplied split tests
        # generalization without exposing the final evaluation split.
        # ``evaluate`` visits the seed iterable once per category, so this must
        # be a reusable sequence rather than a one-shot generator.
        generalization_scores = _evaluate_fixed_budget(
            candidate.policy, tuple(seed + 50_000 for seed in self.seeds), self.budget
        )
        generalization = _mean(summary(generalization_scores).values())
        cost_efficiency = performance / candidate.proposal_cost
        outcome = Evaluation(performance, robustness, falsification, generalization,
                             cost_efficiency, True)
        self._cache[cache_key] = outcome
        return outcome


def _unknown_search(policy: dict[str, Any], seed: int, budget: int = 96) -> float:
    """A held-out 31-bit landscape with a different objective geometry."""
    rng = random.Random(seed)
    target = [rng.randrange(2) for _ in range(31)]
    weights = [rng.uniform(.3, 1.8) for _ in range(31)]

    def score(bits: list[int]) -> float:
        matches = [int(bit == goal) for bit, goal in zip(bits, target)]
        weighted = sum(match * weight for match, weight in zip(matches, weights)) / sum(weights)
        rings = sum(int(matches[index] == matches[(index + 1) % len(matches)]) for index in range(len(matches))) / len(matches)
        blocks = [sum(matches[index:index + 5]) for index in range(0, 30, 5)]
        deceptive = sum(5 if count == 5 else 4 - count for count in blocks) / 30
        return .45 * weighted + .30 * rings + .25 * deceptive

    current = [rng.randrange(2) for _ in range(31)]
    value = best = score(current)
    stagnant = 0
    for _ in range(budget - 1):
        explore = rng.random()
        random_bits = [rng.randrange(2) for _ in range(31)]
        flips = [rng.random() for _ in range(31)]
        forced = rng.randrange(31)
        accept_draw = rng.random()
        restart = stagnant >= policy["restart"]
        if explore < policy["explore"] or restart:
            candidate = random_bits
        else:
            mask = [draw < policy["flip"] for draw in flips]
            if not any(mask):
                mask[forced] = True
            candidate = [bit ^ int(flip) for bit, flip in zip(current, mask)]
        result = score(candidate)
        if result > best:
            best, stagnant = result, 0
        else:
            stagnant += 1
        temperature = policy["temperature"]
        if restart or result >= value or (temperature > 0 and accept_draw < math.exp(max(-700, (result - value) / temperature))):
            current, value = candidate, result
            if restart:
                stagnant = 0
    return best


class UnknownDomainEvaluator:
    """Fixed final evaluator over an objective the development judge never uses."""

    def __init__(self, seeds: range, budget: int = 96):
        self.seeds, self.budget = tuple(seeds), budget
        self._cache: dict[tuple[str, int], Evaluation] = {}

    def evaluate(self, candidate: Candidate) -> Evaluation:
        try:
            validate(candidate.policy)
        except ValueError:
            return Evaluation(0, 0, 0, 0, 0, False)
        cache_key = (json.dumps(candidate.policy, sort_keys=True, ensure_ascii=False), candidate.proposal_cost)
        if cache_key in self._cache:
            return self._cache[cache_key]
        scores = [_unknown_search(candidate.policy, seed, self.budget) for seed in self.seeds]
        generalization_scores = [_unknown_search(candidate.policy, seed + 60_000, self.budget) for seed in self.seeds]
        performance = _mean(scores)
        robustness = min(scores)
        falsification = 1.0 if robustness >= performance - .18 else 0.0
        outcome = Evaluation(performance, robustness, falsification, _mean(generalization_scores),
                             performance / candidate.proposal_cost, True)
        self._cache[cache_key] = outcome
        return outcome


DEVELOPMENT_JUDGE = FixedJudge("development", range(10_000, 10_012))
ADVERSARIAL_JUDGE = FixedJudge("adversarial", range(70_000, 70_008))
FINAL_EXTERNAL_EVALUATOR = FixedJudge("final-external", range(900_000, 900_040))
UNKNOWN_DOMAIN_EVALUATOR = UnknownDomainEvaluator(range(1_200_000, 1_200_040))


def boundary_digest() -> str:
    """Fingerprint the evaluator/budget definitions that candidates cannot edit."""
    material = {
        "immutable_boundaries": sorted(IMMUTABLE_BOUNDARIES),
        "regions": sorted(MUTABLE_REGIONS),
        "budget": DEFAULT_BUDGET,
        "development": list(DEVELOPMENT_JUDGE.seeds),
        "adversarial": list(ADVERSARIAL_JUDGE.seeds),
        "final": list(FINAL_EXTERNAL_EVALUATOR.seeds),
        "unknown_domain_final": list(UNKNOWN_DOMAIN_EVALUATOR.seeds),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _bounded(policy: dict[str, Any], field: str, delta: float) -> dict[str, Any]:
    out = dict(policy)
    if field == "restart":
        out[field] = max(5, min(120, int(round(out[field] + delta))))
    else:
        bounds = {"explore": (0.0, 1.0), "flip": (.001, .5), "temperature": (0.0, .2)}
        lo, hi = bounds[field]
        out[field] = max(lo, min(hi, out[field] + delta))
    return out


def _step(field: str, direction: int, schedule: str) -> float:
    base = {"explore": .10, "flip": .025, "temperature": .02, "restart": 10}[field]
    multipliers = {
        "coordinate": {"explore": 1.0, "flip": .7, "temperature": .5, "restart": .6},
        "balanced": {"explore": .7, "flip": 1.0, "temperature": .8, "restart": 1.0},
        "robustness": {"explore": .5, "flip": .8, "temperature": 1.0, "restart": 1.2},
    }
    return direction * base * multipliers[schedule][field]


class Builder:
    """Generates data-only candidates under a fixed proposal-count budget."""

    def propose(self, policy: dict[str, Any], schedule: str, seed: int, budget: int,
                memory: "ResearchMemory | None" = None) -> list[Candidate]:
        rng = random.Random(seed)
        proposals = []
        focus = memory.preferred_field() if memory and memory.records else None
        fields = ((focus,) + tuple(field for field in POLICY_FIELDS if field != focus)
                  if focus else POLICY_FIELDS)
        for index in range(budget):
            field = fields[index % len(fields)]
            direction = 1 if rng.randrange(2) else -1
            next_policy = _bounded(policy, field, _step(field, direction, schedule))
            # All mutable regions are represented as constrained metadata.  In
            # this offline prototype only the policy changes task behavior;
            # the other regions change candidate-generation, never the judge.
            regions = ("policy", "prompt", "memory", "tools", "algorithm", "agent_topology")
            proposals.append(Candidate(
                policy=next_policy, regions=regions, schedule=schedule,
                proposal_cost=1, rationale=f"{schedule}:{field}:{direction}:{index}",
            ))
        return proposals


class Critic:
    """Rejects malformed/out-of-bound candidates before scoring."""

    def accept(self, candidate: Candidate) -> bool:
        try:
            validate(candidate.policy)
        except ValueError:
            return False
        return (set(candidate.regions) <= MUTABLE_REGIONS and
                candidate.schedule in SCHEDULES and candidate.proposal_cost == 1)


class Adversary:
    """Runs a fixed counterexample split; it cannot choose the adoption rule."""

    def challenge(self, candidate: Candidate) -> Evaluation:
        return ADVERSARIAL_JUDGE.evaluate(candidate)


class Judge:
    """Applies a fixed, multi-objective, non-Goodhart acceptance relation."""

    def select(self, incumbent: Candidate, proposals: list[Candidate]) -> tuple[Candidate, dict[str, Any]]:
        incumbent_dev = DEVELOPMENT_JUDGE.evaluate(incumbent)
        incumbent_adv = ADVERSARIAL_JUDGE.evaluate(incumbent)
        survivors = []
        for proposal in proposals:
            dev, adversarial = DEVELOPMENT_JUDGE.evaluate(proposal), ADVERSARIAL_JUDGE.evaluate(proposal)
            # Predeclared independence: performance, robustness, falsification,
            # generalization, and cost are checked separately.  No agent input
            # can modify these criteria.
            improves = (dev.performance > incumbent_dev.performance + .001 and
                        dev.robustness >= incumbent_dev.robustness - .015 and
                        dev.generalization >= incumbent_dev.generalization - .015 and
                        dev.falsification >= incumbent_dev.falsification and
                        adversarial.safe and adversarial.falsification >= incumbent_adv.falsification and
                        adversarial.robustness >= incumbent_adv.robustness - .02 and
                        dev.cost_efficiency >= incumbent_dev.cost_efficiency)
            if improves:
                survivors.append((proposal, dev, adversarial))
        if not survivors:
            return incumbent, {"accepted": False, "reason": "no candidate met fixed independent checks"}
        # Deterministic tie-break, expressly not model-generated.
        best, dev, adversarial = max(survivors, key=lambda item: (
            item[1].performance, item[1].robustness, item[1].generalization,
            item[1].falsification, item[1].cost_efficiency, item[0].rationale,
        ))
        return best, {"accepted": True, "development": asdict(dev), "adversarial": asdict(adversarial)}


class ResearchMemory:
    """Stores experiment results only, never evaluator controls or instructions."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def preferred_schedule(self) -> str:
        accepted = [r for r in self.records if r["accepted"]]
        if not accepted:
            return "balanced"
        # Meta-optimizer has access only to prior measured deltas and schedule.
        means = {s: [] for s in SCHEDULES}
        for record in accepted:
            means[record["schedule"]].append(record["improvement"])
        return max(SCHEDULES, key=lambda schedule: (statistics.mean(means[schedule]) if means[schedule] else -1, schedule))

    def preferred_field(self) -> str:
        """Return the field associated with the strongest accepted prior delta."""
        accepted = [record for record in self.records if record["accepted"] and record.get("accepted_field")]
        if not accepted:
            return "explore"
        return max(accepted, key=lambda record: (record["improvement"], record["accepted_field"]))["accepted_field"]


class MetaOptimizer:
    """Improves the proposal schedule, not evaluation boundaries or budget."""

    def choose_schedule(self, memory: ResearchMemory, uses_meta_memory: bool, generation: int,
                        algorithm_only: bool = False) -> str:
        if uses_meta_memory:
            # Every meta-capable arm receives the same pre-registered three
            # schedule probes.  Only later choices may depend on outcomes;
            # otherwise a first balanced choice could never gather evidence
            # about coordinate or robustness schedules.
            if generation <= len(SCHEDULES):
                return SCHEDULES[generation - 1]
            return memory.preferred_schedule()
        if algorithm_only:
            # A fixed, precommitted schedule tests algorithm variation without
            # inheriting any empirical success/failure knowledge.
            return SCHEDULES[(generation - 1) % len(SCHEDULES)]
        return "balanced"  # Frozen and memory-only controls share this schedule.


def _initial_candidate() -> Candidate:
    return Candidate(dict(BASELINE), ("policy",), "balanced", 1, "baseline")


def run_arm(seed: int, generations: int, recursive: bool | None = None,
            mode: str | None = None) -> dict[str, Any]:
    """Run one bounded arm.  Final evaluation happens only after all adoption."""
    if mode is None:
        mode = "recursive" if recursive else "frozen"
    if mode not in INHERITANCE_MODES:
        raise ValueError(f"Unknown inheritance mode: {mode}")
    before_digest = boundary_digest()
    builder, critic, adversary, judge, meta = Builder(), Critic(), Adversary(), Judge(), MetaOptimizer()
    memory = ResearchMemory()
    incumbent = _initial_candidate()
    points = [DEVELOPMENT_JUDGE.evaluate(incumbent).performance]
    records = []
    for generation in range(1, generations + 1):
        stores_memory = mode in {"recursive", "memory-only", "meta-only"}
        uses_builder_memory = mode in {"recursive", "memory-only"}
        uses_meta_memory = mode in {"recursive", "meta-only"}
        schedule = meta.choose_schedule(memory, uses_meta_memory, generation, mode == "algorithm-only")
        proposed = builder.propose(
            incumbent.policy, schedule, seed * 10_000 + generation, DEFAULT_BUDGET,
            memory if uses_builder_memory else None,
        )
        reviewed = [candidate for candidate in proposed if critic.accept(candidate) and adversary.challenge(candidate).safe]
        selected, verdict = judge.select(incumbent, reviewed)
        previous = points[-1]
        current = DEVELOPMENT_JUDGE.evaluate(selected).performance
        improvement = current - previous
        record = {"generation": generation, "schedule": schedule, "accepted": verdict["accepted"],
                  "improvement": improvement, "proposal_count": len(proposed),
                  "reviewed_count": len(reviewed), "judge": verdict,
                  "accepted_field": selected.rationale.split(":")[1] if verdict["accepted"] else None}
        if stores_memory:
            memory.append(record)
        incumbent = selected
        points.append(current)
        records.append(record)
    if boundary_digest() != before_digest:
        raise RuntimeError("Immutable evaluation boundary changed during run")
    improvements = [points[index + 1] - points[index] for index in range(len(points) - 1)]
    accelerations = [improvements[index + 1] / improvements[index]
                     for index in range(len(improvements) - 1) if abs(improvements[index]) > 1e-12]
    final = FINAL_EXTERNAL_EVALUATOR.evaluate(incumbent)
    unknown_final = UNKNOWN_DOMAIN_EVALUATOR.evaluate(incumbent)
    return {
        "seed": seed, "arm": mode,
        "points": points, "improvements": improvements, "accelerations": accelerations,
        "records": records, "final_external_evaluation": asdict(final),
        "unknown_domain_evaluation": asdict(unknown_final),
        "memory_records_retained": len(memory.records), "boundary_digest": before_digest,
    }


def paired_statistics(recursive: list[dict[str, Any]], frozen: list[dict[str, Any]], seed: int,
                      left_arm: str = "recursive", right_arm: str = "frozen") -> dict[str, Any]:
    """Paired bootstrap + sign-permutation test; null is no recursive advantage."""
    recursive_by_seed = {row["seed"]: row for row in recursive}
    frozen_by_seed = {row["seed"]: row for row in frozen}
    pairs = sorted(set(recursive_by_seed) & set(frozen_by_seed))
    deltas = [
        recursive_by_seed[item]["final_external_evaluation"]["performance"] -
        frozen_by_seed[item]["final_external_evaluation"]["performance"]
        for item in pairs
    ]
    unknown_deltas = [
        recursive_by_seed[item]["unknown_domain_evaluation"]["performance"] -
        frozen_by_seed[item]["unknown_domain_evaluation"]["performance"]
        for item in pairs
    ]
    discovery_deltas = [sum(recursive_by_seed[item]["improvements"]) - sum(frozen_by_seed[item]["improvements"])
                        for item in pairs]
    observed = _mean(deltas) if deltas else 0.0
    rng = random.Random(seed)
    bootstrap = sorted(_mean(rng.choices(deltas, k=len(deltas))) for _ in range(4_000)) if deltas else [0.0]
    permutation = [_mean([value if rng.randrange(2) else -value for value in deltas]) for _ in range(4_000)] if deltas else [0.0]
    p_value = sum(abs(value) >= abs(observed) for value in permutation) / len(permutation)
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "paired_seeds": pairs,
        "mean_final_performance_delta_recursive_minus_frozen": observed,
        "mean_improvement_discovery_delta": _mean(discovery_deltas) if discovery_deltas else 0.0,
        "mean_unknown_domain_delta_recursive_minus_frozen": _mean(unknown_deltas) if unknown_deltas else 0.0,
        "bootstrap_interval_95": [bootstrap[99], bootstrap[3899]],
        "two_sided_sign_permutation_p": p_value,
        "interpretation": "Reject the no-advantage null only if the preregistered interval excludes 0 and p < .05; this synthetic result does not establish RSI outside this testbed.",
    }


def run_ablation_experiment(seeds: int = 12, generations: int = 4, seed: int = 41) -> dict[str, Any]:
    """Identify whether memory, meta-optimization, or static algorithm changes matter."""
    if seeds < 2 or generations < 2:
        raise ValueError("Use at least two seeds and two generations for an ablation")
    arms = {
        mode: [run_arm(seed + index, generations, mode=mode) for index in range(seeds)]
        for mode in INHERITANCE_MODES
    }
    comparisons = {
        f"{mode}_vs_frozen": paired_statistics(arms[mode], arms["frozen"], seed + 100 + index, mode, "frozen")
        for index, mode in enumerate(INHERITANCE_MODES) if mode != "frozen"
    }
    return {
        "arms": arms,
        "comparisons": comparisons,
        "interpretation": (
            "A recursive effect is attributable to retained improvement knowledge only if recursive "
            "outperforms frozen and the targeted ablations isolate the contributing channel. "
            "No outcome is evidence of general RSI outside this synthetic testbed."
        ),
        "boundary_digest": boundary_digest(),
    }


def run_experiment(seeds: int = 12, generations: int = 4, seed: int = 41) -> dict[str, Any]:
    if seeds < 2 or generations < 2:
        raise ValueError("Use at least two seeds and two generations to measure a contrast and acceleration")
    recursive = [run_arm(seed + index, generations, True) for index in range(seeds)]
    frozen = [run_arm(seed + index, generations, False) for index in range(seeds)]
    return {
        "scope": "Offline synthetic RSI testbed; no external tools, model calls, or code execution from agents",
        "mutable_regions": sorted(MUTABLE_REGIONS),
        "immutable_boundaries": sorted(IMMUTABLE_BOUNDARIES),
        "resource_limit": {"proposals_per_generation": DEFAULT_BUDGET, "objective_calls_per_task": 120},
        "design": {"paired_same_seed": True, "same_budget": True, "frozen_memory_reset_each_generation": True,
                   "incumbent_policy_retained_for_comparable_P_t": True,
                   "roles": ["Builder", "Critic", "Adversary", "Judge", "Research Memory", "Meta-Optimizer"]},
        "recursive": recursive, "frozen": frozen,
        "ablation_modes_available": list(INHERITANCE_MODES),
        "statistics": paired_statistics(recursive, frozen, seed + 99),
        "boundary_digest": boundary_digest(),
        "warning": "Scores and acceleration are measurements on this fixed synthetic task family, not an AGI, autonomous-RSI, or deployment claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded offline recursive-improvement testbed")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ablations", action="store_true", help="Run memory/meta/algorithm ablation arms as well")
    args = parser.parse_args()
    result = (run_ablation_experiment(args.seeds, args.generations, args.seed)
              if args.ablations else run_experiment(args.seeds, args.generations, args.seed))
    output = args.output or ROOT / "runs" / f"rsi-testbed-{time.strftime('%Y%m%d-%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "statistics": result.get("statistics"),
                      "comparisons": result.get("comparisons")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
