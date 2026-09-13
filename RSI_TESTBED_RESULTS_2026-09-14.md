# RSI testbed: 12-seed / 4-generation ablation — 2026-09-14

## Protocol

- Synthetic, offline evaluation only; no model calls, network, external tools,
  executable candidate code, or changes to evaluator/budget boundaries.
- 12 paired seeds (`20260914`–`20260925`), four generations, and six proposals
  per generation in every arm.
- Arms: `recursive`, `frozen`, `memory-only`, `meta-only`, and
  `algorithm-only`.
- Meta-capable arms receive an identical, pre-registered schedule probe during
  generations 1–3; only generation 4 can choose a schedule based on recorded
  outcomes. `algorithm-only` repeats the same schedule sequence without
  outcome memory.
- Primary decision rule: reject the no-advantage null only when both the paired
  bootstrap 95% interval excludes zero and the two-sided sign-permutation
  p-value is below .05. Final evaluation uses both the held-out original task
  family and an unused 31-bit unknown-domain task family.

Raw artifact: `runs/rsi-ablation-12seed-4gen-v2.json`.

## Results

| Contrast | Final performance delta | Unknown-domain delta | Bootstrap 95% interval | Permutation p | Decision |
|---|---:|---:|---:|---:|---|
| recursive − frozen | +0.00881 | +0.00713 | [−0.00025, +0.01914] | 0.16275 | Do not reject null |
| memory-only − frozen | +0.00000 | +0.00000 | [0.00000, 0.00000] | 1.00000 | Do not reject null |
| meta-only − frozen | +0.00881 | +0.00713 | [+0.00023, +0.01919] | 0.18175 | Do not reject null |
| algorithm-only − frozen | +0.00961 | +0.00798 | [+0.00080, +0.01937] | 0.09950 | Do not reject null |

## Finding

The preregistered criterion is not met for any contrast. Therefore this run
does **not** establish a recursive self-improvement advantage.

`recursive` and `meta-only` are identical in this result, while `memory-only`
matches `frozen`. Within this task family, the observed positive shifts are
therefore compatible with the pre-registered schedule changes themselves,
not evidence that retained improvement records increased future improvement
discovery. The unknown-domain score moves in the same direction but is also not
enough to pass the primary decision rule.

This is a completed bounded testbed run, not an AGI claim, deployment decision,
or a claim about unrestricted self-modification.
