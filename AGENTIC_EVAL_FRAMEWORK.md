# Agentic Eval Framework

An eval can only answer questions supported by the units, factors, and
measurement process it logs.

The workflow is:

1. Define the logged units and factors.
2. Decide which questions are identifiable from those data.
3. Choose methods that match the data shape.
4. Mark unavailable questions explicitly.

Questions are always part of the framework. Answers depend on logged dimensions.

## 1. Taxonomy And Data Contract

System factors:

```text
S = M x H x E
```

- `M`: model/backend.
- `H`: prompt, harness, scaffold, tool wrapper, memory, retry policy, planner.
- `E`: environment, tools, simulator, state, policy, evaluator, versions.

Evidence units:

```text
B -> C -> T -> R -> tau -> e
```

- `B`: benchmark/eval suite.
- `C`: domain, scenario, repository, category, reward basis, or other slice.
- `T`: task, competition, or shared unit of work.
- `R`: run, rollout, trial, seed, job, or repeated attempt.
- `tau`: trajectory for one run.
- `e`: event inside a trajectory.

Measurement provenance:

- Every score or label should state how it was produced: ground-truth verifier,
  deterministic checker, simulator reward, human judge, LLM judge, or reward
  model.
- Judged, simulator-based, or model-based scores need provenance fields such as
  `judge_id`, `verifier_id`, `reward_model_id`, `simulator_id`,
  `rubric_version`, repeated labels, or expert labels.

Minimum contract:

```text
eval_id, dataset_version, eval_scope, task_id, system_id/config_id,
metric, score, scoring_method
```

Fields for stronger claims:

```text
run_id, model_id, prompt_id, harness_id, environment_id/version,
domain/slice, judge_id, verifier_id, reward_model_id, simulator_id,
rubric_version, repeated_label, expert_label, trajectory_id, event_type,
cost, latency, tokens, failure_label, severity_label
```

If `scoring_method` is judged, simulator-based, or model-based, the relevant
judge/verifier/reward provenance fields become part of the required contract.
If `scoring_method` is unknown, the outcome table fails the minimum data
contract.

## 2. Question Contract

### Object And Data Contract

| Question | Required data | Methods | Valid claim |
| --- | --- | --- | --- |
| What is being measured? | system/config/model/prompt/harness/env IDs | estimand map; factor-variation audit | Names the compared object. |
| What population is represented? | task/domain/snapshot metadata | coverage; domain mix; effective counts | Describes the task/slice support. |
| Is the outcome table valid? | task-system outcomes; score/provenance fields | coverage matrix; missingness; scoring sanity; aggregation check | Shows whether analysis is well-formed. |
| What cannot be answered? | target questions + missing fields | data-gap matrix | Marks non-identifiable claims. |

The validity check includes conditional-field completeness. If a judged,
simulator-based, or model-based metric lacks its required provenance, observed
scores may still be reported with a measurement caveat, but judge/reward
reliability and strong measurement-quality claims are unavailable.

### Outcome Questions

| Question | Required data | Methods | Valid claim |
| --- | --- | --- | --- |
| Who is better, and by how much? | task-level scores for X/Y | mean gap; paired gap; task/cluster bootstrap CI; BCa when feasible | Observed or estimated performance gap. |
| What rank/top-K did X get? | scores by system | leaderboard; tie-aware rank; top-K | Descriptive fixed-benchmark rank. |
| Where does X win or lose? | task/domain metadata | slice scores; task deltas; stratified/block bootstrap; influence | Domain/task-specific strengths. |
| Is X always better? | task/run/slice outcomes | dominance table; win-rate by task/slice/run; Wilson/Clopper-Pearson interval | Dominance only over observed support. |

### Resolution Questions

| Question | Required data | Methods | Valid claim |
| --- | --- | --- | --- |
| Is X resolvably better than Y? | shared task outcomes | paired estimate; task/cluster CI; paired test; McNemar for paired binary outcomes | Current X-vs-Y evidence. |
| Is the leaderboard locally resolved? | task-level matrix | adjacent/all-pairs; max-T or Holm by family; BH only for exploration | Which displayed gaps survive uncertainty. |
| Are ranks/top-K stable? | task-level matrix | task/cluster bootstrap; stratified/block bootstrap; rank intervals; top-K probability | Rank and cutoff fragility. |
| Are non-separated systems equivalent? | practical margin + paired outcomes | equivalence/non-inferiority test | Equivalence only if interval fits margin. |
| Was X-vs-Y predeclared? | analysis plan or selection record | family definition; multiplicity correction | Separates preplanned from post-hoc claims. |

For current inference, prefer estimate + interval + adjusted test when needed.
Do not use same-data post-hoc MDE as the primary reason to accept or reject a
current X-vs-Y claim.

Multiplicity rule: use max-T for correlated adjacent-rank or all-pairs families
on shared tasks; use Holm for small predeclared confirmatory families when
max-T is unavailable; use BH only for exploratory scans where false discovery
rate, not family-wise error rate, is the target.

### Design And Budget Questions

| Question | Required data | Methods | Valid claim |
| --- | --- | --- | --- |
| What can this eval detect? | design size; pilot/held-out variance | MDE; power; required-N simulation | Prospective detectable gap scale. |
| Can this eval reliably detect gap `delta`? | target effect; variance estimate | power curve; simulation | Planning adequacy for target gap. |
| What uncertainty source dominates? | repeated tasks/runs/judges/domains | nested bootstrap; variance components; mixed models | Task/run/judge/domain variance budget. |
| What reduces uncertainty per run or dollar? | variance components; costs | K-simulation; marginal CI reduction; value-of-information | Budget allocation guidance. |
| More tasks, runs, judges, or domains? | pilot variance by component | design simulation; adaptive allocation | Next eval design choice. |

MDE, power, and required-N belong mainly to pre-eval design or next-eval
planning. Use pilot, historical, or held-out variance when possible.
Nested bootstrap, variance components, and mixed models need enough independent
levels per factor. Report effective task/run/judge/domain counts; with 2-3
judges, 3-5 runs, or tiny/imbalanced domains, treat decomposition as a
sensitivity diagnostic rather than a strong attribution claim.

### Reliability And Failure Questions

| Question | Required data | Methods | Valid claim |
| --- | --- | --- | --- |
| Can X fail? When? | failures; task metadata; traces/events | failure slices; task diagnostics; event analysis | Observed failure conditions. |
| How reliable is X in Y? | domain Y; repeated outcomes if possible | domain score; task/cluster CI; nested bootstrap | Reliability within logged domain Y. |
| What configs/envs are best for X? | varied config/env IDs on comparable tasks | config leaderboard; interaction analysis | Best observed config/env, with caveats. |
| Does prompt/harness/env matter? | crossed factor variation | additive decomposition; interaction residuals; fixed-effect regression/ANOVA; permutation under additive null | Factor effects only if identifiable. |
| Are judges/rewards reliable? | judge/verifier IDs; repeated/expert labels | kappa/alpha; calibration; confusion matrix; judge variance | Measurement-system reliability. |
| Does performance drift? | repeated snapshots/version metadata | time-aware comparison; change-point/snapshot diagnostics | Stability over logged versions. |

Judge methods depend on design: Cohen's kappa for two raters, Fleiss' kappa or
Krippendorff's alpha for multiple or variable raters, calibration curves and
confusion matrices against expert labels, and variance components when repeated
labels identify judge effects.
Factor attribution requires crossed variation and identifies logged factors,
not internal mechanisms such as retry policy or tool wrapper choices unless
those mechanisms are separately ablated. Consecutive benchmark snapshots are
not iid draws.

## 3. Interpretation Rules

These rules summarize recurring notebook caveats, the simulator/checker routing
logic, and standard statistical guardrails.

- Point rank is descriptive.
- Non-separation is not equivalence.
- Repeated runs are not independent tasks.
- Top-K is a reporting window, not a natural statistical boundary.
- Bundled submissions/configs are not pure model claims.
- Factor attribution requires crossed variation.
- Trace availability is not parsed event evidence.
- Judge/verifier reliability is unidentified unless judge data is logged.
- Missingness conventions need sensitivity checks when they affect ranks or
  boundaries.
- Current inference uses estimates, intervals, and adjusted tests. Future design
  uses MDE, power, and required-N.

Task resampling caveats:

- Task bootstrap assumes tasks are exchangeable at the resampling level.
- Use stratified, block, or cluster bootstrap when tasks cluster by repository,
  domain, template, competition, workflow, or reward basis.
- Percentile bootstrap intervals can be too narrow at small task or slice
  counts; prefer BCa or studentized intervals when feasible.
- Report task counts, slice sizes, and effective cluster counts beside
  intervals.
- Treat tiny or imbalanced clusters as sensitivity diagnostics.

Selection caveats:

- A predeclared X-vs-Y comparison is not the same as choosing the top row after
  seeing the leaderboard.
- All-pairs and rank-family claims need multiplicity control.
- Post-hoc "best system" claims should consider shrinkage, partial pooling, or
  regularized ranking, plus rank-stability reporting.

## 4. Current Artifact Fit

- **SWE-bench Verified:** `task x submission`, single-run public outcomes.
  Supports observed scores, paired task comparisons, rank stability, repo
  sensitivity, and design-scale diagnostics. Does not support repeated-run
  stochasticity or pure model attribution.
- **Terminal-Bench 2.0:** `task x submission x trial` for a public subset.
  Supports nested uncertainty and task/run variance analysis. Repeated trials do
  not multiply the independent task count.
- **Harness-Bench:** controlled `task x model x harness` matrix. Supports
  model/harness main effects and interaction diagnostics because factors vary
  independently.
- **DeepSWE:** config-level outcomes under a constant harness. Supports
  config/ranked-unit analysis, task bootstrap, slices, and operational profile
  fields. Does not identify model-harness interaction.
- **MLE-Bench:** competition-level grading reports. Supports task-cell
  aggregation, paired comparisons, metric sensitivity, validity diagnostics,
  run-group caveats, and competition influence.
- **tau-bench / tau2-style data:** agent plus simulator loop. Supports reward
  leaderboards, paired task means, domain checks, and repeated-run, reward
  component, or trajectory diagnostics when those fields are exposed and parsed.

Current reusable code covers normalized DataFrame loading, repeated-cell
aggregation, task bootstrap rank stability, pairwise bootstrap win probability,
slices, coverage, and influence-style diagnostics. Some notebook methods are not
yet shared library features.

BetterBench is used here as benchmark-QA backing rather than copied wholesale:
purpose/scope, metric interpretability, reproducibility, statistical reporting,
versioning, and maintenance are prerequisites for the question contract.
HELM is used as multi-scenario, multi-metric reporting backing: when meaningful
scenario or metric dimensions exist, do not collapse the report to one aggregate
score alone.

## 5. Reporting Package

This package combines the notebook reporting pattern with BetterBench-style
benchmark QA and HELM-style scenario/metric coverage.

A concise eval report should include:

1. Measured object: model, config, submission, job, or full agent system.
2. Data shape: tasks, systems, runs, slices, metrics, missing cells, versions,
   repeated-cell aggregation.
3. Benchmark QA: purpose/scope, metric meaning, reproducibility path,
   statistical-reporting convention, versioning/maintenance status.
4. Observed result: score, gap, rank, ties, selected metric.
5. Current inference: estimate, interval, adjusted test where needed, cluster
   caveat.
6. Stability: bootstrap rank intervals, top-K probabilities, boundary checks.
7. Design: future MDE, power, or required-N from pilot, historical, or held-out
   variance.
8. Reliability: failure conditions, domain reliability, config/env sensitivity,
   judge/reward caveats.
9. Unavailable claims: questions blocked by missing units or factors.

Prefer:

> The paired gap is 1.2 pp with a 95% interval of [-2.8, 5.1] pp, so this eval
> does not resolve X > Y.

and:

> Pilot variance suggests the next eval needs roughly N tasks to detect a 2 pp
> gap at the chosen power.

Avoid:

> The observed gap is smaller than a same-data post-hoc MDE, so X is not better.

## Sources

- Evan Miller, [Adding Error Bars to Evals: A Statistical Approach to Language
  Model Evaluations](https://arxiv.org/abs/2411.00640).
- HELM, [Holistic Evaluation of Language
  Models](https://arxiv.org/abs/2211.09110).
- BetterBench, [Assessing AI Benchmarks, Uncovering Issues, and Establishing
  Best Practices](https://arxiv.org/abs/2411.12990).
- tau2-Bench, [Evaluating Conversational Agents in a Dual-Control
  Environment](https://arxiv.org/abs/2506.07982).
- Harness-Bench, [Measuring Harness Effects across Models in Realistic Agent
  Workflows](https://arxiv.org/abs/2605.27922).
- [Agentic Eval Methods Matrix](https://eternal-mesa-c3ce.here.now/).
