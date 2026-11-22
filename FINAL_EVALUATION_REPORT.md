# Project 25 final evaluation

The benchmark ran 180 episodes across QMIX, Independent Q-Learning (IQL) and Rule-Based. Each algorithm used the same 20 frozen task queues at each fleet size and had up to 100 steps per episode. The [benchmark report](results/project25_holdout/report.md) includes delivery times, throughput, physical collisions, safety conflicts, blocked move groups and paired tests. The [manifest](results/project25_holdout/manifest.json) identifies the exact models, scenarios and trace hashes.

## Task completion

Values are the share of assigned deliveries completed. Parentheses give 95% episode-bootstrap intervals.

| Fleet size | QMIX | IQL | Rule-Based |
|---:|---:|---:|---:|
| 5 | 78.0% (70.0–86.0%) | 78.0% (70.0–86.0%) | 46.0% (34.0–59.0%) |
| 10 | 37.0% (28.0–46.0%) | 23.0% (17.5–28.5%) | 39.5% (30.5–49.5%) |
| 20 | 0.75% (0.0–1.5%) | 0.25% (0.0–0.75%) | 13.75% (10.25–17.25%) |

At N=10, QMIX completed 28 more tasks than IQL, a 14 percentage point difference. The paired 95% interval was 3.5–24 points. The Holm-adjusted p-value was 0.694. Across the three fleet sizes, the results provide limited evidence of a consistent completion advantage.

## Acceptance results

| Requirement | Observed result | Status |
|---|---|---|
| NFR-1. QMIX completes at least 80% at N=5 | 78 of 100 deliveries | Target missed |
| NFR-3. Zero corridor intrusions across 20 attempts | All 20 attempts blocked, zero intrusions | Passed in simulation |
| NFR-4. QMIX physical collision rate at N=10 at most half of IQL | Both algorithms recorded zero contacts. The ratio has a zero denominator. | Inconclusive |
| NFR-5. Episode-bootstrap intervals | 95% intervals with 10,000 resamples for all nine configurations | Passed |

At N=10, QMIX triggered 2.15 safety conflict events per episode and IQL triggered 88.85, a 97.6% reduction in this simulator metric. Blocked move groups averaged 2.10 for QMIX and 40.65 for IQL. These metrics describe the interventions before contact.

The project defines the NFR-4 comparison in terms of robot–robot collisions.
The [Week 5 definitions](WEEK5_EVALUATION_INFO.md#definitions-based-on-the-existing-project)
distinguish contacts from overrides. The simulator resolves contested cells,
direct swaps and moves into newly blocked robots' cells before committing
positions. The trace generator then checks for collisions using the positions
the robots actually reached. With zero observed contacts for both controllers,
the NFR-4 ratio is 0/0 and the verdict is **INCONCLUSIVE**.

## N=20 behavior

In the final 20 steps of each N=20 episode, 384 of 400 QMIX robot trajectories and 338 of 400 IQL trajectories alternated between two cells on every step. Another 61 IQL trajectories were stationary. The selected models completed three QMIX deliveries and one IQL delivery across 400 tasks each.

The last 50 training episodes reached the step limit in both N=20 runs. QMIX averaged 0.04 delivery reward per episode alongside −23.05 in collision penalties. IQL averaged 0.17 and −45.80, respectively. The static action mask excludes shelves and corridors, and the safety layer resolves simultaneous move conflicts. The replayed trajectories show that the learned greedy policies can repeatedly choose legal moves without making delivery progress. The reward imbalance and oscillation are observed failure signals. The benchmark did not isolate their causal contribution.

The remediation trace check found that all 384 QMIX and all 338 IQL
oscillators executed their requested moves throughout the alternating segment.
381 QMIX and 337 IQL oscillators still had unfinished assignments. Task
assignments remained fixed in all 40 learned-policy episodes and all 40 timed
out at step 100. These observations rule out blocked moves and task
reassignment as explanations for those specific alternating segments.

The per-robot model sees current local occupancy, a static map, task geometry
and path costs. It receives no position history. Moving out and back cancels
the un-discounted distance-progress term within the same task phase. The
blocked-action penalty applies to obstacle and boundary overrides, while
robot conflicts use the collision term. Legal oscillations incur the ordinary
step penalty and receive no cycle-specific penalty. At the time limit the
Gymnasium adapter supplies the existing termination flag and
`truncated=False`, so training treats the horizon as an episode boundary.

### Cycle-escape validation

On 20 matched scenarios per controller and variant (seed 251720), the
intervention detected four alternating positions within one task phase. When
the next greedy action would continue the pattern, it selected an unoccupied,
statically legal alternative with the shortest remaining path when available.
The environment's safety resolver still executed every move.

| Weights | Deliveries, baseline → escape | Paired completion change (95% CI) | Alternating final-20 trajectories | Safety conflicts per episode |
|---|---:|---:|---:|---:|
| QMIX | 2 → 121 of 400 | +29.75 points (25.0–35.0) | 387 → 34 | 17.20 → 56.50 |
| IQL | 0 → 12 of 400 | +3.00 points (1.75–4.25) | 334 → 6 | 263.05 → 231.10 |

Breaking cycles improved completion in this validation experiment. The
intervention adds path-planning guidance, so the experiment cannot isolate
the effect of movement history alone. All episodes still timed out at step
100. The escape rule is an experimental inference heuristic, and the main
benchmark and live demo continue to use the original learned controllers.
See the [paired validation data](results/project25_remediation/n20_cycle_escape_validation.json).

### N=20 reward ablation

Two 40-episode QMIX runs started from the same episode-200 weights with
fresh replay and optimizer state, seed 252020 and a 100-step horizon. The
control kept the per-robot conflict penalty at −10. The comparison used −1.
The lower-penalty run completed 72 training deliveries versus 24 for the
control under exploration. Greedy validation on 20 scenarios with seed
252120 yielded the following results.

| Candidate | Completed/tasks | Completion (95% CI) | Safety conflicts per episode |
|---|---:|---:|---:|
| Original episode 200 | 9/400 | 2.25% (0.75–4.00%) | 9.05 |
| Control episode 20 | 2/400 | 0.50% (0.00–1.25%) | 16.30 |
| Control episode 40 | 0/400 | 0.00% (0.00–0.00%) | 10.95 |
| Lower penalty episode 20 | 0/400 | 0.00% (0.00–0.00%) | 45.35 |
| Lower penalty episode 40 | 0/400 | 0.00% (0.00–0.00%) | 196.40 |

The lower penalty increased exploratory deliveries in this training run.
Neither warm-start arm improved greedy validation completion, so the
original N=20 checkpoint remains selected and no new holdout was run.
These results cover one initialization and one training seed.
See the [reward validation report](results/project25_remediation/n20_reward_validation.json).

## N=5 remediation

The fresh validation used 20 scenarios with seed 251705 and a 100-step
horizon. All ten periodic snapshots from the recovered full run and five
compatible smoke snapshots were ranked. The final weights of each run
duplicate its last snapshot. Legacy 50-feature weights are incompatible
with the current 456-feature controller and were excluded.

| Candidate set | Best snapshot | Validation completion |
|---|---|---:|
| Recovered full run | Episode 600 | 65/100 |
| Recovered smoke run | Episode 50 | 42/100 |
| New 100-episode warm start | Episode 75 | 65/100 |

The selected episode-600 snapshot remains the demo checkpoint. Its SHA-256
is `89d4a600569b92e68dfa888cd0ff081539c0417b30772ff4c15b1ebcf8309fe8`.
The snapshot has 59,860 optimizer updates, schema version 2, five agents,
456 local features, a 55-feature global state and four actions.
The best warm-start snapshot tied it, so no replacement was selected for a
new holdout. The original 78/100 holdout result remains the NFR-1 verdict.
The [validation reports](results/project25_remediation/) preserve every
candidate score and digest.

## Provenance and reproduction

All 45 periodic snapshots were ranked on 20 validation scenarios per fleet using seed 20261120 and a 100-step horizon. Selection prioritized completion, then delivery time and blocked move groups. QMIX selected episodes 600, 1,000 and 200 for N=5, 10 and 20, respectively. IQL selected episodes 300, 500 and 400. The [validation reports](results/project25_validation/) list every candidate and its hash.

The final scenario file was generated after training and frozen before evaluation. Its 20 queues per fleet were checked against the task queues generated from training and known validation seeds. The earlier seed-999 runs remain exposed benchmark evidence. The higher-scoring historical P2 Cycle 2 benchmark references checkpoints absent from this checkout.

[Scalability plot](results/project25_holdout/scalability.png) · [Safety results](results/project25_holdout/safety_gate.json) · [Reproduction commands](TRAINING_AND_REPRODUCTION_GUIDE.md)

## Application release experiments

A 40-episode N=5 QMIX warm start used the episode-600 weights, a learning
rate of 0.00002, initial epsilon 0.12, and training seed 253505. On the
20 validation scenarios at seed 253605, episode 600 completed 76/100 tasks
and the best new snapshot completed 53/100. The original holdout verdict
remains 78/100. [N=5 ranking](results/project25_remediation/n5_low_lr_validation.json)

The N=20 validation at seed 253620 tested a cycle escape that also excludes
cells requested by other robots. For QMIX, the original controller, earlier
escape rule and contention-aware rule completed 4, 134 and 132 of 400 tasks,
respectively. They averaged 16.90, 65.40 and 48.40 safety conflicts per
episode. For IQL those counts were 3, 13 and 12 deliveries, with 241.10,
276.80 and 263.45 safety conflicts per episode. The contention-aware rule
reduced conflicts relative to the earlier escape rule. Both QMIX escape
rules still generated more conflicts than the original QMIX controller.
Neither experimental rule was selected for the live application or a new
holdout. [N=20 paired data](results/project25_remediation/n20_contention_validation.json)

The corridor gate was rerun after the live integration work and blocked all
20 attempted entries with zero intrusions.
