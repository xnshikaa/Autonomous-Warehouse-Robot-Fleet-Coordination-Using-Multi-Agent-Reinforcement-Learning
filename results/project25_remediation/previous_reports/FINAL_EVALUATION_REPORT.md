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

## N=20 behavior

In the final 20 steps of each N=20 episode, 384 of 400 QMIX robot trajectories and 338 of 400 IQL trajectories alternated between two cells on every step. Another 61 IQL trajectories were stationary. The selected models completed three QMIX deliveries and one IQL delivery across 400 tasks each.

The last 50 training episodes reached the step limit in both N=20 runs. QMIX averaged 0.04 delivery reward per episode alongside −23.05 in collision penalties. IQL averaged 0.17 and −45.80, respectively. The static action mask excludes shelves and corridors, and the safety layer resolves simultaneous move conflicts. The replayed trajectories show that the learned greedy policies can repeatedly choose legal moves without making delivery progress. The reward imbalance and oscillation are observed failure signals. The benchmark did not isolate their causal contribution.

## Provenance and reproduction

All 45 periodic snapshots were ranked on 20 validation scenarios per fleet using seed 20261120 and a 100-step horizon. Selection prioritized completion, then delivery time and blocked move groups. QMIX selected episodes 600, 1,000 and 200 for N=5, 10 and 20, respectively. IQL selected episodes 300, 500 and 400. The [validation reports](results/project25_validation/) list every candidate and its hash.

The final scenario file was generated after training and frozen before evaluation. Its 20 queues per fleet were checked against the task queues generated from training and known validation seeds. The earlier seed-999 runs remain exposed benchmark evidence. The higher-scoring historical P2 Cycle 2 benchmark references checkpoints absent from this checkout.

[Scalability plot](results/project25_holdout/scalability.png) · [Safety results](results/project25_holdout/safety_gate.json) · [Reproduction commands](TRAINING_AND_REPRODUCTION_GUIDE.md)
