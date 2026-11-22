# Warehouse Fleet Scalability Evaluation

Manifest: `64c7e1df5c82b379`

20 matched episodes were evaluated per algorithm and fleet size. Intervals are 95% episode-bootstrap confidence intervals; paired tests use task-seed-matched episodes and Holm correction across valid tests.

## Metrics

| Algorithm | N | Completed/tasks | Completion (95% CI) | Delivery steps (95% CI) | Collisions/episode (95% CI) | Blocked intent groups/episode (95% CI) | Safety conflicts/episode (95% CI) | Throughput/100 steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IQL | 5 | 74/100 | 0.740 (0.630-0.840) | 34.43 (32.88-35.95) | 0.000 (0.000-0.000) | 0.650 (0.350-1.000) | 0.70 (0.35-1.05) | 5.433 |
| QMIX | 5 | 67/100 | 0.670 (0.590-0.750) | 36.91 (34.86-39.05) | 0.000 (0.000-0.000) | 0.500 (0.200-0.800) | 0.50 (0.25-0.80) | 4.124 |
| Rule-Based | 5 | 44/100 | 0.440 (0.300-0.590) | 31.61 (29.18-33.91) | 0.000 (0.000-0.000) | 100.100 (74.050-124.901) | 286.75 (211.20-360.80) | 3.340 |
| IQL | 10 | 42/200 | 0.210 (0.165-0.255) | 36.75 (32.85-41.11) | 0.000 (0.000-0.000) | 24.850 (9.649-43.750) | 39.30 (11.55-75.10) | 2.100 |
| QMIX | 10 | 55/200 | 0.275 (0.200-0.350) | 35.76 (31.94-40.27) | 0.000 (0.000-0.000) | 5.800 (1.900-12.650) | 6.05 (1.90-13.40) | 2.750 |
| Rule-Based | 10 | 76/200 | 0.380 (0.290-0.470) | 28.63 (27.44-29.80) | 0.000 (0.000-0.000) | 224.700 (194.249-255.151) | 642.55 (570.30-715.00) | 3.800 |
| IQL | 20 | 0/400 | 0.000 (0.000-0.000) | n/a | 0.000 (0.000-0.000) | 677.800 (651.000-704.850) | 1569.80 (1538.15-1601.75) | 0.000 |
| QMIX | 20 | 0/400 | 0.000 (0.000-0.000) | n/a | 0.000 (0.000-0.000) | 5.450 (0.500-14.650) | 5.45 (0.50-14.70) | 0.000 |
| Rule-Based | 20 | 41/400 | 0.103 (0.053-0.173) | 26.44 (23.59-29.46) | 0.000 (0.000-0.000) | 463.850 (419.500-505.201) | 1797.50 (1683.75-1872.95) | 2.050 |

## QMIX Cooperative Completion Gap

| N | QMIX minus IQL completion rate (paired 95% CI) | Paired p | Holm-adjusted p |
|---:|---:|---:|---:|
| 5 | -0.070 (-0.200-0.050) | 0.27844 | 1 |
| 10 | 0.065 (-0.025-0.165) | 0.24194 | 1 |
| 20 | 0.000 (0.000-0.000) | 1 | 1 |

## Findings and Limits

- N=5 QMIX completion target (>=80%): FAIL at 67.0%; 95% bootstrap CI 0.590-0.750.
- No pre-created scenario manifest was supplied; this fixed-seed run is validation evidence, not a sealed held-out evaluation.
- QMIX-IQL completion gap is not maintained across the included fleet sizes.
- Dedicated corridor gate: 20 aggressive entries blocked, 0 intrusions in 20 episodes (PASS).
- No physical robot-robot contacts were observed; safety-prevented conflicts are reported separately and are not physical collisions.
- N=10 physical QMIX/IQL collision-rate ratio is unavailable because IQL recorded zero physical collisions.
- N=10 safety conflict ratio: QMIX 6.05 vs IQL 39.30 (ratio: 0.154, a 84.6% reduction); this safety conflict proxy is reported separately and does not substitute for the physical collision metric.
- N=10 blocked intent groups ratio: QMIX 5.80 vs IQL 24.85 (ratio: 0.233, a 76.7% reduction).
- IQL N=20 delivery-time metrics are unavailable because no task was completed.
- See `scalability.png` for all fleet-size curves. These results describe the supplied checkpoints and do not establish convergence.
