# Warehouse Fleet Scalability Evaluation

Manifest: `868412fb49b0f395`

20 matched episodes were evaluated per algorithm and fleet size. Intervals are 95% episode-bootstrap confidence intervals; paired tests use task-seed-matched episodes and Holm correction across valid tests.

## Metrics

| Algorithm | N | Completed/tasks | Completion (95% CI) | Delivery steps (95% CI) | Collisions/episode (95% CI) | Blocked intent groups/episode (95% CI) | Safety conflicts/episode (95% CI) | Throughput/100 steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IQL | 5 | 74/100 | 0.740 (0.630-0.840) | 34.43 (32.88-35.95) | 0.000 (0.000-0.000) | 0.650 (0.350-1.000) | 0.70 (0.35-1.05) | 5.433 |
| QMIX | 5 | 67/100 | 0.670 (0.590-0.750) | 36.91 (34.86-39.05) | 0.000 (0.000-0.000) | 0.500 (0.200-0.800) | 0.50 (0.25-0.80) | 4.124 |
| Rule-Based | 5 | 44/100 | 0.440 (0.300-0.590) | 31.61 (29.18-33.91) | 0.000 (0.000-0.000) | 100.100 (74.050-124.901) | 286.75 (211.20-360.80) | 3.340 |
| IQL | 10 | 2/200 | 0.010 (0.000-0.025) | 31.00 (30.00-32.00) | 0.000 (0.000-0.000) | 43.200 (27.700-58.950) | 99.00 (59.00-140.40) | 0.100 |
| QMIX | 10 | 0/200 | 0.000 (0.000-0.000) | n/a | 0.000 (0.000-0.000) | 1.050 (0.700-1.400) | 1.05 (0.70-1.40) | 0.000 |
| Rule-Based | 10 | 76/200 | 0.380 (0.290-0.470) | 28.63 (27.44-29.80) | 0.000 (0.000-0.000) | 224.700 (194.249-255.151) | 642.55 (570.30-715.00) | 3.800 |

## QMIX Cooperative Completion Gap

| N | QMIX minus IQL completion rate (paired 95% CI) | Paired p | Holm-adjusted p |
|---:|---:|---:|---:|
| 5 | -0.070 (-0.200-0.050) | 0.27844 | 1 |
| 10 | -0.010 (-0.025-0.000) | 0.49998 | 1 |

## Findings and Limits

- N=5 QMIX completion target (>=80%): FAIL at 67.0%; 95% bootstrap CI 0.590-0.750.
- No pre-created scenario manifest was supplied; this fixed-seed run is validation evidence, not a sealed held-out evaluation.
- QMIX-IQL completion gap is maintained or increases across the included fleet sizes.
- Dedicated corridor gate: 20 aggressive entries blocked, 0 intrusions in 20 episodes (PASS).
- No physical robot-robot contacts were observed; safety-prevented conflicts are reported separately and are not physical collisions.
- N=10 physical QMIX/IQL collision-rate ratio is unavailable because IQL recorded zero physical collisions.
- N=10 safety conflict ratio: QMIX 1.05 vs IQL 99.00 (ratio: 0.011, a 98.9% reduction); this safety conflict proxy is reported separately and does not substitute for the physical collision metric.
- N=10 blocked intent groups ratio: QMIX 1.05 vs IQL 43.20 (ratio: 0.024, a 97.6% reduction).
- Delivery-time metrics are based on completed tasks; unavailable groups are shown as n/a.
- See `scalability.png` for all fleet-size curves. These results describe the supplied checkpoints and do not establish convergence.
