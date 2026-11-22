# Warehouse Fleet Scalability Evaluation

Manifest: `e47ae5704f026c96`

20 matched episodes were evaluated per algorithm and fleet size. Intervals are 95% episode-bootstrap confidence intervals; paired tests use task-seed-matched episodes and Holm correction across valid tests.

## Metrics

| Algorithm | N | Completed/tasks | Completion (95% CI) | Delivery steps (95% CI) | Collisions/episode (95% CI) | Blocked intent groups/episode (95% CI) | Safety conflicts/episode (95% CI) | Throughput/100 steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IQL | 5 | 78/100 | 0.780 (0.700-0.860) | 37.10 (33.87-40.69) | 0.000 (0.000-0.000) | 17.250 (4.850-30.201) | 34.65 (9.55-65.75) | 5.138 |
| QMIX | 5 | 78/100 | 0.780 (0.700-0.860) | 34.78 (32.65-36.94) | 0.000 (0.000-0.000) | 1.100 (0.650-1.550) | 1.10 (0.65-1.55) | 5.765 |
| Rule-Based | 5 | 46/100 | 0.460 (0.340-0.590) | 27.38 (24.89-30.03) | 0.000 (0.000-0.000) | 83.850 (67.700-98.250) | 274.70 (207.10-337.90) | 3.729 |
| IQL | 10 | 46/200 | 0.230 (0.175-0.285) | 36.31 (33.53-39.10) | 0.000 (0.000-0.000) | 40.650 (21.700-61.250) | 88.85 (46.20-135.90) | 2.300 |
| QMIX | 10 | 74/200 | 0.370 (0.280-0.460) | 34.40 (32.42-36.40) | 0.000 (0.000-0.000) | 2.100 (1.550-2.700) | 2.15 (1.60-2.85) | 3.700 |
| Rule-Based | 10 | 79/200 | 0.395 (0.305-0.495) | 29.71 (27.44-32.11) | 0.000 (0.000-0.000) | 180.550 (157.450-203.400) | 641.10 (551.35-723.76) | 3.950 |
| IQL | 20 | 1/400 | 0.003 (0.000-0.008) | 45.00 (45.00-45.00) | 0.000 (0.000-0.000) | 115.650 (98.150-134.101) | 249.90 (209.10-294.75) | 0.050 |
| QMIX | 20 | 3/400 | 0.008 (0.000-0.015) | 42.33 (33.00-49.00) | 0.000 (0.000-0.000) | 7.750 (3.150-15.101) | 7.75 (3.10-15.05) | 0.150 |
| Rule-Based | 20 | 55/400 | 0.138 (0.102-0.173) | 25.08 (22.89-27.23) | 0.000 (0.000-0.000) | 466.200 (412.249-516.801) | 1726.10 (1661.65-1789.60) | 2.750 |

## QMIX Cooperative Completion Gap

| N | QMIX minus IQL completion rate (paired 95% CI) | Paired p | Holm-adjusted p |
|---:|---:|---:|---:|
| 5 | 0.000 (-0.100-0.100) | 0.99395 | 1 |
| 10 | 0.140 (0.035-0.240) | 0.023799 | 0.69447 |
| 20 | 0.005 (-0.005-0.015) | 0.24929 | 1 |

## Findings and Limits

- N=5 QMIX completion target (>=80%): FAIL at 78.0%; 95% bootstrap CI 0.700-0.860.
- Used the supplied frozen scenario manifest `ddc457ee542f3f6076058722fb8a130c022828367ac23fd5c011ea1553f6c7bf`; the benchmark manifest records its hash.
- QMIX-IQL completion gap is not maintained across the included fleet sizes.
- Dedicated corridor gate: 20 aggressive entries blocked, 0 intrusions in 20 episodes (PASS).
- No physical robot-robot contacts were observed; safety-prevented conflicts are reported separately and are not physical collisions.
- N=10 physical QMIX/IQL collision-rate ratio is unavailable because IQL recorded zero physical collisions.
- N=10 safety conflict ratio: QMIX 2.15 vs IQL 88.85 (ratio: 0.024, a 97.6% reduction); this safety conflict proxy is reported separately and does not substitute for the physical collision metric.
- N=10 blocked intent groups ratio: QMIX 2.10 vs IQL 40.65 (ratio: 0.052, a 94.8% reduction).
- Delivery-time metrics are based on completed tasks; unavailable groups are shown as n/a.
- See `scalability.png` for all fleet-size curves. These results describe the supplied checkpoints and do not establish convergence.
