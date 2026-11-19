# Warehouse Fleet Scalability Evaluation

Manifest: `bcc2d4585d74ebbb`

20 matched episodes were evaluated per algorithm and fleet size. Intervals are 95% episode-bootstrap confidence intervals; paired tests use task-seed-matched episodes and Holm correction across valid tests.

## Metrics

| Algorithm | N | Completed/tasks | Completion (95% CI) | Delivery steps (95% CI) | Collisions/episode (95% CI) | Blocked intent groups/episode (95% CI) | Safety conflicts/episode | Throughput/100 steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IQL | 5 | 75/100 | 0.750 (0.640-0.860) | 34.33 (32.43-36.18) | 0.000 (0.000-0.000) | 13.200 (0.300-33.600) | 26.00 | 6.138 |
| QMIX | 5 | 86/100 | 0.860 (0.770-0.940) | 36.85 (34.49-39.20) | 0.000 (0.000-0.000) | 0.650 (0.350-1.000) | 0.70 | 7.770 |
| Rule-Based | 5 | 36/100 | 0.360 (0.250-0.470) | 28.99 (26.44-31.14) | 0.000 (0.000-0.000) | 114.550 (93.400-136.751) | 322.15 | 2.208 |
| IQL | 10 | 102/200 | 0.510 (0.455-0.565) | 37.02 (33.45-40.94) | 0.000 (0.000-0.000) | 24.450 (11.700-39.500) | 49.70 | 5.100 |
| QMIX | 10 | 177/200 | 0.885 (0.820-0.945) | 35.72 (33.91-37.47) | 0.000 (0.000-0.000) | 3.050 (1.850-4.450) | 3.15 | 13.661 |
| Rule-Based | 10 | 82/200 | 0.410 (0.315-0.500) | 29.99 (27.29-32.74) | 0.000 (0.000-0.000) | 224.250 (194.399-253.000) | 611.25 | 4.100 |
| IQL | 20 | 5/400 | 0.013 (0.003-0.022) | 27.00 (22.80-29.80) | 0.000 (0.000-0.000) | 115.200 (91.599-139.351) | 332.15 | 0.250 |
| QMIX | 20 | 91/400 | 0.228 (0.200-0.255) | 32.42 (30.34-34.59) | 0.000 (0.000-0.000) | 3.750 (2.700-4.900) | 3.80 | 4.550 |
| Rule-Based | 20 | 60/400 | 0.150 (0.098-0.210) | 27.85 (25.44-30.22) | 0.000 (0.000-0.000) | 450.000 (409.549-494.401) | 1716.90 | 3.000 |

## QMIX Cooperative Completion Gap

| N | QMIX minus IQL completion rate (paired 95% CI) | Paired p | Holm-adjusted p |
|---:|---:|---:|---:|
| 5 | 0.110 (-0.050-0.260) | 0.21634 | 1 |
| 10 | 0.375 (0.305-0.450) | 4.9998e-05 | 0.0035998 |
| 20 | 0.215 (0.185-0.243) | 4.9998e-05 | 0.0035998 |

## Findings and Limits

- N=5 QMIX completion target (>=80%): PASS at 86.0%; 95% bootstrap CI 0.770-0.940.
- Used the supplied frozen scenario manifest `64e1a531970bba12870f9820f07730314f3dc4bd9de38dd2f2ab56f6af520b71`; the benchmark manifest records its hash.
- QMIX-IQL completion gap is not maintained across the included fleet sizes.
- Dedicated corridor gate: 20 aggressive entries blocked, 0 intrusions in 20 episodes (PASS).
- No physical robot-robot contacts were observed; safety-prevented conflicts are reported separately and are not physical collisions.
- N=10 physical QMIX/IQL collision-rate ratio is unavailable because IQL recorded zero physical collisions.
- N=10 blocked safety-override proxy ratio is 0.063; this does not substitute for the physical-collision requirement.
- Delivery-time metrics are based on completed tasks; unavailable groups are shown as n/a.
- See `scalability.png` for all fleet-size curves. These results describe the supplied checkpoints and do not establish convergence.
