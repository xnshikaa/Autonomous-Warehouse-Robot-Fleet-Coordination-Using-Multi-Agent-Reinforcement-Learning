# Warehouse Fleet Scalability Evaluation

Manifest: `47c08ace928ea885`

20 matched episodes were evaluated per algorithm and fleet size. Intervals are 95% episode-bootstrap confidence intervals; paired tests use task-seed-matched episodes and Holm correction across valid tests.

## Metrics

| Algorithm | N | Completed/tasks | Completion (95% CI) | Delivery steps (95% CI) | Collisions/episode (95% CI) | Blocked intent groups/episode (95% CI) | Safety conflicts/episode | Throughput/100 steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IQL | 5 | 66/100 | 0.660 (0.560-0.760) | 34.59 (32.41-36.93) | 0.000 (0.000-0.000) | 15.750 (4.050-30.300) | 30.95 | 4.125 |
| QMIX | 5 | 83/100 | 0.830 (0.750-0.900) | 37.02 (35.09-38.86) | 0.000 (0.000-0.000) | 0.500 (0.250-0.750) | 0.50 | 6.267 |
| Rule-Based | 5 | 51/100 | 0.510 (0.350-0.670) | 31.85 (29.35-34.43) | 0.000 (0.000-0.000) | 73.900 (51.249-96.350) | 240.70 | 4.695 |
| IQL | 10 | 86/200 | 0.430 (0.365-0.490) | 35.06 (32.79-37.52) | 0.000 (0.000-0.000) | 45.750 (26.499-66.700) | 113.15 | 4.300 |
| QMIX | 10 | 100/200 | 0.500 (0.425-0.580) | 40.64 (37.86-43.81) | 0.000 (0.000-0.000) | 3.150 (2.450-3.850) | 3.35 | 5.480 |
| Rule-Based | 10 | 84/200 | 0.420 (0.305-0.540) | 31.61 (30.21-33.00) | 0.000 (0.000-0.000) | 182.250 (145.399-218.605) | 586.40 | 5.430 |
| IQL | 20 | 3/400 | 0.008 (0.000-0.015) | 27.67 (25.00-31.00) | 0.000 (0.000-0.000) | 140.150 (122.099-159.700) | 392.80 | 0.150 |
| QMIX | 20 | 86/400 | 0.215 (0.182-0.247) | 34.04 (31.87-36.32) | 0.000 (0.000-0.000) | 8.250 (3.700-16.600) | 8.25 | 4.300 |
| Rule-Based | 20 | 60/400 | 0.150 (0.105-0.197) | 28.02 (26.33-29.67) | 0.000 (0.000-0.000) | 471.700 (426.649-517.750) | 1710.75 | 3.000 |

## QMIX Cooperative Completion Gap

| N | QMIX minus IQL completion rate (paired 95% CI) | Paired p | Holm-adjusted p |
|---:|---:|---:|---:|
| 5 | 0.170 (0.050-0.280) | 0.019749 | 0.58742 |
| 10 | 0.070 (-0.020-0.160) | 0.15129 | 1 |
| 20 | 0.208 (0.172-0.243) | 4.9998e-05 | 0.0035998 |

## Findings and Limits

- N=5 QMIX completion target (>=80%): PASS at 83.0%; 95% bootstrap CI 0.750-0.900.
- Used the supplied frozen scenario manifest `877e21e36bf45cb717d3e456b3d046b2545ead5a9e66347767a13f0da248e5f4`; the benchmark manifest records its hash.
- QMIX-IQL completion gap is not maintained across the included fleet sizes.
- Dedicated corridor gate: 20 aggressive entries blocked, 0 intrusions in 20 episodes (PASS).
- No robot-robot contact events were observed. Safety-prevented conflicts are reported separately and are not physical collisions.
- Delivery-time metrics are based on completed tasks; unavailable groups are shown as n/a.
- See `scalability.png` for all fleet-size curves. These results describe the supplied checkpoints and do not establish convergence.
