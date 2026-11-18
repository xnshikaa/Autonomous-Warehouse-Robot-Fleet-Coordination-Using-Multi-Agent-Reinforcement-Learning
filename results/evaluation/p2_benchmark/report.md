# Warehouse Fleet Scalability Evaluation

Manifest: `c343310263787f38`

20 matched episodes were evaluated per algorithm and fleet size. Intervals are 95% episode-bootstrap confidence intervals; paired tests use task-seed-matched episodes and Holm correction across valid tests.

## Metrics

| Algorithm | N | Completed/tasks | Completion (95% CI) | Delivery steps (95% CI) | Collisions/episode (95% CI) | Safety conflicts/episode | Throughput/100 steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| IQL | 5 | 9/100 | 0.090 (0.040-0.140) | 34.81 (28.75-42.19) | 0.000 (0.000-0.000) | 0.20 | 0.450 |
| QMIX | 5 | 83/100 | 0.830 (0.750-0.900) | 35.10 (33.10-37.08) | 0.000 (0.000-0.000) | 0.70 | 6.565 |
| Rule-Based | 5 | 34/100 | 0.340 (0.230-0.460) | 28.74 (26.12-30.96) | 0.000 (0.000-0.000) | 332.15 | 2.108 |
| IQL | 10 | 2/200 | 0.010 (0.000-0.025) | 26.50 (26.00-27.00) | 0.000 (0.000-0.000) | 0.45 | 0.100 |
| QMIX | 10 | 27/200 | 0.135 (0.080-0.190) | 37.52 (33.01-42.21) | 0.000 (0.000-0.000) | 1.65 | 1.350 |
| Rule-Based | 10 | 82/200 | 0.410 (0.320-0.500) | 29.70 (26.89-32.51) | 0.000 (0.000-0.000) | 611.25 | 4.100 |
| IQL | 20 | 0/400 | 0.000 (0.000-0.000) | n/a | 0.000 (0.000-0.000) | 704.70 | 0.000 |
| QMIX | 20 | 65/400 | 0.163 (0.133-0.195) | 35.05 (31.30-39.10) | 0.000 (0.000-0.000) | 16.45 | 3.250 |
| Rule-Based | 20 | 57/400 | 0.142 (0.090-0.205) | 27.45 (25.06-29.90) | 0.000 (0.000-0.000) | 1729.10 | 2.850 |

## QMIX Cooperative Completion Gap

| N | QMIX minus IQL completion rate (paired 95% CI) | Paired p | Holm-adjusted p |
|---:|---:|---:|---:|
| 5 | 0.740 (0.650-0.820) | 4.9998e-05 | 0.0033998 |
| 10 | 0.125 (0.070-0.180) | 0.00069997 | 0.022399 |
| 20 | 0.163 (0.133-0.195) | 4.9998e-05 | 0.0033998 |

## Findings and Limits

- The numeric N=5 QMIX point estimate meets the 80% completion target; its 95% interval extends below 80%.
- The evaluation manifest was created after these checkpoints were trained. It records the tested scenarios and hashes but does not prove a pre-training sealed holdout.
- The QMIX-IQL completion gap is not maintained as fleet size grows; the paired estimates are shown above.
- Dedicated corridor gate: 20 aggressive entries blocked, 0 intrusions in 20 episodes (PASS).
- No robot-robot contact events were observed. Safety-prevented conflicts are measured separately and must not be interpreted as physical collisions.
- Metrics requiring completed deliveries are unavailable for IQL at N=20 because it completed no tasks.
- See `scalability.png` for all fleet-size curves. These results describe the supplied checkpoints and do not establish convergence.
