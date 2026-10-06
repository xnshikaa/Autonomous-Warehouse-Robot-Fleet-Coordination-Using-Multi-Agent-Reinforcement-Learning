# Member 3 Handoff — QMIX Results for Comparative Analysis
## Week 6 AI Track — Member 2 to Member 3

This document provides everything Member 3 needs to compare the QMIX agent
against the Rule-Based baseline and IQL (if available).

---

## What Member 2 Has Produced

- 4 real QMIX training runs (Runs A–D) with hyperparameter variations
- A Best Stable QMIX Checkpoint (Run D)
- Full MLflow experiment database with logged parameters and metrics (`mlflow.db`)
- An honest account of extended-run instability (150 episodes, 200 steps)

## What Is NOT Provided

- Rule-Based baseline results — these must come from their own implementation
- IQL results — these must come from whoever implemented IQL
- Any fabricated comparison numbers

---

## QMIX Configuration Summary

| Property | Value |
|----------|-------|
| Algorithm | Custom PyTorch QMIX (NOT EPyMARL) |
| Agents | 5 |
| Observation dimension | 50-D per agent |
| Global state dimension | 250-D (5 agents x 50-D concatenated) |
| Action space | Discrete(4): 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT |
| Environment | 20x20 Warehouse GridWorld |
| Episodes (tuning) | 60 per run |
| Max steps/episode (tuning) | 40 |

---

## Best Stable Run — Run D

| Metric | Value |
|--------|-------|
| Run ID | `3f1553055ba04a13bf297b2d3b97e746` |
| Learning rate | 8e-4 |
| Batch size | 64 |
| Epsilon start / end | 1.0 / 0.02 |
| Target update interval | 5 |
| Mean reward (60 ep) | -406.75 |
| Best episode reward | -34.74 |
| Early mean reward (1-10) | -488.64 |
| Late mean reward (51-60) | -182.97 |
| Reward improvement | +305.67 |
| Conflict metric | Safety overrides decreased by 66.5 per episode (103.0 -> 36.5) |
| Initial loss | 310.75 |
| Final loss | 194.77 (stable, converged) |

---

## All Runs — Authoritative Comparison Table

| Run | Run ID | LR | Batch | $\epsilon_{\text{end}}$ | Sync | Mean Reward | Best Reward | Early Reward (1-10) | Late Reward (51-60) | Reward Delta | Early Overrides/ep | Late Overrides/ep | Safety Override Delta | Initial Loss | Final Loss | Stability |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | `2395b1fc83394d198ea37e8327edc5bd` | 1e-3 | 32 | 0.05 | 10 | -644.67 | -270.56 | -475.21 | -880.15 | -404.94 | 100.1 | 178.7 | Overrides increased by 78.6 | 348.45 | 4,656,093.01 | UNSTABLE |
| **B** | `cd89dc76619e4b60af8da1b5c1ade6ca` | 3e-4 | 32 | 0.05 | 10 | -399.71 | -144.10 | -489.64 | -304.97 | +184.67 | 103.2 | 59.5 | Safety overrides decreased by 43.7 per episode | 360.96 | 189.43 | STABLE |
| **C** | `6ca54518b614411ebef03112cdf561ad` | 5e-4 | 32 | 0.01 | 5 | -736.97 | -270.42 | -483.62 | -925.58 | -441.95 | 101.4 | 191.2 | Overrides increased by 89.8 | 364.33 | 6,768,464.12 | UNSTABLE |
| **D** | `3f1553055ba04a13bf297b2d3b97e746` | 8e-4 | 64 | 0.02 | 5 | -406.75 | -34.74 | -488.64 | -182.97 | +305.67 | 103.0 | 36.5 | Safety overrides decreased by 66.5 per episode | 310.75 | 194.77 | **BEST STABLE** |

*Notes:*
- Reward improvement = Late Mean Reward (episodes 51–60) minus Early Mean Reward (episodes 1–10).
- Safety override delta = Early minus Late overrides/episode (positive = improving/decreasing overrides).

---

## Extended Run (150 episodes, 200 steps) — Diverged

The extended run used Run D hyperparameters at 200 steps/episode and diverged (loss exploded past $10^6$ by Episode 7, ending at $4.07 \times 10^8$; final reward -4,684.10).
DO NOT use extended run metrics in comparative analysis.
See `WEEK_6_AI_REPORT.md` Section 6 for root cause explanation.

---

## Checkpoint Location

**Run D Best Stable QMIX Checkpoint:**
`artifacts/checkpoints/qmix/3f1553055ba04a13bf297b2d3b97e746/latest/checkpoint_state.pt`

This checkpoint contains:
  - `agent_network` (AgentQNetwork state dict)
  - `mixer` (QMIXMixer state dict)
  - `target_agent_network`
  - `target_mixer`
  - `optimizer`
  - `step`: 600
  - `loss`: 194.77

---

## MLflow Access

All 4 training runs are logged in MLflow SQLite database: `mlflow.db` in workspace root.
Run IDs:
- Run A: `2395b1fc83394d198ea37e8327edc5bd`
- Run B: `cd89dc76619e4b60af8da1b5c1ade6ca`
- Run C: `6ca54518b614411ebef03112cdf561ad`
- Run D: `3f1553055ba04a13bf297b2d3b97e746`
- Extended: `44fb3ff6cf7a4eb0965b0b92bb6dbd67`

---

## Suggested Comparison Protocol for Member 3

1. **Compare at the same scale:**
   Ensure baseline evaluations use 5 agents and 40 steps/episode to match the validated QMIX setting.
2. **Key metrics to compare:**
   - Mean episode reward (Run D: -406.75; best: -34.74)
   - Safety overrides per episode (Run D late: 36.5 overrides/episode)
   - Convergence stability (finite loss vs exploding loss)
