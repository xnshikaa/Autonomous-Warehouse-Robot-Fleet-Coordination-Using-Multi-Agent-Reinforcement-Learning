# Week 6 AI Report — Custom PyTorch QMIX Training
## Autonomous Warehouse Robot Fleet Coordination

**Member:** Member 2 (Week 6 AI Track)  
**Week:** 6 — QMIX Implementation, Training, and Hyperparameter Evaluation  
**Algorithm:** Custom PyTorch QMIX (NOT EPyMARL)  
**Environment:** 20x20 Warehouse GridWorld, 5 Robots, Discrete(4) Actions, 50-D Observations  

---

## Table of Contents
1. [Task Overview](#1-task-overview)
2. [System Architecture](#2-system-architecture)
3. [Training Readiness Validation](#3-training-readiness-validation)
4. [Hyperparameter Experiments](#4-hyperparameter-experiments)
5. [Best Stable QMIX Checkpoint](#5-best-stable-qmix-checkpoint)
6. [Extended Training — Instability Finding](#6-extended-training--instability-finding)
7. [Learning Progression Analysis](#7-learning-progression-analysis)
8. [Limitations](#8-limitations)
9. [Future Work](#9-future-work)
10. [Deliverables](#10-deliverables)

---

## 1. Task Overview

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Train QMIX with 5 robots — Initial QMIX Model | COMPLETE |
| Task 2 | Hyperparameter tuning — Better Rewards | COMPLETE |

QMIX training infrastructure and real training were successfully validated. Multiple hyperparameter configurations were evaluated, with Run D producing the best stable short-horizon result. A longer-horizon experiment exposed reward-scale/horizon instability.

All training used the actual production MARL stack — real environment, real rewards, real backpropagation, real MLflow tracking, real checkpointing. No synthetic metrics.

**This implementation is custom PyTorch QMIX. It is not EPyMARL.**  
The Three.js frontend visualization runs separately and is not live-controlled by the trained QMIX model unless the QMIXController is explicitly loaded with a checkpoint.

---

## 2. System Architecture

### A. Implementation

The QMIX implementation consists of the following connected components:

```
WarehouseGymEnv (Gymnasium wrapper)
    |
    |-- reset() --> observations: ndarray (5, 50)
    |-- step(actions) --> next_obs, reward, done, info
    |
    v
QMIXTrainer (episode loop)
    |
    |-- select_actions() --> QMIXLearner.select_actions(obs, epsilon)
    |         |
    |         v
    |      AgentQNetwork: Linear(50->64)->ReLU->Linear(64->64)->ReLU->Linear(64->4)
    |      Output: Q(s_i, a) for a in {0=UP, 1=DOWN, 2=LEFT, 3=RIGHT}
    |
    |-- replay_buffer.push(obs, state, actions, reward, next_obs, next_state, done)
    |
    |-- replay_buffer.sample(batch_size) --> mini-batch tensors
    |
    |-- learner.train_batch(batch)
    |         |
    |         v
    |      QMIXMixer: state -> HyperNetworks -> non-negative weights
    |      Q_tot = f(Q_agent_1, ..., Q_agent_5, global_state)
    |      TD loss = MSE(Q_tot, r + gamma*(1-done)*Q_tot_target)
    |      optimizer.step() with grad clip norm=10.0
    |
    |-- checkpoint_manager.save_checkpoint(payload, step)
    |-- tracker.log_metrics(metrics, step)
```

### B. Key Properties

- Monotonicity constraint via `torch.abs()` on hypernetwork mixing weights — IGM condition holds
- Shared agent network — all 5 robots share one Q-network (parameter efficiency)
- Hard target network updates at configurable interval
- Gradient clipping at `max_norm=10.0`
- Epsilon-greedy exploration with linear decay across episodes

### C. Component File Map

| File | Role |
|------|------|
| `src/marl/qmix_network.py` | AgentQNetwork + QMIXMixer |
| `src/marl/qmix_learner.py` | select_actions, train_batch, target sync |
| `src/marl/qmix_trainer.py` | Episode loop, MLflow, checkpointing |
| `src/marl/replay_buffer.py` | Circular buffer, state_dict, mini-batch sample |
| `src/marl/gymnasium_env.py` | WarehouseGymEnv (Gymnasium wrapper) |
| `src/marl/environment_adapter.py` | MARLEnvironmentAdapter |
| `src/marl/observation_encoder.py` | 50-D observation encoder |
| `src/marl/qmix_controller.py` | Inference bridge (checkpoint -> live actions) |
| `python_infra/checkpoint_manager.py` | Save/load/resume |
| `python_infra/experiment_tracker.py` | MLflow wrapper |
| `python_infra/config.py` | ExperimentConfig + TrainingMetrics |

---

## 3. Training Readiness Validation

An 18-point readiness test was executed against all real components prior to full training.
The test was executed twice: before training (Week 6 start) and during final closeout.

### Readiness Test Results

| # | Verification Item | Result |
|---|------------------|--------|
| 1 | Environment initializes | PASS |
| 2 | 5 agents receive valid 50-D observations, shape (5,50), no NaN | PASS |
| 3 | Actions generated via epsilon-greedy — shape (5,), values in {0,1,2,3} | PASS |
| 4 | Environment steps successfully | PASS |
| 5 | Real rewards enter replay buffer | PASS |
| 6 | Replay buffer reaches warm-up threshold | PASS |
| 7 | Mini-batch sampling works (buffer >= batch_size) | PASS |
| 8 | Agent Q-network forward pass produces shape (5,4), all finite | PASS |
| 9 | TD loss is finite (initial=386.1, final=205.2 in validation run) | PASS |
| 10 | Gradients produced — optimizer updates > 0 | PASS |
| 11 | Optimizer changes model parameters — L2 param movement = 10.07 | PASS |
| 12 | Target networks synchronize at configured interval | PASS |
| 13 | Episode terminates correctly (6/6 episodes completed) | PASS |
| 14 | MLflow records actual training run with real run_id | PASS |
| 15 | Real training metrics logged per episode | PASS |
| 16 | Real QMIX checkpoint saved to disk | PASS |
| 17 | Checkpoint loads into fresh trainer instance | PASS |
| 18 | Training resumes from loaded checkpoint (updates increment) | PASS |

**OVERALL: 18/18 CHECKS PASSED**

---

## 4. Hyperparameter Experiments

Four training runs were executed sequentially. Each run used real backpropagation,
real environment rewards, and real MLflow tracking (`mlflow.db`). All results are from actual
training logs and MLflow records — no values have been fabricated or adjusted.

### Experimental Configuration (all 4 runs)
- Fleet Size: 5 agents
- Observation Dimension: 50-D per agent
- Global State Dimension: 250-D (concatenated agent observations)
- Action Space: `Discrete(4)` (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
- Episodes per Run: 60
- Max Steps per Episode: 40
- Replay Buffer Capacity: 5,000 transitions
- Replay Warm-up Size: 32 transitions
- Gradient Clip Norm: 10.0
- Discount Factor ($\gamma$): 0.99
- Epsilon Start: 1.0

### Authoritative Results Comparison Table

| Run | Run ID | LR | Batch | $\epsilon_{\text{end}}$ | Sync | Mean Reward | Best Reward | Early Reward (1-10) | Late Reward (51-60) | Reward Delta | Early Overrides/ep | Late Overrides/ep | Safety Override Delta | Initial Loss | Final Loss | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | `2395b1fc83394d198ea37e8327edc5bd` | 1e-3 | 32 | 0.05 | 10 | -644.67 | -270.56 | -475.21 | -880.15 | -404.94 | 100.1 | 178.7 | Overrides increased by 78.6 | 348.45 | 4,656,093.01 | UNSTABLE (Loss exploded) |
| **B** | `cd89dc76619e4b60af8da1b5c1ade6ca` | 3e-4 | 32 | 0.05 | 10 | -399.71 | -144.10 | -489.64 | -304.97 | +184.67 | 103.2 | 59.5 | Safety overrides decreased by 43.7 per episode | 360.96 | 189.43 | STABLE (Converged) |
| **C** | `6ca54518b614411ebef03112cdf561ad` | 5e-4 | 32 | 0.01 | 5 | -736.97 | -270.42 | -483.62 | -925.58 | -441.95 | 101.4 | 191.2 | Overrides increased by 89.8 | 364.33 | 6,768,464.12 | UNSTABLE (Loss exploded) |
| **D** | `3f1553055ba04a13bf297b2d3b97e746` | 8e-4 | 64 | 0.02 | 5 | -406.75 | -34.74 | -488.64 | -182.97 | +305.67 | 103.0 | 36.5 | Safety overrides decreased by 66.5 per episode | 310.75 | 194.77 | **BEST STABLE QMIX CHECKPOINT** |

*Definitions:*
- **Reward Delta:** Late Mean Reward (episodes 51–60) minus Early Mean Reward (episodes 1–10). Positive values indicate policy improvement.
- **Safety Override Delta:** Difference between early and late safety interventions per episode. For Run D, safety overrides decreased by 66.5 per episode (falling from 103.0 down to 36.5).

### Run-by-Run Analysis

**Run A — Baseline (`lr=1e-3`, batch=32, sync=10):**
Catastrophic loss explosion. Q-values grew without bound as the learning rate was too high for unscaled step penalties.
Rewards degraded from early mean -475.21 to late mean -880.15 (delta -404.94). Overrides increased from 100.1 to 178.7. Final loss reached 4,656,093.01.

**Run B — Lower LR (`lr=3e-4`, batch=32, sync=10):**
First stable convergence. Loss decreased stably from 360.96 to 189.43.
Reward improved steadily: late mean -304.97 vs early mean -489.64 (+184.67 delta).
Safety overrides decreased by 43.7 per episode (from 103.2 down to 59.5).

**Run C — Lower Epsilon + Faster Sync (`lr=5e-4`, batch=32, sync=5):**
With smaller exploration (`epsilon_end=0.01`) and faster sync (every 5 episodes), agents prematurely locked onto sub-optimal actions before coordinated fleet policies formed.
Loss exploded to 6,768,464.12, rewards regressed to -925.58, and overrides increased by 89.8.

**Run D — Refined Combination (`lr=8e-4`, batch=64, sync=5):**
Doubling batch size (32 $\rightarrow$ 64) stabilized stochastic gradients across transitions.
This allowed the faster target network update interval (every 5 episodes) to track current policy improvements effectively.
Achieved the highest reward progression: best episode reward of **-34.74**, late reward of **-182.97** (+305.67 improvement), and safety overrides decreased by **66.5 per episode** (103.0 down to 36.5) with final loss converging cleanly at **194.77**.

---

## 5. Best Stable QMIX Checkpoint

**Run D is the Best Stable QMIX Checkpoint for the validated short-horizon configuration.**

| Property | Value |
|----------|-------|
| MLflow Run ID | `3f1553055ba04a13bf297b2d3b97e746` |
| Checkpoint Path | `artifacts/checkpoints/qmix/3f1553055ba04a13bf297b2d3b97e746/latest/` |
| Checkpoint File | `checkpoint_state.pt` |
| File Size | 54,657 bytes |
| Learning Rate | 8e-4 |
| Batch Size | 64 |
| Epsilon End | 0.02 |
| Target Update Interval | 5 |
| Episodes Trained | 60 |
| Max Steps per Episode | 40 |
| Mean Reward (60 ep) | -406.75 |
| Best Episode Reward | -34.74 |
| Reward Improvement (Early vs Late) | +305.67 |
| Safety Conflict Progression | Safety overrides decreased by 66.5 per episode (103.0 -> 36.5) |
| Initial Training Loss | 310.75 |
| Final Training Loss | 194.77 (stable, converged) |

**Checkpoint Validation (Closeout Verification — 13/13 Sub-checks PASS):**
- Checkpoint directory exists: CONFIRMED
- `checkpoint_state.pt` file present: CONFIRMED
- `agent_network` state dict loaded: CONFIRMED
- `mixer` hypernetwork state dict loaded: CONFIRMED
- `target_agent_network` state dict loaded: CONFIRMED
- `target_mixer` state dict loaded: CONFIRMED
- `optimizer` state dict loaded: CONFIRMED
- `step` metadata loaded (600): CONFIRMED
- `loss` metadata loaded (194.77): CONFIRMED
- Forward pass after loading: CONFIRMED — Q-values shape (5,4), all finite
- Greedy inference (`epsilon=0.0`): CONFIRMED — actions in {0,1,2,3}
- Exploration action selection: CONFIRMED
- Parameter integrity verified: CONFIRMED

---

## 6. Extended Training — Instability Finding

An extended training run was executed using the Run D hyperparameter configuration over 150 episodes with an increased episode length of 200 steps per episode.
**This run diverged and is NOT a production model.** It is a validated research finding demonstrating horizon/reward-scale sensitivity.

### Extended Run Actual Results

| Property / Metric | Value from Artifacts & MLflow |
|-------------------|-------------------------------|
| Run ID | `44fb3ff6cf7a4eb0965b0b92bb6dbd67` |
| Run Name | `qmix_final_best_model` |
| Total Episodes | 150 |
| Max Steps per Episode | 200 |
| Learning Rate | 0.0008 (8e-4) |
| Batch Size | 64 |
| Replay Buffer Capacity | 10,000 |
| Replay Warm-up Size | 64 |
| Epsilon Start / End | 1.0 / 0.02 |
| Target Update Interval | 5 |
| Gradient Clip Norm | 10.0 |
| Total Optimizer Updates | 29,937 |
| Initial Loss (Episode 1) | 224.60 |
| Instability Onset | Episode 5 (loss jumped to 38,376.72; reward peaked at -1456.58) |
| Severe Loss Escalation | Episode 7 (loss reached 4,688,763.57; exceeding $10^6$) |
| Extreme Loss Peak | Episodes 30–50 (loss peaked past $9 \times 10^{16}$) |
| Final Loss (Episode 150) | 406,691,151.87 ($4.07 \times 10^8$) |
| Mean Loss Across Updates | $9.30 \times 10^{15}$ |
| Mean Reward (150 ep) | -3,470.66 |
| Best Episode Reward | -1,456.58 (Episode 5) |
| Worst Episode Reward | -5,515.84 |
| Early Mean Reward (Episodes 1–37) | -2,254.02 |
| Late Mean Reward (Episodes 113–150)| -4,535.20 |
| Reward Improvement | -2,281.18 (Severe regression) |
| Final Episode Reward | -4,684.10 |
| Early Conflicts/Episode | 478.6 |
| Late Conflicts/Episode | 872.0 |
| Conflict Change | -393.4 (Conflicts increased by 393.4 per episode) |

### Technical Cause of Divergence

| Dimension | Tuning Runs (A–D) | Extended Run |
|---|---|---|
| Max steps per episode | 40 | 200 (5x longer) |
| Total episodes | 60 | 150 (2.5x more) |
| Cumulative negative penalties | -150 to -700 | -1,500 to -5,500 |
| Number of conflicts per episode | 30 to 190 | 400 to 960 |
| TD target magnitude | Moderate | Overwhelmingly negative |

**Mechanism:**  
Over the 200-step episode horizon, 5 agents continually accumulated dense negative penalties for movements, wait times, out-of-bounds attempts, and collision overrides (averaging 695 safety overrides per episode). Because rewards are unnormalized and unclipped, compounding these negative rewards over 200 timesteps with $\gamma=0.99$ caused the Bellman target values to explode. The QMIX hypernetworks, which enforce non-negative weights via absolute values, amplified these extreme target magnitudes into runaway TD errors that gradient clipping (`max_norm=10.0`) could not stabilize.

**Outcome:**  
The Run D checkpoint (`3f1553055ba04a13bf297b2d3b97e746`) remains the authoritative, unaltered, and validated checkpoint. The extended checkpoint was rejected.

---

## 7. Learning Progression Analysis

### Phase 1 — Exploration (early episodes, $\epsilon \approx 1.0$)
- Random actions dominate; high safety override count (~103 overrides/episode).
- Replay buffer warms up; initial TD loss starts near ~310–360.
- Individual robots explore grid boundaries and obstacle cells.

### Phase 2 — Transition (mid-episodes, $\epsilon$ decaying)
- Q-values begin differentiating favorable corridors from congestion zones.
- Safety override count steadily drops as penalties discourage collisions.
- Loss stabilizes into the 190–250 range.

### Phase 3 — Exploitation (late episodes, $\epsilon \approx 0.02$)
- Near-greedy coordination emerges in Run D.
- Safety overrides decreased by 66.5 per episode (dropping to 36.5 late).
- Reward peaks at -34.74 with late-stage average of -182.97.

---

## 8. Limitations

These are honest research limitations of the current Week 6 simulation scope:

1. **Short horizon only validated:** Run D is validated for 40 steps/episode. Longer horizons (200 steps) destabilize without reward normalization.
2. **Task completions:** In the 40-step short horizon, robots do not complete full 20x20 pickup-to-delivery round trips; longer horizons are required to complete deliveries.
3. **Unnormalized reward scale:** Raw negative step and conflict penalties accumulate linearly with horizon length.
4. **Shared policy network:** All 5 robots share one parameter set (homogeneous policies).
5. **Decoupled frontend:** The Three.js WebGL interface operates separately from the QMIX backend; live driving requires connecting `QMIXController`.

---

## 9. Future Work

Out-of-scope for Week 6; reserved for subsequent iterations:
- Reward normalization / tanh clipping for longer episode horizons.
- Stable training at 200+ steps/episode.
- Task completion reward shaping.
- Prioritized Experience Replay (PER).
- Recurrent QMIX (QMIX-GRU) for partial observability.
- Live integration with Three.js visualization.

---

## 10. Deliverables

| Deliverable | Location | Status |
|-------------|----------|--------|
| Best Stable QMIX Checkpoint (Run D) | `artifacts/checkpoints/qmix/3f1553055ba04a13bf297b2d3b97e746/latest/checkpoint_state.pt` | VALIDATED |
| Hyperparameter Experiment Summary | `scratch_qmix_experiments_summary.json` | PRESENT |
| Extended Run Summary | `artifacts/final_training_summary.json` & `final_training_summary.json` | PRESENT |
| MLflow Experiment Database | `mlflow.db` (SQLite, 2.3 MB) | ACTIVE & VERIFIED |
| Week 6 AI Report | `WEEK_6_AI_REPORT.md` | THIS FILE |
| Final Status Document | `WEEK_6_FINAL_STATUS.md` | PRESENT |
| Member 3 Comparative Handoff | `HANDOFF_MEMBER3.md` | PRESENT |
| Member 4 Integration Handoff | `HANDOFF_MEMBER4.md` | PRESENT |

---
*Report: Week 6 — Member 2*  
*Project: Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning*  
*Algorithm: Custom PyTorch QMIX (Rashid et al., 2018)*
