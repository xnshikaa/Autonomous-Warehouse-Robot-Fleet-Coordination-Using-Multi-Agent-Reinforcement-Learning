# Project 25: Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning (QMIX)
## Complete Technical Project Dossier & Research Audit

**Authors**: Anshika Gupta, Hetanshi, Soha Patel, Tanishka Shukla  
**Academic Year**: 2026 · **Project Identifier**: Project 25  
**Repository Branch**: `integration/final-demo` · **Git Commit**: `8a74a73`  
**Cloud Deployment**: [https://marl-qmix-warehouse-3d.vercel.app](https://marl-qmix-warehouse-3d.vercel.app)

---

# PART 1 — PROJECT IDENTITY AND OVERVIEW

### 1. Official Project Information
* **Official Title**: Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning (QMIX)
* **Project Number**: Project 25
* **Project Category**: Cooperative Multi-Agent Reinforcement Learning (MARL), Centralized Training with Decentralized Execution (CTDE)
* **Group Number**: Team 4 / Week 6 AI Track (Group [TBD / Department Registered])
* **Faculty Mentor**: [Department Faculty Mentor — NOT EXPLICITLY RECORDED IN REPOSITORY]
* **Course & Semester**: Autonomous Systems & Reinforcement Learning, Academic Year 2026
* **Team Members & Documented Contributions**:
  * **Anshika Gupta**: Custom PyTorch QMIX neural architecture (`AgentQNetwork`, `QMIXMixer`, `QMIXLearner`), P6 checkpoint schema validation, FastAPI backend WebSocket communication, and Vercel cloud deployment.
  * **Hetanshi**: 20×20 discrete warehouse simulation environment (`WarehouseEnvironment`), layout library (`layouts.py`), dynamic task generation, and same-cell contention resolution logic.
  * **Soha Patel**: Benchmark evaluation pipeline (`metrics.py`, `plots.py`, `run_benchmark.py`), bootstrap confidence interval computation, paired statistical tests, and holdout trace generation.
  * **Tanishka Shukla**: Pre-action deterministic safety layer (`SafetyLayer`, `CorridorModule`), Three.js 3D visualizer (`WarehouseSceneController.ts`, `CameraManager.ts`), and end-to-end integration safety test suites.

### 2. Problem Statement & Motivation
In modern automated distribution and fulfillment centers, fleets of Autonomous Mobile Robots (AMRs) transport inventory between storage racks and shipping depots. As fulfillment demands rise, operators increase fleet size within fixed floor plans. However:
- **Exponential State-Action Complexity**: As the fleet size grows ($N = 5 \rightarrow 10 \rightarrow 20$), the joint state-action space scales exponentially as $|\mathcal{A}|^N = 4^N$ ($4^5 = 1,024$, $4^{10} \approx 1.05 \times 10^6$, $4^{20} \approx 1.10 \times 10^{12}$).
- **Multi-Agent Non-Stationarity**: Independent path planners (such as A* or IQL) treat peer robots as static obstacles or non-stationary noise. This causes opposing robots in narrow aisles to enter mutual deadlock or persistent reciprocal oscillation.
- **Human Worker Co-Location**: Warehouses feature designated human-worker picking zones and transit corridors. Autonomous robots must strictly respect these boundaries with **zero intrusions**.

### 3. Machine Learning Classification & Research Contribution
* **Correct ML Categorization**: This project is strictly **Cooperative Multi-Agent Reinforcement Learning (MARL)**. It is **NOT** a classification or regression task.
* **Why MARL**: AMRs operate as decentralized decision-making agents with delayed, cumulative rewards, interacting in a shared physical environment where one agent's action alters the transition probabilities for all others.
* **Core Research Contribution**:
  1. Developed a ground-up PyTorch QMIX implementation with monotonicity-enforcing hypernetworks, completely free of external MARL frameworks like EPyMARL.
  2. Implemented a non-bypassable pre-action safety gate that guarantees zero human corridor intrusions regardless of policy exploratory actions.
  3. Formulated the P6 architecture compatibility standard to eliminate silent weight mismatches across fleet sizes.
  4. Conducted a rigorous empirical comparison of QMIX, IQL, and Rule-Based baselines across $N=5, 10, 20$ on a frozen held-out benchmark with bootstrap confidence intervals and paired statistical significance testing.

---

# PART 2 — SYSTEM ARCHITECTURE AND WORKFLOW

The complete architectural specification, component inventories, and Mermaid flowcharts are detailed in [PROJECT_ARCHITECTURE.md](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/presentation_materials/PROJECT_ARCHITECTURE.md).

### Summary Workflow
1. **Simulation Environment**: 20×20 discrete grid (`WarehouseEnvironment`) with aisles, shelf racks, and human corridors.
2. **Perception**: Per-robot 456-D observation vectors extracted via `ObservationEncoder`.
3. **Decentralized Execution**: Shared-weight `AgentQNetwork` computes local utilities $Q_i(o_i, a_i)$ and selects actions greedily $\operatorname{argmax}_{a_i} Q_i$.
4. **Safety Interception**: `SafetyLayer` checks destination cells; unauthorized corridor entries are converted to `NO_OP`.
5. **Contention Resolution**: Contention logic grants priority to stationary cell occupants and blocks incoming contenders.
6. **Centralized Training**: Centralized state $s \in \mathbb{R}^{11 \times N}$ conditions `QMIXMixer` hypernetworks to estimate joint utility $Q_{tot}$. Bellman TD error updates network weights via Adam.

---

# PART 3 — DATASET AND FEATURE SUMMARY

### 1. Online Simulation Experience Generation
This project does not rely on static tabular or vision datasets. Training data consists of online environment transition tuples:
$$\tau_t = \left(s_t, \mathbf{o}_t, \mathbf{a}_t, r_t, s_{t+1}, \mathbf{o}_{t+1}, d_t\right)$$
generated by simulator rollouts in `WarehouseGymEnv`.

### 2. Feature Representation Specifications

#### A. Per-Robot Local Observation Vector (456 Dimensions)
| Feature Sub-Vector | Dimensions | Representation & Normalization | Purpose |
|---|---|---|---|
| **Local Occupancy Grid** | 45 | $3 \times 3$ neighborhood around robot, 5-category one-hot: `[empty, shelf, corridor, empty_robot, carrying_robot]` | Immediate obstacle and dynamic robot perception. |
| **Task Geometry** | 3 | Normalized relative offsets $(\Delta x / 20, \Delta y / 20)$ and Manhattan distance $d / 40$ | Directional bearing toward active pickup/delivery goal. |
| **Task & Cargo Flags** | 2 | Binary boolean flags: `has_active_task`, `carrying_cargo` | Identifies task phase (heading to pickup vs. delivery). |
| **Robot Coordinates** | 2 | Normalized coordinates $(x / 19.0, y / 19.0) \in [0.0, 1.0]$ | Absolute spatial localization on 20×20 grid. |
| **Static Layout Map** | 400 | Flattened $20 \times 20$ grid: `0.0` (walkable), `0.5` (corridor), `1.0` (shelf obstacle) | Global structural layout and wall awareness. |
| **BFS Path Costs** | 4 | Normalized shortest static distance remaining after taking actions: UP, DOWN, LEFT, RIGHT | Gradient path guidance toward target destination. |

#### B. Centralized Global State Vector ($11 \times N$ Dimensions)
Used exclusively by the QMIX mixing network hypernetworks during centralized training:
- **Per-Robot Attributes ($3 \times N$)**: Normalized coordinates $(x/20, y/20)$ and cargo status.
- **Per-Task Attributes ($8 \times N$)**: Task presence, task ID, assigned robot index, pickup $(x, y)$, delivery $(x, y)$, and completion flag.
- Dimensions: **55-D** for $N=5$, **110-D** for $N=10$, **220-D** for $N=20$.

---

# PART 4 — MODEL SELECTION AND JUSTIFICATION

### Comparative Matrix

| Evaluation Criterion | Custom PyTorch QMIX | Independent Q-Learning (IQL) | Rule-Based (BFS Navigator) |
|---|---|---|---|
| **Learning Paradigm** | Cooperative MARL with Value Factorization | Decentralized Independent Learning | Deterministic Static Heuristic |
| **Training / Execution** | Centralized Training, Decentralized Execution (CTDE) | Decentralized Training, Decentralized Execution | Zero Training; Real-Time Execution |
| **Monotonicity Guarantee** | Yes ($\partial Q_{tot} / \partial Q_i \ge 0$) | Not Applicable (No joint value) | Not Applicable |
| **Non-Stationarity Handling** | Handled via centralized state-conditioned mixer | Vulnerable to peer-agent non-stationarity | Not Handled (Reactive dynamic replanning) |
| **Runtime Inter-Robot Comm.** | **Zero communication** during inference | **Zero communication** during inference | Shared central reservation table |
| **Interpretability** | Moderate (Action Q-values per robot) | Moderate (Independent Q-values) | High (Direct shortest-path routes) |
| **Role in Project 25** | **Primary Proposed Research Model** | **Decentralized Learning Baseline** | **Heuristic Operational Baseline** |

---

# PART 5 — BASELINE IMPLEMENTATION AND RESULTS

### 1. Rule-Based Baseline Architecture
Implemented in `src/ai/live_navigation.py` (`WarehouseNavigator`):
- **Task Allocation**: Greedy nearest-available-task assignment based on Manhattan distance.
- **Path Planning**: Breadth-First Search (BFS) computing the shortest static route on the grid.
- **Dynamic Collision Avoidance**: Cell reservation table reserving current positions and one-step lookahead destinations. When routes are blocked, robots yield to adjacent vacant refuge cells.

### 2. Multi-Agent Benchmark Results
Across the frozen 20-episode benchmark:
- **$N=5$**: 46.0% completion rate [34.0–59.0%], 27.38 steps delivery time, 274.7 safety conflicts/ep.
- **$N=10$**: 39.5% completion rate [30.5–49.5%], 29.71 steps delivery time, 641.1 safety conflicts/ep.
- **$N=20$**: 13.8% completion rate [10.2–17.3%], 25.08 steps delivery time, 1726.1 safety conflicts/ep.

### 3. Single-Robot Operational Mission Validation (100 / 100 Seeds)
Tested across 100 consecutive seeded trials (seeds 300001–300100) via `tests/test_single_robot_mission_100.py`:
- **Completions**: **100 / 100 (100.0%)**
- **Physical Collisions**: 0
- **Shelf / Obstacle Contacts**: 0
- **Human Corridor Intrusions**: 0
- **Stuck / Oscillating Episodes**: 0
- **Average Delivery Time**: 31.52 steps (Min: 16, Max: 51 steps)

---

# PART 6 — QMIX MODEL PARAMETERS AND HYPERPARAMETERS

### 1. Trainable Parameter Specifications

#### A. Individual Agent Q-Network (`AgentQNetwork`)
- `fc1`: `Linear(in_features=456, out_features=64)` $\rightarrow$ $456 \times 64 + 64 = 29,248$ params
- `fc2`: `Linear(in_features=64, out_features=64)` $\rightarrow$ $64 \times 64 + 64 = 4,160$ params
- `fc3`: `Linear(in_features=64, out_features=4)` $\rightarrow$ $64 \times 4 + 4 = 260$ params
- **Total Agent Network Parameters**: **33,604 parameters** (shared across all $N$ robots).

#### B. Centralized Mixer (`QMIXMixer`)
- Hypernetwork 1: Maps global state $s \in \mathbb{R}^{11 \times N}$ to $W_1 \in \mathbb{R}^{N \times 32}$.
- Hypernetwork 2: Maps global state $s \in \mathbb{R}^{11 \times N}$ to $W_2 \in \mathbb{R}^{32 \times 1}$.
- State-dependent bias layers: `Linear(11*N, 32)` and `Linear(11*N, 1)`.

### 2. Verified Hyperparameter Values

| Hyperparameter | Verified Value | Purpose | Source File |
|---|---|---|---|
| **Learning Rate** | $1 \times 10^{-4}$ ($0.0001$) | Adam step size for backpropagation | `training_manifest.json` |
| **Discount Factor ($\gamma$)** | $0.99$ | Balances immediate vs. future rewards | `training_manifest.json` |
| **Batch Size** | $32$ transitions | Transition minibatch per gradient update | `training_manifest.json` |
| **Replay Buffer Capacity** | $5,000$ transitions | Capacity of circular multi-agent buffer | `qmix_trainer.py` |
| **Exploration Schedule** | $\epsilon = 1.0 \rightarrow 0.05$ | Linear decay over 500 episodes | `qmix_trainer.py` |
| **Target Update Interval** | $200$ episodes | Hard parameter copy interval to target networks | `training_manifest.json` |
| **Gradient Clipping** | $\text{max\_norm} = 10.0$ | Prevents exploding gradient destabilization | `qmix_learner.py` |
| **TD Loss Function** | Smooth L1 Loss (Huber) | Robust loss less sensitive to reward outliers | `qmix_learner.py` |
| **Training Horizon** | $100$ steps / episode | Maximum simulation steps before timeout | `environment.json` |

---

# PART 7 — OPTUNA HYPERPARAMETER OPTIMIZATION

### 1. Implementation
- Framework implementation: `src/marl/optuna_study.py`.
- Persistence: SQLite database storage (`qmix_study.db`).
- Scalarized multi-objective objective function:
  $$\text{Score} = (\text{Task Completion Rate} \times 100.0) - (\text{Collision/Conflict Count} \times 2.0)$$

### 2. Search Space
- Learning rate: $\left[10^{-4}, 10^{-2}\right]$ (logarithmic float).
- Discount factor $\gamma$: Categorical $\in \{0.95, 0.98, 0.99\}$.
- Mixer hidden units: Categorical $\in \{16, 32, 64\}$.
- Target network update interval: Categorical $\in \{10, 20, 50\}$.
- Collision penalty weight: $[0.5, 5.0]$.

### 3. Factual Audit
- Verification: Tested with 3-trial unit verification in `tests/test_optuna_study.py`.
- **Academic Transparency**: Full 1,000-episode production training runs were executed using controlled empirical sweeps rather than an automated 100-trial Optuna sweep due to local compute limitations. No synthetic Optuna history curves are claimed.

---

# PART 8 — EXPERIMENT TRACKING USING MLFLOW AND GIT

### 1. Centralized MLflow Tracking
Tracking URI: `sqlite:///mlflow.db` (artifacts in `mlruns/`).

| Experiment ID | Experiment Name | Logged Runs | Status | Primary Model Logged |
|---|---|---|---|---|
| `4` | `project25_policy_training` | 7 | ACTIVE | `qmix_n5_remediation_low_lr`, `qmix_n20_remediation_control`, `project25_tracking_qmix`, `project25_tracking_iql` |
| `2` | `infrastructure_smoke_tests` | 2 | ACTIVE | `smoke_test_run` |
| `3` | `test_infra_matrix` | 2 | ACTIVE | `matrix_test_run` |
| `1` | `test_qmix_trainer_experiment` | 1 | ACTIVE | Training pipeline test |

### 2. Checkpoint Provenance
- Master inventory audits 61 checkpoints: [CHECKPOINT_INVENTORY.md](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/CHECKPOINT_INVENTORY.md).
- QMIX $N=5$ Best Checkpoint: `artifacts/training/qmix_n5_v1/snapshots/episode_00600/checkpoint_state.pt`  
  (SHA-256: `89d4a600569b92e68dfa888cd0ff081539c0417b30772ff4c15b1ebcf8309fe8`).
- QMIX $N=10$ Best Checkpoint: `artifacts/training/qmix_n10_v1/snapshots/episode_01000/checkpoint_state.pt`  
  (SHA-256: `43e30259e9de5df5b33b9e16a33503631c449836429d1bfe53762551ae37192e`).

---

# PART 9 — TRAINING, VALIDATION AND TEST STRATEGY

```
   ┌────────────────────────────────────────────────────────┐
   │ 1. TRAINING PHASE                                      │
   │ Base Seeds: 42 (N=5), 43 (N=10), 44 (N=20)             │
   │ 1,000 Episodes QMIX / 500 Episodes IQL                 │
   │ Snapshots saved every 100 episodes                     │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 2. VALIDATION RANKING PHASE                            │
   │ 20 Matched Scenario Queues (Seeds 20261120-20261139)   │
   │ Horizon: 100 steps · Evaluates intermediate snapshots  │
   │ Selected: Snap 600 (N=5), Snap 1000 (N=10)             │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 3. FROZEN HOLDOUT BENCHMARK                            │
   │ Evaluation Seed: 816597421 (Manifest ddc457ee...)      │
   │ 20 Unseen Task Queues per fleet · 100-step horizon     │
   │ 180 Matched Episodes across all 9 configurations       │
   └────────────────────────────────────────────────────────┘
```

---

# PART 10 — EVALUATION METRICS AND MATHEMATICAL EXPRESSIONS

1. **Task Completion Rate ($\text{CR}$)**:
   $$\text{CR} = \frac{\text{Completed Deliveries}}{\text{Total Assigned Tasks}} \times 100\%$$
2. **Throughput ($\text{TP}$)**:
   $$\text{TP} = \frac{\text{Completed Deliveries}}{\text{Total Simulation Steps}} \times 100$$
3. **Average Delivery Time ($\bar{T}_{\text{deliv}}$)**:
   $$\bar{T}_{\text{deliv}} = \frac{1}{|C|} \sum_{j \in C} \left(t_{\text{delivery}}^{(j)} - t_{\text{spawn}}^{(j)}\right)$$
4. **Physical Collision Rate**:
   $$\text{Collision Rate} = \frac{\text{Physical Intersecting Coordinate Events}}{\text{Episodes}} \equiv 0.000$$
5. **Safety Conflict Rate**:
   $$\text{Conflict Rate} = \frac{\text{Safety Override Interventions}}{\text{Episodes}}$$
6. **Corridor Intrusions**:
   $$\text{Intrusions} = \sum \mathbb{I}\left(c_{\text{robot}} \in \mathcal{C}_{\text{human}}\right) \equiv 0$$

---

# PART 11 — FINAL MODEL PERFORMANCE COMPARISON

Authoritative results from frozen holdout manifest `e47ae5704f026c96` ([summary.json](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/results/project25_holdout/summary.json)):

| Fleet Size | Algorithm | Completed / Tasks | Completion Rate (95% CI) | Avg Delivery Steps | Physical Collisions | Safety Conflicts / Ep | Throughput / 100 Steps |
|---|---|---:|---|---:|---:|---:|---:|
| **N = 5** | **QMIX** | **78 / 100** | **78.0% [70.0–86.0%]** | **34.78** | **0.00** | **1.10 [0.65–1.55]** | **5.765** |
| N = 5 | IQL | 78 / 100 | 78.0% [70.0–86.0%] | 37.10 | 0.00 | 34.65 [9.55–65.75] | 5.138 |
| N = 5 | Rule-Based | 46 / 100 | 46.0% [34.0–59.0%] | 27.38 | 0.00 | 274.70 [207.1–337.9] | 3.729 |
| **N = 10** | **QMIX** | **74 / 200** | **37.0% [28.0–46.0%]** | **34.40** | **0.00** | **2.15 [1.60–2.85]** | **3.700** |
| N = 10 | IQL | 46 / 200 | 23.0% [17.5–28.5%] | 36.31 | 0.00 | 88.85 [46.2–135.9] | 2.300 |
| N = 10 | Rule-Based | 79 / 200 | 39.5% [30.5–49.5%] | 29.71 | 0.00 | 641.10 [551.4–723.8] | 3.950 |
| **N = 20** | **QMIX** | **3 / 400** | **0.75% [0.0–1.5%]** | **42.33** | **0.00** | **7.75 [3.10–15.05]** | **0.150** |
| N = 20 | IQL | 1 / 400 | 0.25% [0.0–0.75%] | 45.00 | 0.00 | 249.90 [209.1–294.8] | 0.050 |
| N = 20 | Rule-Based | 55 / 400 | 13.8% [10.2–17.3%] | 25.08 | 0.00 | 1726.10 [1661.7–1789.6] | 2.750 |

---

# PART 12 — CALIBRATION AND RELIABILITY

- **Theoretical Justification**: QMIX outputs action-value utilities $Q(s, a) \in \mathbb{R}$, which represent unbounded discounted cumulative rewards, **not class probabilities** $P(y|x) \in [0, 1]$.
- **Inapplicability of Brier Score**: Computing Brier scores or Reliability Diagrams on action-value utilities is mathematically invalid.
- **Monotonicity Verification**: Verified that hypernetworks output non-negative weights via absolute-value projections ($|W| \ge 0$), preserving consistent decentralized action selection.

---

# PART 13 — TRAINING CHALLENGES AND CORRECTIVE ACTIONS

| Identified Problem | Root Cause | Evidence | Corrective Action | Verification Status |
|---|---|---|---|---|
| **N=20 Policy Oscillation** | Opposing robots in narrow aisles alternate between two cells; distance rewards cancel out. | Trace logs in `n20_cycle_escape_validation.json`. | Documented as primary architectural scaling limitation; requires macro-action cycle escapes. | **IDENTIFIED & AUDITED** |
| **Same-Cell Contention Inversion** | Contention tie-breaking blocked the stationary occupant instead of incoming contenders. | Contention trace in `src/ai/environment.py`. | Patched `WarehouseEnvironment.step` so stationary occupants maintain priority. | **FIXED & VERIFIED** |
| **3D Camera In-Shelf Teleportation** | Hardcoded preset `(5, 6, 8)` placed camera inside a 2.4m shelf rack in Aisle 2. | Visual glitch in follow mode. | Implemented elevated isometric tracking at $(x+9, 12, z+9)$ and manual orbit disengagement. | **FIXED & VERIFIED** |
| **AMR & Shelf Visual Overlap** | Status rings had 0.88m radius and shelves 1.7m width, leaving only 27cm clearance. | Visual intersection in Three.js. | Reduced ring radius to 0.60m and shelf width to 1.6m, providing >50cm clearance. | **FIXED & VERIFIED** |

---

# PART 14 — MONITORING, DEPLOYMENT AND RESPONSIBLE AI

- **Cloud Deployment**: Deployed on Vercel at [https://marl-qmix-warehouse-3d.vercel.app](https://marl-qmix-warehouse-3d.vercel.app) with single-page routing.
- **REST Inference Service**: FastAPI endpoint `/decide` running ONNX Runtime; numerical parity matching PyTorch to $2.39 \times 10^{-7}$.
- **Data Drift Monitoring**: `src/monitoring/drift_monitor.py` implementing Population Stability Index (PSI) and Wasserstein distance on 456-D observation streams.
- **Carbon Tracking**: `src/monitoring/carbon_tracker.py` recording energy consumption and emissions.
- **Responsible AI Scope**: Strictly scoped to academic demonstration; pre-action safety gate enforces zero human-worker corridor intrusions.

---

# PART 15 — FINAL ACCEPTANCE VERDICT

| Requirement | Target | Empirical Result | Acceptance Status |
|---|---|---|---|
| **NFR-1** | QMIX completion $\ge 80\%$ at $N=5$ | 78.0% [70.0–86.0%] | **TARGET MISSED (PARTIAL)** |
| **NFR-2** | QMIX vs IQL scaling analysis | +14.0% advantage at $N=10$ ($p=0.694$) | **VERIFIED** |
| **NFR-3** | Zero corridor intrusions (20 episodes) | 0 intrusions (20/20 blocked) | **PASS** |
| **NFR-4** | Collision reduction at $N=10$ | 0 physical collisions; 97.6% conflict reduction | **PASS (Interventions)** |
| **NFR-5** | Bootstrap 95% confidence intervals | 10,000 resamples across all 9 configurations | **PASS** |
| **NFR-6** | Reproducibility manifests | Cryptographic SHA-256 manifest `e47ae570...` | **PASS** |
| **Single-Robot Mission** | 100% reliable pickup & delivery | 100/100 completed deliveries (seeds 300001–300100) | **PASS** |
| **Cloud Deployment** | Live web demonstration | Live on Vercel (HTTP 200 verified) | **PASS** |
