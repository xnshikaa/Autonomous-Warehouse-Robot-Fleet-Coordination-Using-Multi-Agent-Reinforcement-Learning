# 15-Slide Presentation Content: Project 25 Technical Presentation

**Title**: Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning (QMIX)  
**Target Duration**: 10–12 Minutes (approx. 45–55 seconds per slide)  
**Tone**: Academically rigorous, technically precise, factually honest, and grounded in empirical evidence.

---

## Slide 1 — Title Slide
- **Title**: Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning
- **Subtitle**: Cooperative Multi-Agent Reinforcement Learning (QMIX) for Scalable AMR Logistics
- **Exact Bullet Points**:
  * **Project**: Project 25 — Academic Capstone & Research Demonstration
  * **Contributors**: Anshika Gupta, Hetanshi, Soha Patel, Tanishka Shukla
  * **Group / Mentor**: Group [Recorded: Team 4 / Week 6 AI Track] · Faculty Mentor: [Department Faculty Mentor]
  * **Course & Year**: Autonomous Systems & Reinforcement Learning · Academic Year 2026
  * **Core Stack**: Custom PyTorch QMIX, Gymnasium, FastAPI, Three.js / React 18, MLflow, Vercel
- **Recommended Visual**: High-resolution 3D warehouse isometric overview showing the 20×20 grid, shelf rows, and AMR fleet.
- **Evidence Reference**: `results/project25_live/screenshots/01-warehouse-overview.png`
- **Speaker Notes**: "Good morning, respected committee members and faculty mentor. Today we present Project 25: Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning. Our research tackles the complex challenge of coordinating multi-robot autonomous fleets in a congested 20 by 20 warehouse using custom PyTorch QMIX under hard human-worker safety constraints."
- **Speaking Time**: ~45 seconds

---

## Slide 2 — Project Overview & Problem Statement
- **Title**: Project Overview & Problem Motivation
- **Exact Bullet Points**:
  * **Core Problem**: Autonomous Mobile Robots (AMRs) in modern fulfillment centers must navigate narrow aisles, avoid gridlocks, and fulfill dynamic pickup-and-delivery orders.
  * **The Scaling Bottleneck**: As fleet density grows ($N=5 \rightarrow 10 \rightarrow 20$), the joint state-action space expands exponentially, causing multi-agent non-stationarity and frequent deadlocks.
  * **ML Categorization**: This is **NOT** a classification or regression problem; it is a **Cooperative Multi-Agent Reinforcement Learning (MARL)** sequential decision process.
  * **Core Research Question**: Can centralized value factorization (QMIX) achieve superior coordination, conflict reduction, and throughput over decentralized independent learners (IQL) and rule-based heuristics under strict human safety boundaries?
- **Recommended Visual**: Diagram illustrating multi-agent congestion, narrow aisles, and the human exclusion corridor constraint.
- **Evidence Reference**: `results/project25_live/screenshots/02-qmix-five-running.png`
- **Speaker Notes**: "In automated warehouse logistics, moving from 5 to 20 robots is not a linear scaling problem. Independent pathfinding causes frequent head-on blockages and reciprocal oscillation. We model this as a cooperative MARL problem where robots share common fulfillment objectives while executing decentralized motion in real time."
- **Speaking Time**: ~50 seconds

---

## Slide 3 — Dataset and Feature Summary
- **Title**: Environment Simulation & State Representation
- **Exact Bullet Points**:
  * **Experience-Generated Dataset**: No static external dataset exists; training data consists of online environment interaction rollouts sampled via a custom Gymnasium simulation.
  * **Per-Robot Observation Space (456-D Vector)**:
    - *Local Occupancy Grid*: 45-D one-hot encoding of $3 \times 3$ neighborhood across 5 spatial categories.
    - *Task Geometry & Flags*: 5-D relative distance, row/col offsets, cargo carrying status, and assignment flag.
    - *Robot Pose*: 2-D normalized $(x, y)$ warehouse coordinates.
    - *Static Global Map*: 400-D flattened binary mask of walkable tiles, shelf racks, and human corridors.
    - *BFS Path Guidance*: 4-D shortest static path distances remaining after taking each directional action.
  * **Centralized Global State Space ($11 \times N$-D Vector)**:
    - 3 features per robot (`norm_x`, `norm_y`, `cargo_flag`) + 8 features per task covering coordinates and completion status.
  * **Discrete Action Space**: 4 directional actions: UP (`0`), DOWN (`1`), LEFT (`2`), RIGHT (`3`). Unsafe actions execute as `NO_OP`.
- **Recommended Visual**: Diagram breaking down the 456-D observation vector components alongside the $20 \times 20$ grid cell map.
- **Evidence Reference**: [DATA_CARD.md](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/DATA_CARD.md#L30-L50)
- **Speaker Notes**: "Because this is reinforcement learning, our dataset consists of online environment transitions. Each robot receives a comprehensive 456-dimensional observation vector that blends local perception with global navigation cues. Crucially, we supply four BFS path-cost features that guide the agents toward their targets while the hypernetwork mixer receives an 11-times-N centralized global state vector."
- **Speaking Time**: ~50 seconds

---

## Slide 4 — Model Selection & Technical Justification
- **Title**: Algorithm Selection: QMIX vs. IQL vs. Rule-Based
- **Exact Bullet Points**:
  * **QMIX (Centralized Training, Decentralized Execution - CTDE)**:
    - *Individual Q-Networks*: Shared-parameter 3-layer MLP predicting utilities $Q_i(o_i, a_i)$ from local observations.
    - *Mixing Network*: Hypernetworks conditioned on global state $s$ produce positive weights ($|W| \ge 0$), enforcing the monotonicity constraint: $\frac{\partial Q_{tot}}{\partial Q_i} \ge 0$.
    - *Decoupled Execution*: At test time, agents select actions greedily $\operatorname{argmax}_{a_i} Q_i$ without inter-robot communication.
  * **Independent Q-Learning (IQL)**:
    - Treats other robots as dynamic parts of the environment; suffers from environment non-stationarity and lack of coordination.
  * **Rule-Based Baseline (Deterministic BFS Reservation)**:
    - Greedy nearest-task heuristic with dynamic cell-reservation pathfinding; serves as an empirical non-learning benchmark.
- **Recommended Visual**: Comparison matrix table showing learning approach, coordination mechanism, and computational requirements.
- **Evidence Reference**: [MODEL_CARD.md](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/MODEL_CARD.md#L38-L52), `presentation_materials/PROJECT_ARCHITECTURE.md`
- **Speaker Notes**: "We evaluated three distinct paradigms. QMIX guarantees that the optimal joint action corresponds to the set of individual greedy actions through its monotonic mixing constraint. IQL strips away centralized mixing to measure the penalty of non-stationarity. Our Rule-Based baseline provides a deterministic heuristic comparison."
- **Speaking Time**: ~55 seconds

---

## Slide 5 — Baseline Model Implementation & Results
- **Title**: Rule-Based Baseline Implementation & Operational Audit
- **Exact Bullet Points**:
  * **Algorithm Architecture**:
    - Task Allocation: Greedy nearest-available-task assignment based on Manhattan distance.
    - Navigation: Dynamic Breadth-First Search (BFS) pathfinder replanning every simulation tick.
    - Collision Avoidance: Reservation table reserving starting cells and planned destination cells.
  * **Multi-Agent Baseline Benchmark ($N=5, 10, 20$)**:
    - $N=5$: 46.0% completion rate (27.38 steps avg delivery time; 274.7 safety conflicts/ep).
    - $N=10$: 39.5% completion rate (29.71 steps avg delivery time; 641.1 safety conflicts/ep).
    - $N=20$: 13.8% completion rate (25.08 steps avg delivery time; 1726.1 safety conflicts/ep).
  * **Single-Robot Operational Reliability (100 / 100 Seeds Verified)**:
    - Tested across 100 consecutive seeds (300001–300100) using `test_single_robot_mission_100.py`.
    - **100% mission completion (100/100)**, 0 collisions, 0 shelf contacts, 0 corridor intrusions, 0 stuck episodes.
    - Average delivery time: 31.52 steps (Min: 16, Max: 51).
- **Recommended Visual**: 4-panel image sequence of the single-robot mission: Start, Cargo Pickup, Delivery, and Next Mission.
- **Evidence Reference**: `results/project25_mission/01-mission-ready.png` to `04-next-mission-ready.png`, `tests/test_single_robot_mission_100.py`
- **Speaker Notes**: "Our rule-based baseline serves two distinct roles. In multi-agent scenarios, it completes 46% at N=5 but experiences extreme conflict interventions—over 1,700 conflicts per episode at N=20. Separately, we used this controller to establish a 100% reliable single-robot mission benchmark across 100 seeded trials with zero collisions and zero corridor intrusions."
- **Speaking Time**: ~50 seconds

---

## Slide 6 — Model Parameters & Training Hyperparameters
- **Title**: Neural Architecture & Verified Hyperparameters
- **Exact Bullet Points**:
  * **Agent Network Architecture**:
    - Input: $456$-D $\rightarrow$ `Linear(456, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(64, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(64, 4)` $\rightarrow$ 4 action Q-values.
    - Trainable Parameters: $33,604$ parameters (shared across all $N$ agents).
  * **Mixing Network Hypernetworks**:
    - State Dimension: $11 \times N$ ($55$-D for $N=5$, $110$-D for $N=10$, $220$-D for $N=20$).
    - Hypernetwork 1: Linear mapping from State to $W_1 \in \mathbb{R}^{N \times 32}$ with absolute value activation.
    - Hypernetwork 2: Linear mapping from State to $W_2 \in \mathbb{R}^{32 \times 1}$ with absolute value activation.
  * **Verified Training Hyperparameters**:
    - Optimizer: Adam ($\text{learning rate} = 1 \times 10^{-4}$), Gradient Clipping ($\text{max norm} = 10.0$).
    - Discount Factor: $\gamma = 0.99$, Batch Size: $32$, Replay Buffer Capacity: $5,000$ transitions.
    - Exploration: $\epsilon$-greedy schedule ($\epsilon_{\text{start}} = 1.0 \rightarrow \epsilon_{\text{end}} = 0.05$, linear decay over 500 episodes).
    - Target Network Update Interval: Hard parameter copy every $200$ episodes.
- **Recommended Visual**: Architectural schematic showing tensor dimensions and layer specifications for agent network and hypernetworks.
- **Evidence Reference**: `artifacts/training/qmix_n5_v1/training_manifest.json`, `src/marl/qmix_network.py`
- **Speaker Notes**: "Here are our exact hyperparameters extracted directly from the verified training manifests. Notice that the agent network uses shared weights across all robots, maintaining parameter efficiency at 33,604 parameters. The mixing network weights, however, scale directly with fleet size N, which is why checkpoints cannot be loaded across mismatched fleet sizes."
- **Speaking Time**: ~50 seconds

---

## Slide 7 — Hyperparameter Optimization (Optuna)
- **Title**: Automated Hyperparameter Optimization via Optuna
- **Exact Bullet Points**:
  * **Framework Implementation**:
    - Automated study script implemented in `src/marl/optuna_study.py` using SQLite storage backend (`qmix_study.db`).
    - Scalarized multi-objective optimization function:
      $$\text{Score} = (\text{Task Completion Rate} \times 100.0) - (\text{Conflict Count} \times 2.0)$$
  * **Search Space Defined**:
    - Learning rate: $\left[10^{-4}, 10^{-2}\right]$ (log scale); Discount factor $\gamma \in \{0.95, 0.98, 0.99\}$.
    - Mixing hidden dimension $\in \{16, 32, 64\}$; Target update interval $\in \{10, 20, 50\}$ episodes.
    - Collision penalty weight: $[0.5, 5.0]$.
  * **Factual Status & Scientific Transparency**:
    - Framework functionality verified via test suite (`tests/test_optuna_study.py` executing 3 test trials).
    - **Academic Transparency**: Long-horizon 1,000-episode runs were executed using controlled empirical parameter configurations rather than a completed 100-trial automated Optuna study due to local GPU compute limits.
- **Recommended Visual**: Code snippet and optimization workflow diagram of the scalarized objective function.
- **Evidence Reference**: `src/marl/optuna_study.py`, `tests/test_optuna_study.py`
- **Speaker Notes**: "In accordance with strict scientific honesty, we report that while our Optuna optimization framework was fully implemented and unit-tested with SQLite persistence, our final 1,000-episode production training runs relied on controlled empirical tuning rather than an automated 100-trial sweep due to compute limitations. We do not fabricate synthetic Optuna history curves."
- **Speaking Time**: ~45 seconds

---

## Slide 8 — Experiment Tracking & Artifact Traceability
- **Title**: MLOps: MLflow Tracking & Cryptographic Traceability
- **Exact Bullet Points**:
  * **MLflow Tracking Database**:
    - Centralized SQLite tracking database (`mlflow.db`) and artifact store (`mlruns/`).
    - 13 verified experimental runs logged across 4 active experiments:
      1. `project25_policy_training` (Run IDs: `3c34dbe5...`, `85b173b1...`, `78a90ed5...`, `f5d0dd30...`)
      2. `infrastructure_smoke_tests` (Run ID: `0f83bb06...`)
      3. `test_infra_matrix` (Run ID: `9bf13b04...`)
  * **Cryptographic Checkpoint Integrity**:
    - Every saved model includes Schema V2 metadata, git commit hash, and unique SHA-256 digest.
    - Checkpoint Inventory audits all 61 checkpoints across the repository ([CHECKPOINT_INVENTORY.md](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/CHECKPOINT_INVENTORY.md)).
  * **Reproducibility Command**:
    - To launch the local dashboard: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`
- **Recommended Visual**: Screenshot or diagram of the MLflow experiment tracking dashboard showing run parameters and loss curves.
- **Evidence Reference**: `mlflow.db`, `CHECKPOINT_INVENTORY.md`
- **Speaker Notes**: "To ensure reproducibility, every training episode logged metrics, loss curves, and artifact digests to MLflow. Our repository maintains a complete inventory of 61 checkpoints, each stamped with its exact SHA-256 hash. The MLflow tracking server can be inspected locally at port 5000."
- **Speaking Time**: ~45 seconds

---

## Slide 9 — Training, Validation & Testing Strategy
- **Title**: Rigorous Three-Tier Evaluation Methodology
- **Exact Bullet Points**:
  * **Tier 1 — Online Simulation Training**:
    - QMIX: 1,000 episodes per fleet size ($N=5, 10, 20$) with base seeds $42, 43, 44$.
    - IQL: 500 episodes per fleet size ($N=5, 10, 20$).
    - Snapshots saved periodically every 100 episodes.
  * **Tier 2 — Model Validation & Selection (No Test Peeking)**:
    - Snapshots evaluated across 20 validation task queues (100 steps max horizon).
    - Selected best checkpoints:
      * $N=5$: Snapshot 600 (70.0% validation completion, 34.36 steps avg delivery).
      * $N=10$: Snapshot 1000 (23.0% validation completion, 31.61 steps avg delivery).
      * $N=20$: Snapshot 200 (selected early before policy oscillation).
  * **Tier 3 — Frozen Held-Out Benchmark**:
    - 20 completely independent, pre-frozen task queues per fleet size (Evaluation seed: 816597421).
    - Frozen scenario manifest hash: `ddc457ee542f3f6076058722fb8a130c022828367ac23fd5c011ea1553f6c7bf`.
- **Recommended Visual**: Three-stage workflow diagram: Training (Seeds 42–44) $\rightarrow$ Validation Selection (20 Queues) $\rightarrow$ Frozen Holdout Benchmark (Seed 816597421).
- **Evidence Reference**: [FINAL_EVALUATION_REPORT.md](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/FINAL_EVALUATION_REPORT.md), [DATA_CARD.md](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/DATA_CARD.md#L50-L65)
- **Speaker Notes**: "We instituted a strict three-tier evaluation workflow to eliminate data leakage. Models were trained on base seeds 42 to 44; intermediate snapshots were ranked using a separate validation set; and final reported metrics were computed on an untouched, frozen held-out benchmark with 20 distinct task queues per fleet size."
- **Speaking Time**: ~50 seconds

---

## Slide 10 — Evaluation Metrics & Mathematical Definitions
- **Title**: RL Evaluation Metrics vs. Classical ML Metrics
- **Exact Bullet Points**:
  * **Why Classification/Regression Metrics Do Not Apply**:
    - Accuracy, precision, recall, and $R^2$ assume static ground-truth labels.
    - MARL involves sequential decision making, non-stationary environment transitions, and cumulative reward maximization.
  * **Core Domain Metrics & Formulations**:
    1. **Task Completion Rate**: $\text{CR} = \frac{\sum \text{Completed Tasks}}{\sum \text{Total Assigned Tasks}} \times 100\%$
    2. **Throughput**: $\text{TP} = \frac{\text{Completed Deliveries}}{\text{Total Steps Elapsed}} \times 100$
    3. **Average Delivery Time**: $\bar{T}_{\text{deliv}} = \frac{\sum_{j=1}^{C} (\text{Step}_{\text{delivery}} - \text{Step}_{\text{start}})}{C}$
    4. **Physical Collision Rate**: Contacts per episode (committed overlapping robot coordinates).
    5. **Safety Conflict Rate**: $\frac{\text{Safety Override Events}}{\text{Episodes}}$ (interventions before contact).
    6. **Corridor Intrusions**: Direct intrusions into annotated human-worker exclusion zones.
- **Recommended Visual**: Formula table paired with `presentation_materials/figures/delivery_time_comparison.png`.
- **Evidence Reference**: `src/evaluation/metrics.py`, [WEEK5_EVALUATION_INFO.md](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/WEEK5_EVALUATION_INFO.md#L25-L45)
- **Speaker Notes**: "Because this is an autonomous logistics system, standard ML accuracy or MSE metrics are meaningless. We measure operational KPIs: task completion percentage, delivery time in environment steps, system throughput per 100 steps, physical collisions, and safety override events."
- **Speaking Time**: ~50 seconds

---

## Slide 11 — Model Performance Comparison
- **Title**: Frozen Benchmark Results across Algorithms & Fleets
- **Exact Bullet Points**:
  * **Full $3 \times 3$ Performance Matrix (20 Held-Out Episodes, 100 Steps Horizon)**:

| Fleet | Algorithm | Completion Rate (95% CI) | Avg Delivery Time | Safety Conflicts / Ep | Throughput / 100 Steps |
|---|---|---|---|---|---|
| **N = 5** | **QMIX** | **78.0% [70.0–86.0%]** | 34.78 steps | **1.10** | **5.76** |
| N = 5 | IQL | 78.0% [70.0–86.0%] | 37.10 steps | 34.65 | 5.14 |
| N = 5 | Rule-Based | 46.0% [34.0–59.0%] | 27.38 steps | 274.70 | 3.73 |
| **N = 10** | **QMIX** | **37.0% [28.0–46.0%]** | 34.40 steps | **2.15** | **3.70** |
| N = 10 | IQL | 23.0% [17.5–28.5%] | 36.31 steps | 88.85 | 2.30 |
| N = 10 | Rule-Based | 39.5% [30.5–49.5%] | 29.71 steps | 641.10 | 3.95 |
| **N = 20** | **QMIX** | 0.8% [0.0–1.5%] | 42.33 steps | **7.75** | 0.15 |
| N = 20 | IQL | 0.3% [0.0–0.75%] | 45.00 steps | 249.90 | 0.05 |
| N = 20 | Rule-Based | 13.8% [10.2–17.3%] | 25.08 steps | 1726.10 | 2.75 |

  * **Key Findings**:
    - **N=5**: QMIX and IQL tie on completion (78.0%), but QMIX triggers **31x fewer safety conflicts** (1.10 vs 34.65).
    - **N=10**: QMIX significantly outperforms IQL by **+14 percentage points** (37.0% vs 23.0%) with **97.6% fewer conflicts** (2.15 vs 88.85).
    - **N=20**: Both learned policies collapse due to mutual 2-cell oscillation; Rule-Based achieves 13.8% but incurs massive conflict interventions (1726.1/ep).
- **Recommended Visual**: Grouped bar chart: `presentation_materials/figures/task_completion_comparison.png`.
- **Evidence Reference**: `results/project25_holdout/report.md`, `presentation_materials/EXPERIMENT_RESULTS_MASTER_TABLE.csv`
- **Speaker Notes**: "This is our master results slide. At N=5, QMIX achieves 78% completion—just shy of the 80% target—while producing 31 times fewer conflicts than IQL. At N=10, QMIX demonstrates clear coordination superiority, outperforming IQL by 14 percentage points with a 97.6% reduction in safety interventions. At N=20, severe grid congestion causes both learned models to collapse."
- **Speaking Time**: ~60 seconds

---

## Slide 12 — Calibration & Probability Reliability Analysis
- **Title**: Calibration & Uncertainty in QMIX
- **Exact Bullet Points**:
  * **Why Probability Calibration Is Excluded**:
    - QMIX outputs state-action utilities (Q-values) representing expected discounted returns $Q(s, a) \in \mathbb{R}$, **not class posterior probabilities** $P(y|x) \in [0, 1]$.
    - Exploration probabilities via $\epsilon$-greedy are arbitrary stochastic exploration controls, not confidence estimates of task success.
  * **Non-Applicability of Standard Reliability Metrics**:
    - Expected Calibration Error (ECE), Brier Score, and Reliability Diagrams are mathematically inapplicable to action-value Q-networks.
  * **Action-Value Stability Analysis**:
    - Monotonicity constraint $\frac{\partial Q_{tot}}{\partial Q_i} \ge 0$ maintains ranking consistency: individual action maximizes joint utility.
    - TD-error convergence monitored: Initial TD loss of $310.75$ stabilized to $194.77$ in stable runs without gradient explosion.
- **Recommended Visual**: Plot/diagram explaining why Q-values are unbounded utilities rather than probabilities, contrasting softmax classifiers with Q-value distributions.
- **Evidence Reference**: `src/marl/qmix_network.py`, [FINAL_EVALUATION_REPORT.md](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/FINAL_EVALUATION_REPORT.md)
- **Speaker Notes**: "The presentation rubric asks about probability calibration. We must clarify that QMIX outputs action-value utilities, not classification probabilities. Calculating a Brier score or plotting a calibration curve would be theoretically invalid. Instead, we assess reliability through TD-loss convergence and monotonic value ranking."
- **Speaking Time**: ~45 seconds

---

## Slide 13 — Technical Challenges & Corrective Actions
- **Title**: Engineering Challenges, Root Causes & Fixes
- **Exact Bullet Points**:
  * **Challenge 1: The N=20 Policy Oscillation Collapse**:
    - *Root Cause*: In dense 20-robot traffic, robots facing head-on conflict alternate between two adjacent cells without moving forward; distance rewards cancel out.
    - *Investigation*: Traced in remediation experiments (`n20_cycle_escape_validation.json`); confirmed greedy policies get trapped in non-progress cycles.
  * **Challenge 2: Same-Cell Destination Contention Bug**:
    - *Root Cause*: Prior contention resolution blocked the stationary occupant instead of incoming contenders.
    - *Fix*: Implemented occupant priority in `WarehouseEnvironment.step`: stationary robots hold cells; incoming contenders are blocked.
  * **Challenge 3: 3D Camera Jumping & Shelf Zoom Glitches**:
    - *Root Cause*: Follow camera used hardcoded coordinates `(5, 6, 8)` directly inside a 2.4m shelf rack.
    - *Fix*: In `CameraManager.ts`, implemented dynamic elevated tracking at $(x+9, 12, z+9)$ and automatic orbit disengagement.
  * **Challenge 4: Visual AMR & Shelf Mesh Overlap**:
    - *Fix*: Reduced status ring radius to $0.60$m and shelf width to $1.6$m, creating $>50$cm physical clearance in 2m aisles.
- **Recommended Visual**: Side-by-side diagram showing the 2-cell oscillation failure mode vs. the corrected collision resolution.
- **Evidence Reference**: `src/ai/environment.py#L338`, `src/camera/CameraManager.ts#L26`, `results/project25_remediation/`
- **Speaker Notes**: "We encountered and resolved several complex engineering bugs. At N=20, robots experienced 2-cell oscillation, where policies rapidly alternate between two legal moves without making delivery progress. In the 3D visualizer, we fixed camera follow bugs that previously placed the viewpoint inside shelf racks, and corrected backend cell contention logic so stationary robots retain spatial priority."
- **Speaking Time**: ~55 seconds

---

## Slide 14 — MLOps, Cloud Deployment & Responsible AI
- **Title**: Deployment Architecture & Responsible AI
- **Exact Bullet Points**:
  * **Production Cloud Deployment**:
    - 3D Digital Twin and Control Room deployed live on **Vercel**: [marl-qmix-warehouse-3d.vercel.app](https://marl-qmix-warehouse-3d.vercel.app).
    - Single-Page Application (SPA) routing with Three.js hardware-accelerated WebGL rendering.
  * **Inference & MLOps Pipelines**:
    - High-performance REST inference endpoint (`/decide`) with ONNX Runtime parity verified to $2.39 \times 10^{-7}$.
    - Evidently AI drift monitoring: Population Stability Index (PSI) tracking input observation distribution shifts.
    - CodeCarbon tracking: verified energy consumption and compute footprint.
  * **Responsible AI & Human-Centric Safety**:
    - Academic research prototype: explicitly **NOT** authorized for physical unconstrained warehouse deployment.
    - Non-bypassable deterministic safety layer: prioritizes human worker safety by enforcing hard exclusion boundaries.
- **Recommended Visual**: Architecture diagram of the Vercel edge deployment connected to the FastAPI backend and ONNX inference service.
- **Evidence Reference**: [RESPONSIBLE_AI_REPORT.md](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/RESPONSIBLE_AI_REPORT.md), [vercel.json](file:///c:/Users/Admin/Documents/MARL/Autonomous-Warehouse-Robot-Fleet-Coordination-Using-Multi-Agent-Reinforcement-Learning/vercel.json)
- **Speaker Notes**: "Our full deployment pipeline includes a live cloud digital twin on Vercel, an ONNX-optimized inference service with numerical parity verified to seven decimal places, and Evidently drift monitoring. In our Responsible AI audit, we emphasize that this is a simulated academic prototype and enforce non-bypassable pre-action safety for human worker corridors."
- **Speaking Time**: ~50 seconds

---

## Slide 15 — Conclusion, Acceptance Verdict & Viva Defense
- **Title**: Conclusions, Acceptance Status & Q&A
- **Exact Bullet Points**:
  * **Authoritative Acceptance Summary**:
    - **NFR-1 (80% Completion at N=5)**: Achieved **78.0% [70.0–86.0%]** (Target missed by 2 deliveries; honestly reported).
    - **NFR-2 (Scalability Gap Analysis)**: Fully verified; QMIX shows **+14% advantage** over IQL at $N=10$.
    - **NFR-3 (Zero Corridor Intrusions)**: **PASSED (100% blocked, 0 intrusions)** across 20 aggressive safety episodes.
    - **NFR-4 (Collision Rate Reduction)**: **PASSED (Interventions)** with **97.6% conflict reduction** for QMIX vs IQL.
    - **NFR-5 (Bootstrap 95% CIs)**: **PASSED** across all 9 configurations with 10,000 resamples.
    - **NFR-6 (Reproducibility & Manifests)**: **PASSED** with complete cryptographic SHA-256 traceability.
  * **Core Takeaway**: QMIX provides exceptional multi-agent coordination and conflict minimization at small to medium fleet densities ($N=5, 10$), while extreme scaling ($N=20$) requires macro-action cycle-breaking and spatial zoning.
  * **Thank You — We welcome your questions.**
- **Recommended Visual**: Final summary scorecard table showing all requirements with green/amber status indicators.
- **Evidence Reference**: `presentation_materials/PROJECT_REQUIREMENT_ACCEPTANCE_MATRIX.md`
- **Speaker Notes**: "To conclude: Project 25 successfully demonstrated cooperative multi-agent reinforcement learning with custom PyTorch QMIX, achieving 78% completion at N=5, a 14-point advantage over IQL at N=10, 97.6% conflict reduction, zero human corridor intrusions, and a live 3D digital twin on Vercel. We welcome your questions."
- **Speaking Time**: ~45 seconds
