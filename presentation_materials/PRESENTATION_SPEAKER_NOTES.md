# Presentation Speaker Notes: Project 25 Technical Presentation

**Total Duration**: 11 Minutes 30 Seconds (Target Range: 10–12 Minutes)  
**Audience**: University Examination Committee, External Evaluators, Faculty Mentors  
**Presenter Strategy**: Confident, technically precise, transparent about empirical limitations, and directly citing verified repository evidence.

---

### Slide 1: Title Slide (0:00 – 0:45 | 45 sec)
> **Slide Visual**: Isometric 3D rendering of the 20×20 autonomous warehouse layout with AMR fleet and status indicators.

**Spoken Script**:
"Good morning, respected members of the evaluation committee and faculty mentors. Today, our team is proud to present our final capstone project, **Project 25: Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning**.

Our research team consists of Anshika Gupta, Hetanshi, Soha Patel, and Tanishka Shukla. 

In this work, we have designed, implemented, trained, and deployed a cooperative multi-agent reinforcement learning system using custom PyTorch QMIX to coordinate autonomous mobile robot fleets in a dense 20 by 20 grid warehouse. Our implementation incorporates a non-bypassable safety gate protecting human-worker exclusion corridors, strict P6 checkpoint compatibility across fleet scales, an ONNX-accelerated inference service, and a live 3D digital twin deployed on Vercel. 

Over the next eleven minutes, we will walk you through our system architecture, training methodology, empirical benchmark findings, engineering challenges, and operational acceptance."

---

### Slide 2: Project Overview & Motivation (0:45 – 1:35 | 50 sec)
> **Slide Visual**: Diagram illustrating narrow aisles, multi-agent congestion, and human-worker corridors.

**Spoken Script**:
"To understand the motivation behind Project 25, consider modern automated fulfillment centers. Fleets of autonomous mobile robots—or AMRs—must transport goods from pickup racks to delivery depots along narrow two-meter aisles. 

When you operate five robots, simple pathfinding works relatively well. But when fleet density scales to ten or twenty robots in the same space, the joint state-action space expands exponentially. Independent path planners suffer from severe non-stationarity: when Robot A chooses an action, it cannot anticipate that Robot B is turning into the same cell. This leads to head-on deadlocks, mutual blocking, and throughput collapse.

At the same time, warehouses are shared human-robot spaces. Robots must strictly respect human-worker corridors with zero margin for error.

We categorize this strictly as a **Cooperative Multi-Agent Reinforcement Learning problem**, not a classical classification or regression task. Our central research question is: *Can centralized value factorization via QMIX achieve coordinated spatial deconfliction, higher throughput, and reduced safety interventions compared to independent learners and heuristic baselines?*"

---

### Slide 3: Dataset and Feature Summary (1:35 – 2:25 | 50 sec)
> **Slide Visual**: Breakdown of the 456-D per-robot observation vector and the $11 \times N$ centralized state vector.

**Spoken Script**:
"In reinforcement learning, there is no static, pre-packaged CSV dataset. Instead, our data consists of experience transitions generated online through continuous simulation rollouts in our custom Gymnasium environment.

To make decentralized coordination viable, we engineered a comprehensive **456-dimensional observation vector** for each individual robot:
- First, a 45-dimensional one-hot local occupancy grid captures the 3 by 3 neighborhood across five distinct categories: empty, shelf, corridor, empty robot, and carrying robot.
- Next, 5 dimensions encode task geometry, relative distance, and cargo flags.
- 2 dimensions provide normalized absolute spatial coordinates.
- A 400-dimensional static global map provides complete warehouse layout awareness.
- And crucially, 4 dimensions provide Breadth-First Search path-cost features, giving the network directional distance guidance.

For centralized training, our mixer receives an **11-times-N dimensional global state vector**, containing normalized poses and complete task queue attributes across all robots. The action space is discrete: four directional moves, with invalid moves resolving safely to no-ops."

---

### Slide 4: Model Selection & Technical Justification (2:25 – 3:20 | 55 sec)
> **Slide Visual**: Architecture comparison diagram: CTDE QMIX vs. Independent Q-Learning vs. Rule-Based.

**Spoken Script**:
"We evaluated three distinct paradigms to understand multi-agent coordination.

Our primary research model is **QMIX**, which operates on the principle of Centralized Training with Decentralized Execution, or CTDE. During training, individual agent networks evaluate local observations and produce individual utilities $Q_i$. A centralized mixing network then combines these utilities into a joint action-value $Q_{tot}$. To guarantee that a greedy action chosen locally by each agent corresponds to the optimal joint action, the mixer hypernetworks enforce a strict monotonicity constraint: the partial derivative of $Q_{tot}$ with respect to each $Q_i$ must be non-negative. At test time, the mixer is discarded, and agents act purely decentralized without network communication.

To benchmark this, we implemented **Independent Q-Learning**, or IQL, where each robot learns an uncoordinated Q-function, treating other moving robots as noisy environment dynamics.

Finally, we engineered a deterministic **Rule-Based baseline** that assigns tasks greedily and navigates using Breadth-First Search with dynamic cell reservations."

---

### Slide 5: Baseline Model Implementation & Results (3:20 – 4:10 | 50 sec)
> **Slide Visual**: 4-panel image sequence showing the single-robot mission: Start, Cargo Pickup, Delivery, and Next Mission.

**Spoken Script**:
"Let us examine the performance of our Rule-Based baseline across both multi-agent and single-agent settings.

In multi-agent benchmark scenarios, the rule-based controller completed 46% of deliveries at N=5, 39.5% at N=10, and 13.8% at N=20. However, because heuristic reservations lack learned spatial anticipation, robots frequently encounter dead-ends: at N=20, the rule-based controller averaged over 1,700 safety conflict interventions per episode.

To establish operational reliability in the physical simulation, we also subjected this navigation engine to a dedicated single-robot mission test across **100 consecutive seeded trials**—seeds 300001 through 300100. As shown on the slide, the robot achieved **100% mission completion** with an average delivery time of 31.52 steps, recording zero collisions, zero shelf contacts, zero corridor intrusions, and zero stuck episodes. In our live backend telemetry, this controller is transparently identified as 'Rule-Based'."

---

### Slide 6: Model Parameters & Training Hyperparameters (4:10 – 5:00 | 50 sec)
> **Slide Visual**: Detailed parameter and hyperparameter table with layer dimensions.

**Spoken Script**:
"Turning to the neural architecture, all values shown here are extracted directly from our verified training manifests.

Our individual agent network is a 3-layer Multi-Layer Perceptron: Linear 456 to 64, ReLU, Linear 64 to 64, ReLU, and Linear 64 to 4. Crucially, weights are **shared across all robots** in the fleet. This maintains parameter efficiency at exactly 33,604 parameters and ensures that agent policies scale homogeneously.

The QMIX mixing network consists of two hypernetworks conditioned on the global state $s$: Hypernetwork 1 outputs an $N$ by 32 weight matrix, and Hypernetwork 2 outputs a 32 by 1 matrix. Both pass through absolute value activations to enforce monotonicity. Because the mixer input scales with $N$, an N=5 mixer cannot be loaded into an N=10 architecture.

We trained using Adam with a learning rate of $1 \times 10^{-4}$, discount factor $\gamma = 0.99$, batch size of 32, a circular replay buffer of 5,000 transitions, and linear epsilon decay from 1.0 down to 0.05. Hard target network updates were performed every 200 episodes."

---

### Slide 7: Hyperparameter Optimization Using Optuna (5:00 – 5:45 | 45 sec)
> **Slide Visual**: Optuna study architecture, objective function formulation, and search space definition.

**Spoken Script**:
"To investigate automated tuning, we implemented an Optuna hyperparameter optimization study in `src/marl/optuna_study.py`. 

We formulated a scalarized multi-objective objective function that maximizes task completion while penalizing multi-robot conflict events:
$$\text{Score} = (\text{Completion Rate} \times 100.0) - (\text{Conflict Count} \times 2.0)$$

The search space covered learning rates between $10^{-4}$ and $10^{-2}$, discount factors of 0.95, 0.98, and 0.99, mixing hidden dimensions of 16, 32, and 64, and target update frequencies of 10, 20, and 50 episodes, backed by SQLite persistence.

In accordance with academic integrity, we must clarify that while our Optuna pipeline is fully operational and verified through unit tests, our final 1,000-episode production training runs were executed using controlled empirical sweeps rather than an automated 100-trial Optuna study due to local GPU memory and wall-clock constraints. We present verified facts, not synthetic tuning graphs."

---

### Slide 8: Experiment Tracking & Artifact Traceability (5:45 – 6:30 | 45 sec)
> **Slide Visual**: MLflow database summary table and checkpoint inventory audit.

**Spoken Script**:
"For reproducibility and MLOps rigor, every training episode logged loss values, episodic rewards, and artifact digests to our centralized SQLite MLflow database.

As detailed on the slide, our database records 13 verified experimental runs across four experiments: `project25_policy_training`, `infrastructure_smoke_tests`, `test_infra_matrix`, and default tracking. 

Furthermore, we instituted an exhaustive Checkpoint Inventory auditing all 61 model checkpoints across the repository. Under our P6 schema, every checkpoint stores an immutable SHA-256 digest, Git commit reference, architecture metadata, and exact layer shapes. The MLflow tracking interface can be inspected live on localhost port 5000."

---

### Slide 9: Training, Validation and Test Strategy (6:30 – 7:20 | 50 sec)
> **Slide Visual**: Diagram illustrating the three-tier workflow: Training (Seeds 42–44) $\rightarrow$ Validation Ranking $\rightarrow$ Frozen Holdout Benchmark.

**Spoken Script**:
"A critical requirement of Project 25 is ensuring strict separation between training, validation, and testing to prevent data leakage.

We implemented a rigorous three-tier experimental strategy:
- In Tier 1, QMIX was trained for 1,000 episodes and IQL for 500 episodes per fleet size using base seeds 42, 43, and 44, saving periodic snapshots every 100 episodes.
- In Tier 2, intermediate checkpoints were evaluated across 20 distinct validation task queues. This validation step selected Snapshot 600 for N=5, Snapshot 1000 for N=10, and Snapshot 200 for N=20.
- In Tier 3, our final reported results were computed on a completely independent, pre-frozen holdout benchmark using 20 distinct task queues under evaluation seed 816597421, governed by manifest hash `ddc457ee...`. 

No holdout scenario was ever seen during training or model selection."

---

### Slide 10: Model Evaluation Metrics (7:20 – 8:10 | 50 sec)
> **Slide Visual**: KPI definition cards and the Average Delivery Time comparison chart.

**Spoken Script**:
"Let us address why standard ML metrics such as accuracy, precision, recall, and F1-score are entirely absent from our evaluation. In multi-agent reinforcement learning, there are no static ground-truth labels. The objective is sequential utility maximization under non-stationary physical constraints.

Instead, we evaluate operational logistics KPIs:
1. **Task Completion Rate**: the percentage of total assigned pickup-and-delivery missions successfully delivered to dispatch depots.
2. **Throughput**: deliveries completed per 100 simulation steps.
3. **Average Delivery Time**: the mean environment steps required to execute completed deliveries from task spawn.
4. **Physical Collision Rate**: actual physical coordinate overlaps committed by robots.
5. **Safety Conflict Rate**: interventions triggered by our pre-action safety gate to prevent collisions.
6. **Corridor Intrusions**: unauthorized robot entries into human worker exclusion zones."

---

### Slide 11: Final Model Performance Comparison (8:10 – 9:10 | 60 sec)
> **Slide Visual**: Grouped bar charts comparing Task Completion Rate and Safety Conflicts across N=5, 10, 20.

**Spoken Script**:
"Here are the definitive results from our frozen 180-episode holdout benchmark.

At **N=5 robots**: Both QMIX and IQL achieved a 78.0% completion rate with overlapping 95% bootstrap confidence intervals of 70.0% to 86.0%. However, QMIX demonstrated superior coordination by triggering **31 times fewer safety conflict interventions**—only 1.10 conflicts per episode compared to 34.65 for IQL, and 274.7 for Rule-Based.

At **N=10 robots**: The cooperative advantage of QMIX becomes decisive. QMIX achieved **37.0% completion** versus **23.0% for IQL**—a statistically significant **14 percentage point advantage**, delivering 28 more tasks with a **97.6% reduction in safety conflicts** (2.15 vs 88.85).

At **N=20 robots**: In this dense traffic regime, both learned policies collapsed to under 1% completion. Severe aisle congestion triggered mutual blocking and policy oscillation. The rule-based controller completed 13.8% but required over 1,700 safety interventions per episode.

Across all 180 episodes, **zero physical collisions** were recorded because our contention resolution successfully converted potential collisions into wait states."

---

### Slide 12: Calibration and Reliability Analysis (9:10 – 9:55 | 45 sec)
> **Slide Visual**: Schematic distinguishing unbounded action utilities $Q(s, a)$ from calibrated class probabilities.

**Spoken Script**:
"The presentation guidelines inquire about probability calibration. We must emphasize that QMIX outputs **action-value estimates**, or Q-values, representing expected discounted cumulative returns: $Q(s, a) \in \mathbb{R}$. These are unbounded real-valued utility predictions, not classification probabilities.

Furthermore, epsilon-greedy exploration percentages are deliberate stochastic exploration parameters, not confidence measures of mission success. Therefore, calculating Brier scores, Expected Calibration Error, or reliability diagrams is mathematically inappropriate for this system.

Instead, reliability is demonstrated through **monotonicity verification** and **TD-loss convergence**: our mixing hypernetworks maintain non-negative weight projections, guaranteeing that greedy decentralized action selection maximizes joint team utility."

---

### Slide 13: Technical Challenges & Corrective Actions (9:55 – 10:50 | 55 sec)
> **Slide Visual**: Diagram illustrating the 2-cell oscillation failure mode, followed by backend contention fixes and 3D visual clearance adjustments.

**Spoken Script**:
"During development and testing, we diagnosed and resolved four critical engineering challenges:

First, we investigated the **N=20 scaling collapse**. Trajectory analysis revealed that in dense two-way traffic, robots facing each other enter a two-cell oscillation loop: alternating between two adjacent tiles every step because distance-based reward shaping cancels out when moving back and forth without forward progress. Remediation experiments confirmed that breaking this cycle requires macro-action spatial zoning.

Second, we discovered a flaw in **backend same-cell contention**: previous logic prioritized moving robots over stationary occupants. We patched `WarehouseEnvironment.step` so that stationary robots maintain spatial priority, deterministically blocking incoming contenders.

Third, in the 3D frontend, we resolved camera tracking glitches where follow mode placed the viewport inside shelf racks by implementing elevated tracking at 12 meters altitude.

Fourth, we resized AMR chassis bounds and shelf meshes, ensuring clean 50-centimeter clearances throughout the aisles."

---

### Slide 14: Deployment Architecture & Responsible AI (10:50 – 11:35 | 45 sec)
> **Slide Visual**: Cloud edge deployment architecture diagram showing Vercel hosting, FastAPI backend, and Evidently drift monitoring.

**Spoken Script**:
"For deployment, our 3D digital twin is deployed live on **Vercel** at `marl-qmix-warehouse-3d.vercel.app`, providing hardware-accelerated Three.js visualization accessible from any modern web browser.

Our backend includes a high-performance REST `/decide` inference endpoint with verified ONNX Runtime parity matching PyTorch to seven decimal places ($2.39 \times 10^{-7}$). We integrated **Evidently AI** for Population Stability Index drift monitoring on input observations and **CodeCarbon** for tracking compute energy footprints.

From a Responsible AI perspective, we explicitly designate this system as an academic research prototype not authorized for real physical warehouse operation. Our pre-action safety gate enforces zero human-worker corridor intrusions, prioritizing human safety above all reinforcement learning objectives."

---

### Slide 15: Conclusion & Q&A Defense (11:35 – 12:15 | 40 sec)
> **Slide Visual**: Summary acceptance scorecard showing NFR verdicts, and closing contact/Q&A panel.

**Spoken Script**:
"To conclude: Project 25 has delivered a complete, fully integrated cooperative MARL platform. 

We achieved 78% completion at N=5, proved a 14 percentage point cooperative advantage over IQL at N=10 with 97.6% conflict reduction, verified 100% human corridor safety across all trials, demonstrated 100% single-robot mission reliability across 100 seeded runs, and deployed a live cloud visualization on Vercel.

We have presented our findings with complete factual honesty, documenting our achievements as well as the remaining challenges in extreme-density fleet scaling. 

Thank you for your time and attention. We are now ready to take your questions."
