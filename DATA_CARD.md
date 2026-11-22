# Data Card — Autonomous Warehouse Robot Fleet Coordination

## Dataset Description

| Property | Description |
|---|---|
| Dataset Name | Warehouse Grid Simulation Task Scenarios & Traces |
| Domain | Autonomous Mobile Robot (AMR) Logistics & Fleet Management |
| Generator | Custom Gymnasium Warehouse Simulation (`src/ai/environment.py`) |
| Layout Type | 20×20 Discrete Grid World with Obstacles, Aisles, and Human Corridors |
| Fleet Configurations | Homogeneous fleets of N = 5, 10, 20 robots |
| Action Space | Discrete(4): UP, DOWN, LEFT, RIGHT |
| Observation Dimension | 456 features (map-aware local occupancy + BFS path costs + robot/task state) |
| State Dimension | 11 × N features (centralized state for QMIX mixing network) |
| Split Scheme | Training base seeds 42–44, validation seeds 20261120–20261139, frozen evaluation seeds 816597421–816597440 |

---

## Data Generation Pipeline

The warehouse dataset is generated online during simulation rollouts. Tasks are created by sampling pickup and delivery pairs:

1. Pickup and delivery candidates come from opposite halves of the walkable floor.
2. The generator excludes worker corridors and robot starts. It checks path reachability, travel distances and unique task endpoints.
3. Each robot starts with one assigned pickup and delivery task in the benchmark.
4. The action mask excludes static obstacles and worker corridors. The simulator safety layer also resolves simultaneous move conflicts.

---

## Feature Specifications

### Per-Robot Local Observation Vector (456-D)

| Feature Group | Dimensions | Description | Range |
|---|---|---|---|
| Local occupancy | 45 | 3×3 neighborhood with five one-hot categories | [0.0, 1.0] |
| Task geometry | 3 | Relative row and column offsets and distance | [-1.0, 1.0] |
| Task and cargo flags | 2 | Task present and carrying cargo | [0.0, 1.0] |
| Robot position | 2 | Normalized absolute coordinates | [0.0, 1.0] |
| Static map | 400 | 20×20 floor, shelf and corridor cells | [0.0, 1.0] |
| BFS path costs | 4 | Remaining path distance after each movement action | [0.0, 1.0] |

### Centralized Global State Vector (11 × N)

Used exclusively during centralized training by the QMIX hypernetwork mixer:
- 3 features per robot: (norm_x, norm_y, carrying_cargo)
- 8 features per task covering presence, task ID, assignee, pickup and delivery coordinates, and completion.

---

## Data Splits and Seeding

To prevent data leakage and ensure rigorous generalization measurement:

- Training covers 1,000 episodes for QMIX and 500 for IQL per fleet, with base seeds 42, 43 and 44 for N=5, 10 and 20.
- Validation ranks snapshots saved every 100 episodes using 20 matched task queues per fleet.
- Final evaluation uses 20 different task queues per fleet, frozen after these runs trained and before the evaluation. The earlier seed-999 benchmark is validation evidence.
- The separate corridor gate tests 20 deliberate attempts to enter human zones.

---

## Data Drift and Monitoring

- **Tool**: Evidently AI (`src/monitoring/drift_monitor.py`) with Population Stability Index (PSI) fallback.
- **Monitored Features**:
  - `queue_length`: Number of outstanding delivery orders.
  - `avg_delivery_distance`: Mean Manhattan distance from pickup to dropoff.
  - `max_delivery_distance`: Peak route length in active queue.
  - `assigned_ratio`: Proportion of orders currently assigned to robots.
  - `robot_density`: Fleet count per walkable floor area.
- **Drift Triggers**: A high-demand surge (3× order volume) triggers retrain recommendations when PSI > 0.25 across ≥2 features.

---

## Ethical & Privacy Considerations

- **No Personal Data**: The dataset contains zero Personally Identifiable Information (PII) or real-world operational logs.
- **Synthetic Simulation**: All movements, tasks, and routes are generated programmatically within the simulation environment.
- **Worker Safety Modeling**: Human presence is simulated via designated spatial exclusion zones, prioritizing safety by design.
