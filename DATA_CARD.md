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
| Split Scheme | Training (seeds 42, 43, 44) vs Evaluation (seed 999, held-out fixed scenarios) |

---

## Data Generation Pipeline

The warehouse dataset is generated online during simulation rollouts. Tasks are created by sampling pickup and delivery pairs:

1. **Pickup Positions**: Sampled uniformly from configured storage rack aisles (`layout.walkable_cells` adjacent to shelf obstacles).
2. **Delivery Positions**: Sampled from dedicated dispatch / conveyor output bays at warehouse boundaries.
3. **Queue Discipline**: FIFO task queue with dynamic nearest-available assignment.
4. **Safety Masking**: Human-worker exclusion corridors are strictly off-limits. Any action targeting a corridor cell is masked or intercepted by the hard safety layer.

---

## Feature Specifications

### Per-Robot Local Observation Vector (456-D)

| Feature Group | Dimensions | Description | Range |
|---|---|---|---|
| Local Occupancy Grid | 45 | 5×9 local egocentric patch (empty, obstacle, robot, corridor) | [-1.0, 1.0] |
| BFS Path Cost Estimates | 9 | Shortest path distance to assigned pickup/delivery per direction | [0.0, 1.0] |
| Robot Internal State | 6 | Normalized (x, y), cargo status (0/1), battery, current task ID | [0.0, 1.0] |
| Assigned Task Features | 8 | Pickup (x, y), Delivery (x, y), task status, elapsed steps | [0.0, 1.0] |
| Team Relative Features | 20 | Relative offsets to nearest 5 teammate robots | [-1.0, 1.0] |
| Global Map Representation | 368 | Downsampled layout and static obstacle feature map | [0.0, 1.0] |

### Centralized Global State Vector (11 × N)

Used exclusively during centralized training by the QMIX hypernetwork mixer:
- 3 features per robot: (norm_x, norm_y, carrying_cargo)
- 8 features per active task: pickup (x,y), delivery (x,y), assigned_flag, time_in_queue

---

## Data Splits and Seeding

To prevent data leakage and ensure rigorous generalization measurement:

- **Training Sets**:
  - N=5: Seed 42 (1000 episodes, max 100 steps/episode)
  - N=10: Seed 43 (1000 episodes, max 100 steps/episode)
  - N=20: Seed 44 (1000 episodes, max 100 steps/episode)
- **Validation Snapshots**: Periodic evaluation snapshots every 100 episodes on validation seed 20261120.
- **Sealed Evaluation Scenarios**: 20 held-out evaluation episodes per fleet size using seed 999.
- **Corridor Safety Gate**: 20 aggressive adversarial test scenarios forcing robots toward human zones.

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
