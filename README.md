# Autonomous Warehouse Robot Fleet Coordination Visualizer (MARL / QMIX)

A premium, research-grade 3D visualization layer built for university project demonstrations on **Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning (QMIX)**.

---

## 🌟 Visual Identity & Design Principles

- **Industrial Robotics Lab Aesthetic**: Clean, dark polished concrete floor, instanced modular shelving racks, painted yellow hazard borders, defined storage aisles, conveyor delivery bays, inductive charging docks, and translucent human safety zones.
- **Decoupled Architecture**: Strict state separation (`WarehouseState`). Three.js renders state snapshots received from either a local **DemoMode** simulator or a live **Python WebSocket API** (`ws://localhost:8000/ws`).
- **Research Transparency**: High-performance multi-robot fleet rendering (N=5, 10, 20), animated 3D path line ribbons, live engineering control dashboard, interactive robot inspector drawers, and presentation mode.

---

## 🏗 System Architecture

```
┌────────────────────────────────────────────────────────┐
│             Python RL Environment                      │
│      (Gym / PyMARL / QMIX / IQL / Greedy+A*)           │
└───────────────────────────┬────────────────────────────┘
                            │ WebSocket JSON Telemetry
                            ▼
┌────────────────────────────────────────────────────────┐
│            PythonWebSocketProvider                     │
│  (Subscribes to ws://localhost:8000/ws, with automatic │
│   fallback to DemoStateProvider when offline)          │
└───────────────────────────┬────────────────────────────┘
                            │ State Snapshot (WarehouseState)
                            ▼
┌────────────────────────────────────────────────────────┐
│            Three.js Visualization Layer                │
│  ├── WarehouseFloor (20x20 Grid, Aisles & Markings)   │
│  ├── ShelfSystem (Instanced Racks & Storage Crates)   │
│  ├── ZoneSystem (Dispatch, Charging, Human Zone)      │
│  ├── RobotSystem (AMR Fleet R01-R20 & LiDAR Towers)    │
│  ├── PathSystem (Glowing Directional Route Lines)     │
│  ├── TaskSystem (Pickup & Delivery Beacons)           │
│  └── SafetySystem (Human Zone Overrides & Collisions) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│        Industrial Engineering Control HUD (React)      │
│  ├── Live Dashboard (Algorithm, Fleet, Timestep, Ep)   │
│  ├── Robot & Task Inspector Panels                     │
│  ├── Multi-Camera Viewports (Overview, Top-Down, Free) │
│  └── Presentation Mode & Replay Speed Controls         │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 How to Run

Use Python 3.11 and Node.js 22 LTS. From the repository root, install the
backend and frontend dependencies:

```bash
# Python backend and test dependencies
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Frontend dependencies (package-lock.json pins the npm dependency tree)
npm ci
```

Start each service in a separate terminal from the repository root. Activate
`.venv` in the backend terminal:

```bash
source .venv/bin/activate
python -m src.api.warehouse_backend
```

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). To run the available
baseline checks:

```bash
source .venv/bin/activate
python -m pytest -q
npm run lint
npm run build
```

`requirements.txt` specifies minimum Python package versions rather than an
exact lock, so a Python dependency installation is not byte-for-byte pinned.
The repository does not define a frontend unit-test script; `npm run lint` and
`npm run build` are its available frontend checks.

---

## 🎮 Demo & Keyboard Controls

- **Camera Viewports**:
  - `OVERVIEW`: Angled high-bay perspective of the complete 20x20 warehouse layout.
  - `TOP-DOWN`: Tactical 2D overhead camera view for multi-agent coordination analysis.
  - `FREE ORBIT`: Left-click drag to rotate, right-click drag to pan, scroll to zoom.
- **Path Ribbon Visibility**:
  - Click `PATHS: ALL / SELECTED / OFF` in top right toolbar to cycle route overlay modes.
- **Robot Inspection**:
  - Left-click any AMR robot (`R01`–`R20`) in the 3D scene to open its inspector panel and click `TRACK ROBOT CAMERA`.
- **Presentation Mode**:
  - Click `PRESENTATION MODE` button in the top right header to toggle distraction-free widescreen view.
- **Simulation Replay Controls**:
  - Click `PLAY / PAUSE`, `RESET`, or `0.5x / 1x / 2x / 5x` speed multiplier in the bottom control bar.

---

## 📡 Python WebSocket Backend API Specification

The visualizer connects to `ws://localhost:8000/ws` and expects JSON payloads conforming to the `WarehouseState` schema:

```json
{
  "type": "state_update",
  "timestep": 1432,
  "episode": 812,
  "algorithm": "QMIX",
  "robots": [
    {
      "id": "R01",
      "position": [5, 7],
      "targetPosition": [12, 14],
      "rotation": 1.57,
      "status": "MOVING",
      "taskId": "T04",
      "hasCargo": true,
      "battery": 92,
      "path": [[5, 7], [6, 7], [7, 7], [8, 7], [9, 7]],
      "safetyAlert": false
    }
  ],
  "tasks": [
    {
      "id": "T04",
      "pickupPos": [5, 7],
      "pickupShelfId": "S14",
      "deliveryPos": [3, 18],
      "deliveryZoneId": "D01",
      "assignedRobotId": "R01",
      "status": "IN_PROGRESS"
    }
  ],
  "safetyZones": [],
  "recentCollisions": [],
  "metrics": {
    "activeRobots": 10,
    "totalRobots": 10,
    "activeTasks": 7,
    "completedTasks": 142,
    "throughputPerHour": 180,
    "averageDeliveryTimeSec": 12.4,
    "totalCollisions": 0,
    "safetyOverrides": 2,
    "timestep": 1432,
    "episode": 812
  },
  "safetyOverrideActive": false
}
```

---

## 🔗 Python RL (QMIX) Connection

The Week 4 Python backend is now connected to the simulator. Its live action
path is:

```text
QMIXController -> MARLEnvironmentAdapter -> WarehouseEnvironment.step()
                                                    |
                                      movement + safety + rewards
```

The adapter is the small integration boundary that translates the warehouse
environment into the multi-agent interface expected by QMIX or an external
EPyMARL training loop. It keeps the frontend, policy, and environment on the
same episode state. The backend uses the QMIX learner for action selection by
default; `policy_source="demo"` remains available as a deterministic fallback.

Start the backend from the repository root with:

```bash
source .venv/bin/activate
python -m src.api.warehouse_backend
```

The UI connects to `ws://localhost:8000/ws`. The `/state` response includes
`policy.source`, `policy.checkpointLoaded`, and `policy.trained`. The repository
does not currently contain a trained QMIX checkpoint, so the default live
connection is a real QMIX inference path with initial (untrained) weights. A
trained checkpoint can be supplied with:

```bash
export QMIX_CHECKPOINT="/path/to/qmix_checkpoint.pt"
python -m src.api.warehouse_backend
```

The checkpoint format is a PyTorch dictionary containing `agent_network` and,
optionally, `mixer` state dictionaries. `QMIXTrainer` remains the in-repo
training example; EPyMARL is not vendored here.

---

## Reproducible policy evaluation (P2)

Before training a new policy, generate and commit a fresh evaluation manifest.
Do not train or tune on its task queues. It materializes 20 scenarios per fleet
size and hashes the layout/configuration and scenario-generation code.

```bash
.venv/bin/python -m src.evaluation.scenario_manifest \
  --fleet-sizes 5 10 20 --episodes 20 --max-steps 100 --seed 976304181 \
  --output results/evaluation/p2_holdout_2026_11/scenario_manifest.json
```

After committing the manifest and training, evaluate the same frozen scenarios
with QMIX, IQL, and the rule-based baseline at N=5, 10, and 20. The command writes compressed
per-step traces, checkpoint/config/source hashes, aggregate metrics,
episode-bootstrap 95% confidence intervals, paired sign-flip comparisons,
Holm-adjusted p-values, and the specification's scalability plots. The
checkpoint paths must match the local training outputs.

```bash
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python -m src.evaluation.run_benchmark \
  --fleet-sizes 5 10 20 \
  --algorithms QMIX IQL Rule-Based \
  --qmix-checkpoint 5=artifacts/checkpoints/qmix/p1_pathcost_n5_refine500/checkpoint_3600 \
  --qmix-checkpoint 10=artifacts/checkpoints/qmix/p2_qmix_n10_seed300010/latest \
  --qmix-checkpoint 20=artifacts/checkpoints/qmix/p2_qmix_n20_seed300020/latest \
  --iql-checkpoint 5=artifacts/checkpoints/iql/p2_iql_n5_seed400005 \
  --iql-checkpoint 10=artifacts/checkpoints/iql/p2_iql_n10_seed400010 \
  --iql-checkpoint 20=artifacts/checkpoints/iql/p2_iql_n20_seed400020 \
  --episodes 20 --max-steps 100 \
  --scenario-manifest results/evaluation/p2_holdout_2026_11/scenario_manifest.json \
  --output results/evaluation/p2_benchmark
```

`manifest.json`, `summary.json`, `report.md`, and `scalability.png` are the report files;
`safety_gate.json` records the separate 20-episode aggressive corridor-entry
gate. `traces/` contains gzip-compressed raw episodes and can be loaded by
`src.evaluation.run_evaluation`. Collision counts represent actual collision
events. Safety-blocked conflicts and overrides are reported separately. The
benchmark is an evaluation harness, not evidence that a policy converged; use
the completion metrics and traces to assess the selected checkpoints.

## ⚡ Technical Performance Optimizations

- **`InstancedMesh`**: Used for all storage rack tiers and crates, reducing draw calls from hundreds down to 2 draw calls.
- **Pixel Ratio Capping**: WebGLRenderer pixel ratio capped at `Math.min(window.devicePixelRatio, 2)` for high-DPI displays.
- **GSAP Tweens**: Smooth position & angular rotation interpolation prevents frame stuttering during grid transitions.

---

## 🎯 Current Status

The simulator, Python environment, QMIX inference bridge, safety checks, and
WebSocket telemetry are implemented. The remaining model-dependent step is to
train a policy or connect an externally trained EPyMARL checkpoint; until then
the UI correctly reports `QMIX: UNTRAINED` when the default controller is used.


