# Responsible AI Report
## Project 25 — Autonomous Warehouse Robot Fleet Coordination (QMIX MARL)

---

## Executive Summary

This report documents the responsible AI practices applied in the development of a cooperative multi-agent reinforcement learning (MARL) system for autonomous warehouse robot fleet coordination. The system is designed exclusively for academic research and demonstration purposes.

---

## 1. Intended Use and Scope

**Intended use:** Academic research demonstration of cooperative MARL (QMIX/IQL) in a simulated warehouse environment.

**Not intended for:**
- Deployment in real warehouses or with physical robots
- Safety-critical industrial applications
- Production fleet management systems

**Scope:**
- Software simulation only (custom grid-world + PyBullet software fallback)
- Homogeneous robot fleets of N=5, 10, 20 simulated AMRs
- 20×20 warehouse grid with annotated human-worker exclusion corridors

---

## 2. Safety Layer

### 2.1 Hard Safety Enforcement

The system implements a **mandatory, non-bypassable safety layer** that:

- Enforces human-worker corridor exclusion at the environment level
- Overrides any robot action that would enter a designated human corridor
- Operates independently of the learned QMIX/IQL policy
- Cannot be disabled via policy gradients, reward shaping, or external configuration

```
Robot action → Safety Layer → Environment step
                 ↓ (if corridor entry detected)
              NO_OP (action overridden, event logged)
```

### 2.2 Safety Validation

The evaluation pipeline includes a dedicated **safety gate** (20 episodes, 200 steps each) that must pass with **zero corridor intrusions** before benchmark results are accepted.

### 2.3 NFR-3 Requirement

> Zero corridor intrusions across all 20 safety episodes.

This is enforced at the infrastructure level, not the policy level — the safety requirement holds regardless of model quality.

---

## 3. Transparency and Accountability

### 3.1 Checkpoint Traceability (P6)

Every saved checkpoint includes full architecture metadata (Schema Version 2):
- Algorithm, fleet size, observation/action/state dimensions
- Architecture identifiers (agent network + mixer)
- Training episode, seed, timestamp
- Unique `checkpoint_id` for traceability

Checkpoints are **rejected** if metadata is missing or mismatched. There is no silent partial loading.

### 3.2 Honest Model Status Reporting

The backend explicitly reports `trained: true/false` and `model_status` in telemetry. Synthetic checkpoints (used in tests) are explicitly labeled. Randomly initialized weights are never reported as trained models.

### 3.3 Evaluation Integrity

- QMIX and IQL evaluations use **identical task seeds** (matched episodes) for fair comparison
- Bootstrap 95% confidence intervals accompany all aggregate metrics
- Paired significance tests with Holm correction prevent false discoveries
- Evaluation seeds are separated from training seeds

---

## 4. Fairness and Bias

### 4.1 Algorithm Comparison Fairness

- All three algorithms (QMIX, IQL, Rule-Based) operate in the same environment with identical reward functions
- No algorithm receives information unavailable to others at inference time
- Rule-based baseline uses deterministic nearest-task-assignment (a standard warehouse heuristic)

### 4.2 Fleet Size Fairness

- Independent training per fleet size (N=5, 10, 20) — no cross-fleet transfer
- Separate evaluation seeds per fleet size
- Metrics reported per fleet size, not aggregated across fleet sizes

---

## 5. Data and Privacy

### 5.1 Data Sources

All data is **synthetically generated** within the simulation:
- Task positions are sampled from layout-defined pickup/delivery zones
- No real warehouse operational data is used
- No personal data is collected or processed

### 5.2 Reproducibility

All training runs use fixed random seeds (42, 43, 44 for N=5, 10, 20). All evaluation runs use a fixed held-out seed (999). Full reproduction is possible from the training commands in `TRAINING_AND_REPRODUCTION_GUIDE.md`.

---

## 6. Environmental Impact

### 6.1 Compute Energy Tracking

Training energy is tracked via **CodeCarbon** (`src/monitoring/carbon_tracker.py`). Reports include:
- Estimated wall-clock training time
- CodeCarbon carbon emissions estimate (gCO₂)
- Simulated physical AMR energy model (200W motor × fleet size × training steps)

### 6.2 Training Efficiency

- CPU-only training (no GPU accelerator required)
- Short episode lengths (100 steps) reduce per-episode cost
- Checkpoint selection from periodic snapshots avoids full retraining

### 6.3 Physical AMR Energy Model

The `SustainabilityTracker` includes a physical robot energy model:
- Motor power: 200W × fleet size
- Speed: 1 m/s, grid cell: 1 m
- Idle power: 20% of motor power
- Energy per delivery = total travel energy / deliveries completed

---

## 7. Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Grid-world approximation | Real warehouse dynamics differ | Document as research prototype only |
| CPU-only training | Slower convergence, fewer trials | Long episode counts, Optuna HPO |
| PyBullet unavailable on Windows | No true 3D physics | Software physics fallback |
| Small training budget | NFR-1 (≥80% completion) may not be met | Report actual metrics, never fabricate |
| No real sensor data | Observation space may not generalize | Document out-of-scope |
| IQL not QMIX | IQL convergence is weaker | Used as baseline comparison only |

---

## 8. Incident Response

If the system is used for demonstration and an unexpected robot behavior is observed:
1. The backend sends `safetyOverrideActive: true` in the WebSocket payload
2. The frontend renders the safety zone warning overlay
3. The event is logged in `environment.get_safety_events()`
4. No physical robot is affected (simulation only)

---

## 9. Compliance Statement

This system:
- ✅ Implements a hard, non-bypassable human-corridor safety layer
- ✅ Reports model status and training provenance honestly
- ✅ Uses matched evaluation episodes for fair algorithm comparison
- ✅ Tracks compute energy consumption
- ✅ Documents all known limitations
- ✅ Is scoped exclusively to academic demonstration
- ✅ Processes no real personal or operational data
- ❌ Is **not** certified for use in physical warehouse operations
- ❌ Is **not** subject to CE/IEC 61508 or ISO 10218 safety standards (simulation only)
