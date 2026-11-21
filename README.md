# Autonomous Warehouse Robot Fleet Coordination
## Multi-Agent Reinforcement Learning (QMIX / IQL / Rule-Based)

> **Project 25** — Cooperative MARL for autonomous warehouse AMR fleets (N=5, 10, 20 robots).

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   MARL Training Pipeline                        │
│  QMIX (CTDE)  ──┐                                              │
│  IQL (independent) ─┤── QMIXTrainer / IQLTrainer               │
│  Rule-Based (greedy) ┘                                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │  checkpoint_state.pt (Schema V2)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│               FastAPI WebSocket Backend                      │
│   /ws — live simulation + QMIX decisions                     │
│   /health, /state — REST health / state                      │
│   Inference Service /decide — ONNX / PyTorch decisions       │
└──────────────────────┬───────────────────────────────────────┘
                       │  ws://localhost:8000/ws (JSON telemetry)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│          React / Three.js / TypeScript Frontend              │
│   3D 20×20 warehouse grid, AMR fleet visualization          │
│   Safety zone overlays, task assignment, algorithm selector  │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Windows PowerShell)

### Prerequisites

- Python 3.12 (`C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe`)
- Node.js 20+
- Git

### 1. Install Dependencies

```powershell
# Python dependencies
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# Frontend dependencies
npm ci
```

### 2. Train Models

```powershell
# QMIX N=5 (full training — ~1000 episodes)
python -m src.marl.train_p2_policy --algorithm QMIX --fleet-size 5 `
    --episodes 1000 --max-steps 100 --seed 42 `
    --checkpoint-interval 100 --output artifacts/training/qmix_n5_v1

# IQL N=5
python -m src.marl.train_p2_policy --algorithm IQL --fleet-size 5 `
    --episodes 500 --max-steps 100 --seed 42 `
    --checkpoint-interval 100 --output artifacts/training/iql_n5_v1

# QMIX N=10
python -m src.marl.train_p2_policy --algorithm QMIX --fleet-size 10 `
    --episodes 1000 --max-steps 100 --seed 43 `
    --checkpoint-interval 100 --output artifacts/training/qmix_n10_v1

# QMIX N=20
python -m src.marl.train_p2_policy --algorithm QMIX --fleet-size 20 `
    --episodes 1000 --max-steps 100 --seed 44 `
    --checkpoint-interval 100 --output artifacts/training/qmix_n20_v1
```

### 3. Run Evaluation Benchmark

```powershell
python -m src.evaluation.run_benchmark `
    --algorithms QMIX IQL Rule-Based `
    --fleet-sizes 5 10 20 `
    --episodes 20 --max-steps 100 --seed 999 `
    --qmix-checkpoint 5=artifacts/training/qmix_n5_v1/qmix.pt `
    --qmix-checkpoint 10=artifacts/training/qmix_n10_v1/qmix.pt `
    --qmix-checkpoint 20=artifacts/training/qmix_n20_v1/qmix.pt `
    --iql-checkpoint 5=artifacts/training/iql_n5_v1/agents `
    --iql-checkpoint 10=artifacts/training/iql_n10_v1/agents `
    --iql-checkpoint 20=artifacts/training/iql_n20_v1/agents `
    --output results/benchmark_v1
```

### 4. ONNX Export

```powershell
python -c "from src.marl.export_onnx import export_agent_to_onnx, verify_onnx_parity; export_agent_to_onnx('artifacts/training/qmix_n5_v1/qmix.pt', 'artifacts/onnx/qmix_agent_n5.onnx'); verify_onnx_parity('artifacts/training/qmix_n5_v1/qmix.pt', 'artifacts/onnx/qmix_agent_n5.onnx')"
```

### 5. Start Backend + Frontend

```powershell
# Backend (WebSocket simulation server)
python -m uvicorn src.api.warehouse_backend:create_app --factory --port 8000

# Frontend (in a separate terminal)
npm run dev
```

---

## 🧪 Testing

```powershell
# Full test suite (Windows)
python -m pytest tests/ --ignore=tests/test_optuna_study.py --ignore=tests/test_monitoring.py -q

# P6 Checkpoint compatibility tests
python -m pytest tests/test_p6_checkpoint_compatibility.py -v

# Inference service tests
python -m pytest tests/test_inference_service.py -v

# Frontend tests
npm run test:frontend -- --run

# Frontend production build
npm run build
```

---

## 📁 Project Structure

```
├── src/
│   ├── ai/                       # Core warehouse environment
│   │   ├── environment.py        # 20×20 grid warehouse, task management
│   │   ├── safety_layer.py       # Hard corridor exclusion enforcement
│   │   └── reward_function.py    # Cooperative + collision reward shaping
│   ├── marl/                     # MARL algorithms
│   │   ├── qmix_network.py       # AgentQNetwork + QMIXMixer
│   │   ├── qmix_trainer.py       # QMIX training loop (CTDE)
│   │   ├── iql_trainer.py        # Independent Q-Learning trainer
│   │   ├── qmix_controller.py    # Fleet inference controller (P6 compatible)
│   │   ├── checkpoint_validator.py  # Schema V2 validation + compatibility checks
│   │   ├── observation_encoder.py   # Map-aware 456-dim encoder
│   │   ├── export_onnx.py        # ONNX export + parity verification
│   │   └── optuna_study.py       # Hyperparameter optimization
│   ├── evaluation/               # Benchmarking + evaluation pipeline
│   │   ├── run_benchmark.py      # QMIX/IQL/Rule-Based comparison
│   │   ├── metrics.py            # Bootstrap CI, paired tests
│   │   ├── safety_gate.py        # Corridor safety evaluation
│   │   └── trace_generator.py    # Episode trace recording
│   ├── api/
│   │   ├── warehouse_backend.py  # FastAPI WebSocket simulation bridge
│   │   └── inference_service.py  # ONNX/PyTorch /decide REST endpoint
│   ├── monitoring/
│   │   ├── drift_monitor.py      # Evidently task-queue drift monitoring
│   │   └── carbon_tracker.py     # CodeCarbon + AMR energy model
│   └── simulation/
│       └── pybullet_warehouse.py # PyBullet 3D bridge (software fallback)
├── python_infra/
│   └── checkpoint_manager.py     # Schema V2 checkpoint save/load
├── tests/                        # pytest test suite (250+ tests)
├── artifacts/
│   ├── training/                 # Training run outputs
│   ├── checkpoints/              # Legacy + current checkpoints
│   └── onnx/                     # ONNX model exports
├── results/                      # Benchmark and evaluation outputs
├── .github/workflows/ci.yml      # GitHub Actions CI pipeline
├── Dockerfile                    # Inference service container
├── MODEL_CARD.md                 # Model documentation
└── requirements.txt              # Python dependencies
```

---

## 🔐 Checkpoint Compatibility (P6)

All checkpoints use **Schema Version 2** with full architecture metadata. The checkpoint validator:

- Rejects cross-fleet-size loading (N=5 checkpoint → N=10 raises `CheckpointCompatibilityError`)  
- Rejects obs_dim mismatches (legacy 50-dim vs current 456-dim)
- Never performs silent partial loading
- Requires explicit metadata for all saved checkpoints

```python
from src.marl.checkpoint_validator import validate_checkpoint_architecture
validate_checkpoint_architecture(checkpoint_path, expected_num_agents=5)
# Raises CheckpointCompatibilityError if mismatch
```

---

## 📊 Key NFRs

| NFR | Requirement | Status |
|---|---|---|
| NFR-1 | QMIX ≥80% completion at N=5 (20 episodes) | 🔄 Training |
| NFR-3 | Zero corridor intrusions (20 safety episodes) | ✅ Safety layer enforced |
| NFR-4 | QMIX collision rate ≤50% of IQL at N=10 | 🔄 Training |
| NFR-5 | Bootstrap 95% CI for all algo × fleet combinations | ✅ Evaluation pipeline ready |

---

## 🛠 Technology Stack

| Component | Technology |
|---|---|
| MARL Training | PyTorch (custom QMIX + IQL) |
| Environment | Custom grid-world (PyBullet 3D bridge with software fallback) |
| Backend API | FastAPI + WebSocket |
| Frontend | React 18 + Three.js + TypeScript |
| Model Tracking | MLflow |
| HPO | Optuna |
| ONNX Export | torch.onnx + ONNX Runtime |
| Drift Monitoring | Evidently |
| Carbon Tracking | CodeCarbon |
| CI/CD | GitHub Actions |
| Containerization | Docker |

---

## 📋 Notes

- PyBullet requires MSVC 14+ to compile from source on Windows — a **software physics fallback** is used automatically
- All training uses CPU (no GPU required)
- Synthetic test checkpoints are explicitly labeled with `synthetic: true` in metadata
- Do not use legacy 50-dim checkpoints from `artifacts/checkpoints/qmix/3f1553055ba04a13bf297b2d3b97e746/` — they will be rejected by the validator

---

## 📄 License

This project is for academic/research purposes as part of Project 25 requirements.
