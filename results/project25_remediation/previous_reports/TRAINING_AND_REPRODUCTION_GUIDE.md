# Training and Reproduction Guide

## Environment

| Component | Version / Value |
|---|---|
| Python | 3.12 |
| PyTorch | ≥2.0.0 (CPU-only) |
| OS | Windows 11 / Ubuntu (CI) |
| PyBullet | Not available on Windows without MSVC 14+ — software physics fallback active |

---

## Reproducing Training Runs

The commands reproduce the run settings with fixed seeds and fresh output directories. Use the same runtime and CPU setup when comparing results. The CLI records episode metrics, duration, run configuration and artifact manifests in MLflow.

### QMIX Training

```powershell
# N=5 (seed 42)
python -m src.marl.train_p2_policy --algorithm QMIX --fleet-size 5 `
    --episodes 1000 --max-steps 100 --seed 42 `
    --checkpoint-interval 100 --output artifacts/training/repro_qmix_n5

# N=10 (seed 43)
python -m src.marl.train_p2_policy --algorithm QMIX --fleet-size 10 `
    --episodes 1000 --max-steps 100 --seed 43 `
    --checkpoint-interval 100 --output artifacts/training/repro_qmix_n10

# N=20 (seed 44)
python -m src.marl.train_p2_policy --algorithm QMIX --fleet-size 20 `
    --episodes 1000 --max-steps 100 --seed 44 `
    --checkpoint-interval 100 --output artifacts/training/repro_qmix_n20
```

### IQL Training

```powershell
# N=5 (seed 42)
python -m src.marl.train_p2_policy --algorithm IQL --fleet-size 5 `
    --episodes 500 --max-steps 100 --seed 42 `
    --checkpoint-interval 100 --output artifacts/training/repro_iql_n5

# N=10 (seed 43)
python -m src.marl.train_p2_policy --algorithm IQL --fleet-size 10 `
    --episodes 500 --max-steps 100 --seed 43 `
    --checkpoint-interval 100 --output artifacts/training/repro_iql_n10

# N=20 (seed 44)
python -m src.marl.train_p2_policy --algorithm IQL --fleet-size 20 `
    --episodes 500 --max-steps 100 --seed 44 `
    --checkpoint-interval 100 --output artifacts/training/repro_iql_n20
```

---

## Checkpoint Format (Schema V2)

Every QMIX checkpoint produced by this codebase includes:

```json
{
  "schema_version": 2,
  "metadata": {
    "schema_version": 2,
    "algorithm": "QMIX",
    "num_agents": 5,
    "observation_dim": 456,
    "action_dim": 4,
    "state_dim": 55,
    "agent_network_arch": "AgentQNetwork",
    "mixer_arch": "QMIXMixer",
    "agent_hidden_dim": 64,
    "mixing_hidden_dim": 32,
    "episode": 1000,
    "timestamp": "...",
    "config": { ... }
  },
  "agent_network": { ... },
  "mixer": { ... },
  ...
}
```

The P6 checkpoint validator enforces that `num_agents`, `observation_dim`, `action_dim`, `state_dim`, and architecture identifiers all match before loading.

---

## Observation Encoder

The map-aware observation encoder produces a **456-dimensional** observation vector per robot:

| Component | Dims | Description |
|---|---|---|
| Local occupancy grid | 45 | 3×3 neighborhood with five occupancy categories |
| Task geometry | 3 | Normalized row and column offsets and distance |
| Task and cargo flags | 2 | Task assigned and carrying item |
| Robot position | 2 | Normalized absolute row and column |
| Layout features | 400 | Static 20×20 grid |
| BFS path costs | 4 | Remaining path distance after each movement action |

**Total: 456 dimensions**

Legacy checkpoints with 50-dim observations are **incompatible** and will be rejected with `CheckpointCompatibilityError`.

---

## Hyperparameter Search (Optuna)

```powershell
python -m src.marl.optuna_study --fleet-size 5 --trials 20 --episodes 10 --steps 50
```

Best parameters are logged to `optuna_study.db` (SQLite). To use best params in full training:

```python
from src.marl.optuna_study import run_optuna_study
result = run_optuna_study(fleet_size=5, n_trials=20, episodes_per_trial=10, max_steps=50)
print(result["best_params"])
```

---

## Evaluation Pipeline

### Verify recovered checkpoints and rerun validation

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m python_infra.generate_checkpoint_inventory
foreach ($n in 5,10,20) {
    foreach ($algo in 'QMIX','IQL') {
        $name = $algo.ToLower() + "_n${n}"
        python -m src.evaluation.select_checkpoint --algorithm $algo --fleet-size $n `
            --candidate "artifacts/training/${name}_v1/snapshots" `
            --episodes 20 --max-steps 100 --seed 20261120 `
            --output "results/reproduced_validation/$name.json"
        if ($LASTEXITCODE -ne 0) { throw "Validation failed for $name" }
    }
}
```

### Replay the frozen evaluation

The supplied scenario manifest was generated after training and before this evaluation.
Its seed and task queues were excluded from training and checkpoint ranking. Use the
same file to preserve the exact evaluation tasks and its manifest ID.

```powershell
$flags = @()
foreach ($n in 5,10,20) {
    foreach ($algo in 'qmix','iql') {
        $selection = Get-Content -Raw "results/project25_validation/${algo}_n${n}.json" | ConvertFrom-Json
        foreach ($entry in $selection.results[0].checkpoint_sha256.PSObject.Properties) {
            if ((Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Name).Hash -ne $entry.Value) {
                throw "Checkpoint hash mismatch: $($entry.Name)"
            }
        }
        $flags += "--${algo}-checkpoint", "$n=$($selection.selected_checkpoint)"
    }
}
python -m src.evaluation.run_benchmark @flags `
    --scenario-manifest results/project25_holdout/scenario_manifest.json `
    --output results/reproduced_holdout
```

### Safety gate (rule-based, 20 episodes)

```python
from src.evaluation.safety_gate import run_safety_gate
result = run_safety_gate(episodes=20, fleet_size=5, max_steps=200)
# result["passed"] must be True (zero corridor intrusions)
```

---

## MLflow Experiment Tracking

MLflow is configured at `mlflow.db` / `mlruns/`.

```powershell
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

## ONNX Export

```python
from src.marl.export_onnx import export_agent_to_onnx, verify_onnx_parity

export_agent_to_onnx(
    "artifacts/training/qmix_n5_v1/snapshots/episode_00600/checkpoint_state.pt",
    "artifacts/onnx/reproduced_qmix_agent_n5.onnx"
)
verify_onnx_parity(
    "artifacts/training/qmix_n5_v1/snapshots/episode_00600/checkpoint_state.pt",
    "artifacts/onnx/reproduced_qmix_agent_n5.onnx"
)
```

---

## Drift Monitoring

```python
from src.monitoring.drift_monitor import run_drift_analysis
summary = run_drift_analysis(
    num_robots=5,
    reference_episodes=10,
    current_episodes=10,
    high_demand_multiplier=3.0,
    output_dir="results/monitoring",
)
print(f"Retraining recommended: {summary['retraining_recommended']}")
```

---

## Carbon Tracking

```python
from src.monitoring.carbon_tracker import SustainabilityTracker
tracker = SustainabilityTracker(algorithm="QMIX", fleet_size=5, output_dir="results/carbon")
tracker.start_tracking()
# ... training or evaluation ...
report = tracker.stop_tracking(total_steps=50000, completed_deliveries=400, idle_steps=5000)
```

---

## Known Limitations

1. **No GPU** — CPU-only training on this machine
2. **PyBullet** — unavailable without MSVC 14+; software physics fallback used throughout
3. **Training time** — 1000 episodes on CPU takes ~30-60 minutes per fleet size
4. **Legacy checkpoints** — `artifacts/checkpoints/qmix/3f1553055ba04a13bf297b2d3b97e746/` uses 50-dim obs and is permanently incompatible with the current 456-dim architecture
