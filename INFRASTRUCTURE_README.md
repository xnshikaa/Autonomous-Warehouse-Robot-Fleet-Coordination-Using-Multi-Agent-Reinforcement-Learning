# MLflow Experiment Tracking & Checkpoint Management Infrastructure

This directory contains the experiment-tracking, metric-logging, and algorithm-agnostic model checkpointing infrastructure for the **Autonomous Warehouse Robot Fleet Coordination (MARL / QMIX)** project.

---

## 🏗 System Architecture & Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│            20x20 Warehouse Environment (TypeScript / Python)           │
│  - 50-D Observation Vector per robot                                   │
│  - Discrete(4) Action Space (UP=0, DOWN=1, LEFT=2, RIGHT=3)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Step Transitions & Observations
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Multi-Agent Reinforcement Learning Loop                  │
│       (Member 2 Integration: EPyMARL / QMIX / IQL / Baselines)         │
└─────────┬──────────────────────────────────────────────────────┬───────┘
          │ Parameters & Metrics                                 │ Weights & State Dicts
          ▼                                                      ▼
┌──────────────────────────────────┐            ┌──────────────────────────────────┐
│        ExperimentTracker         │            │        CheckpointManager         │
│  - MLflow Run Lifecycle          │            │  - Algorithm-agnostic storage    │
│  - Hyperparameters & Tags        │            │  - Periodic & Best checkpoints   │
│  - 8+ Warehouse Metric Hooks     │            │  - Symlinked latest/ and best/   │
└─────────────────┬────────────────┘            └─────────────────┬────────────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │ Associated MLflow Artifacts
                                          ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Local MLflow UI & Artifact Storage                       │
│  - MLruns store: ./mlruns                                             │
│  - Checkpoints store: ./artifacts/checkpoints/{algorithm}/{run_id}/   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start & Commands

### 1. Setup Python Virtual Environment

```bash
# Create local virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Infrastructure Smoke Test

```bash
# Execute the infrastructure smoke test
python -m python_infra.smoke_test
```

> **Note**: The smoke test validates MLflow run initialization, metric logging, parameter tagging, and checkpoint saving/loading. All metrics logged during the smoke test are **synthetic test data** clearly tagged as `smoke_test`.

### 3. Run Automated Pytest Suite

```bash
# Run unit tests for experiment tracker and checkpoint manager
pytest tests/ -v
```

### 4. Launch Local MLflow Tracking UI

```powershell
# SQLite backend — supports all MLflow features (traces, dataset search, UI)
.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to view logged parameters, metrics, tags, traces, and checkpoint artifacts.

---

## 📁 Artifacts & Storage Layout

Saved runs and checkpoints are organized locally under `artifacts/checkpoints/` and `mlruns/`:

```
artifacts/
  checkpoints/
    qmix/
      <run_id>/
        checkpoint_1000/
          checkpoint_state.pt
          metadata.json
        checkpoint_2000/
          checkpoint_state.pt
          metadata.json
        best/
          checkpoint_state.pt
          metadata.json
        latest/
          checkpoint_state.pt
          metadata.json
    iql/
      <run_id>/

mlruns/
  <experiment_id>/
    <run_id>/
      params/
      metrics/
      tags/
      artifacts/
```

---

## 💡 How Member 2 Will Use This Infrastructure

When Member 2 connects EPyMARL / QMIX / IQL training, they can integrate the infrastructure with minimal code:

```python
from python_infra import ExperimentConfig, TrainingMetrics, ExperimentTracker, CheckpointManager

# 1. Initialize configuration
config = ExperimentConfig(
    algorithm="QMIX",
    number_of_agents=10,  # Fleet size: 5, 10, 20
    observation_dim=50,   # 50-D observation vector
    action_space="Discrete(4)", # Discrete(4) action space
    learning_rate=0.0005,
    gamma=0.99,
    batch_size=32,
    episode_count=10000,
    random_seed=42
)

# 2. Start MLflow run
tracker = ExperimentTracker(experiment_name="warehouse_qmix_training")

with tracker:
    # Log hyperparameters and tags
    tracker.log_config(config)
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(algorithm=config.algorithm, run_id=tracker.run_id)

    # 3. Training Loop
    for episode in range(1, config.episode_count + 1):
        # ... Run episode in environment ...
        
        # Log episode metrics
        metrics = TrainingMetrics(
            episode_reward=current_reward,
            task_completion_rate=completion_rate,
            completed_tasks=tasks_done,
            collision_or_conflict_count=collisions,
            average_delivery_time=avg_time,
            throughput=throughput_val,
            idle_time=idle_val,
            episode_length=steps
        )
        tracker.log_metrics(metrics, step=episode)

        # 4. Save Checkpoints
        if episode % 1000 == 0:
            state_dict = {
                "mac": mac.state_dict(),
                "mixer": mixer.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode
            }
            checkpoint_mgr.save_checkpoint(state_dict, step=episode, tracker=tracker)
            checkpoint_mgr.save_best_checkpoint(state_dict, step=episode, current_metric_value=completion_rate, tracker=tracker)

# 5. Resuming Training from Checkpoint
loaded = checkpoint_mgr.load_checkpoint("latest")
mac.load_state_dict(loaded["checkpoint_data"]["mac"])
mixer.load_state_dict(loaded["checkpoint_data"]["mixer"])
start_episode = loaded["metadata"]["step"]
```

---

## 📊 Summary of Implemented vs Future Features

| Feature | Implemented NOW (Week 4) | Used LATER (Member 2 Integration) |
|---|---|---|
| **MLflow Tracker** | Fully implemented (`ExperimentTracker`) | Used during live EPyMARL training runs |
| **Parameters & Tags** | Fully configured (`ExperimentConfig`) | Populated from YAML training configs |
| **Metrics Pipeline** | 8+ Metric hooks (`TrainingMetrics`) | Populated from live QMIX/IQL env returns |
| **Checkpoint Storage** | Algorithm-agnostic (`CheckpointManager`) | Stores PyMARL MAC & Mixer state dicts |
| **Verification** | Passed via `smoke_test.py` & `pytest` | Validated on multi-thousand episode runs |
