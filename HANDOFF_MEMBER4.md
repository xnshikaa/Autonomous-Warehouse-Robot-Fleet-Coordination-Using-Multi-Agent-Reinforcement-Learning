# Member 4 Handoff — QMIX Integration Guide
## Week 6 AI Track — Member 2 to Member 4

This document provides everything Member 4 needs to integrate the trained QMIX
checkpoint into the broader system for testing and simulation.

---

## Checkpoint

**Run D Best Stable QMIX Checkpoint:**
`artifacts/checkpoints/qmix/3f1553055ba04a13bf297b2d3b97e746/latest/checkpoint_state.pt`

This checkpoint is validated for Member 4 integration/testing:
  - All 4 network state dicts loaded: CONFIRMED
  - Optimizer state loaded: CONFIRMED
  - Forward pass produces finite Q-values: CONFIRMED
  - Greedy action selection works: CONFIRMED

DO NOT use the 150-episode extended run checkpoint (run_id: `44fb3ff6cf7a4eb0965b0b92bb6dbd67`).
That run diverged. The Run D checkpoint above is the validated research/simulation QMIX checkpoint.

---

## Model Input / Output

| Property | Value |
|----------|-------|
| Number of agents | 5 |
| Observation vector per agent | 50-D float32 numpy array |
| Input tensor shape | (5, 50) — shape (num_agents, observation_dim) |
| Output | Action index per agent: shape (5,) with values in {0, 1, 2, 3} |

**Action mapping:**
  0 = UP    (row - 1)
  1 = DOWN  (row + 1)
  2 = LEFT  (col - 1)
  3 = RIGHT (col + 1)

---

## Global State

The global state is the concatenation of all agent observations:
  `global_state = observations.reshape(-1)`  # shape (250,)

This is used only during training (mixer input). For pure inference you only
need the per-agent observations.

---

## How to Load the Checkpoint and Run Inference

### Option A — Use QMIXTrainer (simplest)

```python
from src.marl.qmix_trainer import QMIXTrainer
import torch

trainer = QMIXTrainer(
    num_agents=5,
    observation_dim=50,
    action_dim=4,
    state_dim=250,
)

checkpoint_path = (
    "artifacts/checkpoints/qmix/"
    "3f1553055ba04a13bf297b2d3b97e746/latest"
)
trainer.load_checkpoint(checkpoint_path)

# Get observations from environment (shape: (5, 50))
obs = torch.tensor(observation_array, dtype=torch.float32)

# Greedy inference (epsilon=0.0 = fully greedy, no random actions)
actions = trainer.learner.select_actions(obs, epsilon=0.0)
# actions: tensor([a0, a1, a2, a3, a4]) with values in {0, 1, 2, 3}
```

### Option B — Use QMIXController (inference bridge)

```python
from src.marl.qmix_controller import QMIXController
from src.marl.environment_adapter import MARLEnvironmentAdapter

adapter = MARLEnvironmentAdapter(environment)  # your WarehouseEnvironment
controller = QMIXController(
    adapter=adapter,
    checkpoint_path=(
        "artifacts/checkpoints/qmix/"
        "3f1553055ba04a13bf297b2d3b97e746/latest/"
        "checkpoint_state.pt"
    ),
    epsilon=0.0,  # greedy
)

# Returns dict {robot_id: action_int}
action_map = controller.select_actions()
```

### Option C — Load weights manually

```python
import torch
from src.marl.qmix_learner import QMIXLearner

learner = QMIXLearner(num_agents=5, observation_dim=50, action_dim=4, state_dim=250)

ckpt = torch.load(
    "artifacts/checkpoints/qmix/3f1553055ba04a13bf297b2d3b97e746/latest/checkpoint_state.pt",
    map_location="cpu",
    weights_only=False,
)
payload = ckpt["checkpoint_data"]
learner.agent_network.load_state_dict(payload["agent_network"])
learner.mixer.load_state_dict(payload["mixer"])
learner.agent_network.eval()

obs = torch.tensor(obs_array, dtype=torch.float32)  # shape (5, 50)
with torch.no_grad():
    q_values = learner.agent_network(obs)  # shape (5, 4)
    actions = q_values.argmax(dim=-1)      # shape (5,)
```

---

## Environment Interface

The trained QMIX uses WarehouseGymEnv:
```python
from src.marl.gymnasium_env import WarehouseGymEnv
env = WarehouseGymEnv(num_agents=5, max_steps=40)
obs, info = env.reset()          # obs shape: (5, 50)
obs, reward, done, trunc, info = env.step(actions)  # actions: list[int] len 5
```

Observations are produced by ObservationEncoder (`src/marl/observation_encoder.py`):
  - 9 spatial features (3x3 occupancy grid around robot)
  - 41 task/agent/environment features
  - Total: 50-D float32

---

## Frontend Integration Status

The Three.js frontend (`src/`) is NOT currently live-controlled by the trained QMIX model.

The `QMIXController` class (`src/marl/qmix_controller.py`) exists as an inference bridge
and CAN be connected to the frontend via the Python backend. However, this connection
requires explicit integration work — the frontend currently runs its own rule-based
or random movement simulation.

To connect QMIX to the frontend:
  1. Load `QMIXController` with the Run D checkpoint path
  2. Hook `controller.select_actions()` into the Python backend step loop
  3. Pass resulting actions to the frontend via the existing WebSocket/API

The model is a validated research/simulation QMIX checkpoint and should not be described as a production autonomous warehouse controller.

---

## PyTorch Version Note

The checkpoint was saved with NumPy arrays in the payload (replay buffer).
Always load with `weights_only=False`:
  `torch.load(..., weights_only=False)`

The codebase has been patched for this in:
  - `src/marl/qmix_controller.py` (line 72)
  - `python_infra/checkpoint_manager.py` (line 64)

---

## Known Limitations

1. Validated at 40 steps/episode (60 episodes). For longer episodes (e.g. 200 steps), learning rate and reward scaling must be retuned.
2. 5 agents only. Scaling to different fleet sizes requires rebuilding the learner with matching `num_agents` and `state_dim = num_agents * 50`.
3. `Discrete(4)` actions only. Continuous action spaces are not supported.

---

*Member 2 — Week 6 AI Track*  
*Algorithm: Custom PyTorch QMIX*
