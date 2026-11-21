# P6 — Checkpoint and Fleet-Size Compatibility Report

## Executive Summary

This report documents the complete implementation, verification, and architectural compatibility framework for **P6: Checkpoint and Fleet-Size Compatibility** in our Autonomous Warehouse Robot Fleet Coordination system using custom PyTorch QMIX.

The objective of P6 is to ensure that QMIX checkpoints are loaded only into strictly compatible architectures, that transitions between fleet sizes ($N=5, 10, 20$) execute transactionally and safely, and that silent partial loading, accidental use of randomly initialized mixer weights, stale controller state, and misleading model status are eliminated.

---

## 1. Root Cause Analysis

Prior to this implementation, the codebase exhibited several vulnerabilities when saving, loading, and switching fleet sizes:

1. **Dimensional Asymmetry between Agents and Mixer**:
   - `AgentQNetwork` parameters depend only on local observation dimensions (50-D legacy or 456-D map-aware) and action dimension (4). Agent weights are shared across all robots and are invariant to the fleet size $N$.
   - In contrast, `QMIXMixer` parameters scale directly with $N$ and the centralized state dimension $S = 11 \cdot N$. Hypernetwork layer `hyper_w1` produces mixing weights of dimension $32 \cdot N$ from the global state, and its second linear layer operates on $(32 \cdot N) \times (32 \cdot N)$.
2. **Silent Partial Loading**:
   - `QMIXController.load_checkpoint()` previously loaded `agent_network` first. If `mixer` was missing from the checkpoint payload, it silently skipped the mixer and flagged `checkpoint_loaded = True` (`trained = True`), misleadingly reporting that a fully trained QMIX model was active while running an untrained, randomly initialized mixer.
3. **Destructive Fleet Resizing**:
   - When changing fleet sizes via `set_fleet_size`, `warehouse_backend.py` updated `environment.set_num_robots(size)` before reconfiguring the controller.
   - `QMIXController.reset_for_fleet()` replaced `self.learner` with `_new_learner()` before loading the checkpoint. If the checkpoint was incompatible (e.g., attempting to load an $N=5$ mixer into an $N=10$ model), a `RuntimeError` was caught and converted to a `ValueError`, leaving the environment with 10 robots and the controller in a corrupted state: agent weights partially updated, mixer randomly initialized, and `checkpoint_loaded = False`.
4. **Metadata Deficiency**:
   - Checkpoints saved via `CheckpointManager` contained minimal metadata (`step`, `algorithm`, `run_id`, `observation_vector`), omitting fleet size `num_agents`, `state_dim`, architecture identifiers, and schema versions.

---

## 2. Four Key Conceptual Distinctions

As instructed, we strictly differentiate four claims that are often conflated:

1. **Code Supports a Fleet Size**:
   The simulator, observation encoder, environment adapter, QMIX networks, and controller dynamically support fleet sizes $N \in \{5, 10, 20\}$ (as well as arbitrary positive integers in unconstrained demo modes).
2. **A Checkpoint is Architecturally Compatible**:
   A checkpoint file matches the exact expected tensor names and dimensions for a given fleet size $N$ (e.g. 456-D observation, 4 actions, $11 \cdot N$ state dimension, and $(32 \cdot N)$ mixing hypernetwork weights).
3. **A Trained Checkpoint Actually Exists**:
   Whether a persistent file on disk is the artifact of actual multi-agent training.
   - **$N=5$**: A historical trained checkpoint exists at `artifacts/checkpoints/qmix/3f1553055ba04a13bf297b2d3b97e746/latest/checkpoint_state.pt`, but it was trained on the legacy 50-D observation space and is therefore architecturally incompatible with the current 456-D map-aware encoder.
   - **$N=10$**: **No genuine trained checkpoint exists.**
   - **$N=20$**: **No genuine trained checkpoint exists.**
4. **A Policy Has Demonstrated Acceptable Performance**:
   Whether a trained policy meets task completion, collision avoidance, and throughput thresholds on holdout evaluation scenarios. Untrained models or architecturally synthesized test weights have **not** demonstrated acceptable warehouse coordination.

---

## 3. Checkpoint Metadata Format (Schema Version 2)

All checkpoints saved by `CheckpointManager` and `QMIXTrainer` now use Schema Version 2.

### JSON Metadata Specification (`metadata.json` and `payload["metadata"]`)

```json
{
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
  "action_space": "Discrete(4)",
  "run_id": "3f1553055ba04a13bf297b2d3b97e746",
  "step": 60,
  "episode": 60,
  "timestamp": "2026-11-21T17:40:00Z",
  "is_best": false,
  "best_metric_val": -128.38,
  "checkpoint_sha256": "4b68f9a...",
  "config": {
    "num_agents": 5,
    "observation_dim": 456,
    "action_dim": 4,
    "state_dim": 55,
    "learning_rate": 0.0008,
    "gamma": 0.99,
    "batch_size": 64
  }
}
```

### Backward Compatibility & Inference Rules
- If `metadata` is present, `schema_version` is validated ($\in \{1, 2\}$).
- If legacy checkpoints lack `metadata` but contain a `config` dictionary, architecture fields are parsed from `config`.
- If a legacy checkpoint lacks metadata and config, architecture is inferred **only** when tensor shapes are completely unambiguous:
  - Agent network: $\text{observation\_dim} = \text{shape}[1]$ of `network.0.weight`, $\text{action\_dim} = \text{shape}[0]$ of `network.4.weight`.
  - Mixer: $\text{mixing\_hidden\_dim} = \text{shape}[0]$ of `hyper_b1.weight`, $\text{state\_dim} = \text{shape}[1]$ of `hyper_b1.weight`, $\text{num\_agents} = \text{shape}[0] / \text{mixing\_hidden\_dim}$ of `hyper_w1.0.weight`.
- If shapes do not match expected target architecture or if mixer is missing, the load is rejected with `CheckpointCompatibilityError`.

---

## 4. Fleet-Size Loading Strategy and Transitions

### Strategy
- **Shared Agent Network**: When transferring knowledge, agent network weights can only be loaded if observation dimension ($456$) and action dimension ($4$) match.
- **Dedicated Mixer per Fleet Size**: An $N=5$ mixer is mathematically and structurally incompatible with $N=10$ or $N=20$. A checkpoint loaded for a given fleet size must contain a mixer matching $N$.
- **Explicit Fleet-to-Checkpoint Mapping**: `QMIXController` supports a `fleet_checkpoints: dict[int, str | Path]` dictionary (e.g. `{5: "path/to/n5.pt", 10: "path/to/n10.pt", 20: "path/to/n20.pt"}`), environment variables `QMIX_CHECKPOINT_N5`, `QMIX_CHECKPOINT_N10`, `QMIX_CHECKPOINT_N20`, or direct path arguments.
- **No Silent Fallback**: If no checkpoint exists for the requested fleet size:
  - In `allow_untrained=False` mode, a descriptive exception is raised.
  - In `allow_untrained=True` mode, the controller explicitly sets `checkpointLoaded = False`, `trained = False`, and `modelStatus = "untrained"`.

### Transactional Fleet Reconfiguration (`reconfigure_fleet`)

The transition procedure executes atomically:
1. **Validation**: Validates that `new_fleet_size` is in `SUPPORTED_FLEET_SIZES = (5, 10, 20)`.
2. **Resolution**: Resolves checkpoint path from argument, mapping, or environment.
3. **Isolated Candidate Construction**: Constructs `candidate_encoder` and `candidate_learner` in separate memory without altering the active controller.
4. **Strict Checkpoint Validation**: Loads state dictionaries with `strict=True`, validating exact parameter names and shapes.
5. **Inference Verification**: Executes a dry-run inference step verifying that observation shape is $(N, 456)$, agent Q-values are $(N, 4)$, actions are within $[0, 3]$, and joint Q-value $Q_{\text{tot}}$ is $(1, 1)$, finite, and free of NaNs or infinities.
6. **Atomic Commit**: Replaces active learner and encoder on `self` **only after** all steps pass. If any check fails, the active controller remains completely untouched and valid.
7. **Backend Transaction Safety**: `warehouse_backend.py` calls `qmix_controller.reconfigure_fleet` **before** updating `environment.set_num_robots`. If controller reconfiguration fails, the environment remains at its previous fleet size and returns a structured command error.

---

## 5. Files Modified and Added

| File | Change Type | Description |
|---|---|---|
| `src/marl/checkpoint_validator.py` | **Added** | Checkpoint metadata schema, SHA-256 calculation, strict tensor validation, architecture compatibility verification, and inference dry-run validation. |
| `src/marl/qmix_controller.py` | **Modified** | Added `reconfigure_fleet()`, fleet-to-checkpoint mapping, atomic candidate construction, transactional rollback, strict tensor validation, and enhanced telemetry. |
| `src/marl/qmix_trainer.py` | **Modified** | Updated `_get_checkpoint_payload()` to embed Schema Version 2 metadata; updated `load_checkpoint()` to strictly validate metadata and tensors (`strict=True`). |
| `python_infra/checkpoint_manager.py` | **Modified** | Enhanced `save_checkpoint()` to build and embed Version 2 metadata (`metadata.json` and `.pt` payload). |
| `src/api/warehouse_backend.py` | **Modified** | Updated `_process_command_message()` to reconfigure the controller transactionally before mutating environment robots. |
| `tests/test_p6_checkpoint_compatibility.py` | **Added** | 23 comprehensive tests covering matching/mismatched loads, transitions, invalid/corrupt checkpoints, inference validation, and backend integration. |

---

## 6. Verification and Test Results

### 1. New P6 Compatibility Test Suite (`tests/test_p6_checkpoint_compatibility.py`)
Executed via Python 3.12:
- Matching checkpoint loads: $N=5 \to 5$, $N=10 \to 10$, $N=20 \to 20$ (**PASS**)
- Mismatched checkpoint loads: $N=5 \to 10$, $5 \to 20$, $10 \to 5$, $20 \to 10$ explicitly rejected (**PASS**)
- Fleet transitions: $5 \to 10 \to 20 \to 5$ (**PASS**)
- Repeated transition cycles: $[5, 10, 20, 10, 5, 20, 5]$ (**PASS**)
- Transition failure rollback: invalid size, corrupt file, mismatched checkpoint, and missing checkpoint all preserve the previous valid controller and its inference capabilities (**PASS**)
- Invalid checkpoints: missing file, corrupt file, 50-D observation, 6 actions, missing mixer, unexpected keys, shape mismatch, unsupported schema version (**PASS**)
- Inference validation: tensor shapes, action spaces, finite joint Q-values, no NaNs/Infs (**PASS**)
- Rejection of silent fallback to random weights (**PASS**)
- Backend WebSocket command handling integration (**PASS**)
**Result: 23 passed in 14.10s.**

### 2. Relevant Existing QMIX and Checkpoint Tests
- `tests/test_qmix_controller.py`: 4 passed
- `tests/test_qmix_network.py`: 3 passed
- `tests/test_checkpoint_manager.py`: 5 passed
- `tests/test_qmix_learner.py`: 5 passed
- `tests/test_qmix_trainer.py`: 7 passed
- `tests/test_qmix_resume.py`: 2 passed
- `tests/test_warehouse_backend.py`: 23 passed
**Result: 49 passed.**

### 3. Complete Python Test Suite
Executed all 251 tests in `tests/`:
- 250 passed.
- 1 failure in `tests/test_run_benchmark.py` due to a pre-existing Unix path assertion (`path.startswith("/")`) on Windows (`C:\...`). Unrelated to P6.

### 4. Frontend Verification
- `npm run test:frontend`: **6/6 assertions passed.**
- `npm run build`: **TypeScript compilation (`tsc`) and production Vite build succeeded without errors.**

---

## 7. Known Limitations and Honest Statement of Checkpoints

1. **Absence of Genuine Trained Policies for $N=10$ and $N=20$**:
   - The repository currently contains **no trained checkpoints** for $N=10$ or $N=20$.
   - The only pre-existing checkpoint (`artifacts/checkpoints/qmix/3f1553055ba04a13bf297b2d3b97e746/latest/checkpoint_state.pt`) was trained on the early 50-D observation space and is correctly rejected by the map-aware 456-D architecture.
   - P6 establishes the architecture, validation, and transition machinery to safely load $N=10$ and $N=20$ models once they are trained, while strictly preventing fake or mismatched models from running.
2. **Mixer Transferability**:
   - QMIX mixers cannot be directly transferred across fleet sizes without fine-tuning or training due to parameter dimension scaling. Shared agent weights can theoretically be transferred, but full QMIX evaluation requires a matching mixer.
