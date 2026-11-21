"""Comprehensive test suite for P6: Checkpoint and Fleet-Size Compatibility.

Tests:
1. Matching checkpoint loads (N=5, 10, 20).
2. Mismatched checkpoint loads (N=5->10, 5->20, 10->5, 20->10).
3. Fleet transitions (5->10, 10->20, 20->5, repeated cycles, failure rollback).
4. Invalid checkpoint cases (missing, corrupt, wrong obs_dim, wrong action_dim,
   missing mixer, unexpected keys, shape mismatch, unsupported version).
5. Inference validation across all fleet sizes.
6. Verify no silent fallback to random weights for incompatible loads.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from src.ai.environment import WarehouseEnvironment
from src.marl.checkpoint_validator import (
    CHECKPOINT_SCHEMA_VERSION,
    SUPPORTED_FLEET_SIZES,
    CheckpointCompatibilityError,
    build_checkpoint_metadata,
)
from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.marl.observation_encoder import ObservationEncoder
from src.marl.qmix_controller import QMIXController
from src.marl.qmix_learner import QMIXLearner


def _create_synthetic_qmix_checkpoint(
    path: Path,
    num_agents: int,
    observation_dim: int = 456,
    action_dim: int = 4,
    state_dim: int | None = None,
    run_id: str = "test_provenance_run",
    step: int = 1000,
    schema_version: int = 2,
    omit_mixer: bool = False,
    extra_keys: dict | None = None,
    corrupt_shape: bool = False,
) -> Path:
    """Helper to generate structurally valid and controlled test checkpoints."""
    effective_state_dim = state_dim if state_dim is not None else 11 * num_agents
    learner = QMIXLearner(
        num_agents=num_agents,
        observation_dim=observation_dim,
        action_dim=action_dim,
        state_dim=effective_state_dim,
    )

    metadata = build_checkpoint_metadata(
        num_agents=num_agents,
        observation_dim=observation_dim,
        action_dim=action_dim,
        state_dim=effective_state_dim,
        agent_hidden_dim=64,
        mixing_hidden_dim=32,
        run_id=run_id,
        step=step,
    )
    if schema_version != 2:
        metadata["schema_version"] = schema_version

    agent_state = learner.agent_network.state_dict()
    if extra_keys:
        agent_state.update(extra_keys)

    payload = {
        "schema_version": schema_version,
        "metadata": metadata,
        "agent_network": agent_state,
        "config": {
            "num_agents": num_agents,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "state_dim": effective_state_dim,
        },
    }

    if not omit_mixer:
        mixer_state = learner.mixer.state_dict()
        if corrupt_shape:
            # Modify a tensor shape deliberately
            mixer_state["hyper_w1.0.weight"] = torch.randn(17, effective_state_dim)
        payload["mixer"] = mixer_state

    torch.save(payload, path)
    return path


@pytest.fixture
def test_checkpoints(tmp_path):
    """Generate valid test checkpoints for fleet sizes 5, 10, and 20."""
    ckpts = {}
    for n in (5, 10, 20):
        p = tmp_path / f"qmix_n{n}.pt"
        _create_synthetic_qmix_checkpoint(p, num_agents=n, run_id=f"run_n{n}")
        ckpts[n] = p
    return ckpts


# ---------------------------------------------------------------------------
# 1. Matching Checkpoint Loads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fleet_size", [5, 10, 20])
def test_matching_checkpoint_loads_succeed(test_checkpoints, fleet_size):
    ckpt_path = test_checkpoints[fleet_size]
    env = WarehouseEnvironment(num_robots=fleet_size, load_layout_tasks=True)
    adapter = MARLEnvironmentAdapter(environment=env)
    adapter.reset()
    controller = QMIXController(adapter, checkpoint_path=ckpt_path)

    assert controller.checkpoint_loaded is True
    assert controller.checkpoint_path == str(ckpt_path)
    assert controller.checkpoint_id == f"run_n{fleet_size}"
    assert controller.telemetry["trained"] is True
    assert controller.telemetry["modelStatus"] == "trained"
    assert controller.telemetry["activeFleetSize"] == fleet_size

    actions = controller.select_actions()
    assert len(actions) == fleet_size
    assert all(a in range(4) for a in actions.values())


# ---------------------------------------------------------------------------
# 2. Mismatched Checkpoint Loads (Must be rejected explicitly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ckpt_fleet,target_fleet",
    [
        (5, 10),
        (5, 20),
        (10, 5),
        (20, 10),
    ],
)
def test_mismatched_checkpoint_loads_rejected(test_checkpoints, ckpt_fleet, target_fleet):
    ckpt_path = test_checkpoints[ckpt_fleet]
    env = WarehouseEnvironment(num_robots=target_fleet, load_layout_tasks=True)
    adapter = MARLEnvironmentAdapter(environment=env)

    with pytest.raises((ValueError, CheckpointCompatibilityError)) as exc_info:
        QMIXController(adapter, checkpoint_path=ckpt_path)

    err = str(exc_info.value)
    assert (
        "incompatible" in err.lower()
        or f"num_agents={ckpt_fleet}" in err
        or f"N={target_fleet}" in err
    )


# ---------------------------------------------------------------------------
# 3. Fleet Transitions
# ---------------------------------------------------------------------------


def test_fleet_transitions_forward_and_backward(test_checkpoints):
    env = WarehouseEnvironment(num_robots=5, load_layout_tasks=True)
    adapter = MARLEnvironmentAdapter(environment=env)
    adapter.reset()
    controller = QMIXController(
        adapter,
        checkpoint_path=test_checkpoints[5],
        fleet_checkpoints=test_checkpoints,
    )

    # 5 -> 10
    env.set_num_robots(10)
    controller.reconfigure_fleet(10)
    assert controller.learner.num_agents == 10
    assert controller.checkpoint_loaded is True
    assert controller.checkpoint_id == "run_n10"
    actions = controller.select_actions()
    assert len(actions) == 10

    # 10 -> 20
    env.set_num_robots(20)
    controller.reconfigure_fleet(20)
    assert controller.learner.num_agents == 20
    assert controller.checkpoint_loaded is True
    assert controller.checkpoint_id == "run_n20"
    actions = controller.select_actions()
    assert len(actions) == 20

    # 20 -> 5
    env.set_num_robots(5)
    controller.reconfigure_fleet(5)
    assert controller.learner.num_agents == 5
    assert controller.checkpoint_loaded is True
    assert controller.checkpoint_id == "run_n5"
    actions = controller.select_actions()
    assert len(actions) == 5


def test_repeated_fleet_transitions_cycle(test_checkpoints):
    env = WarehouseEnvironment(num_robots=5, load_layout_tasks=True)
    adapter = MARLEnvironmentAdapter(environment=env)
    adapter.reset()
    controller = QMIXController(adapter, fleet_checkpoints=test_checkpoints)

    cycle = [5, 10, 20, 10, 5, 20, 5]
    for target in cycle:
        env.set_num_robots(target)
        controller.reconfigure_fleet(target)
        assert controller.learner.num_agents == target
        assert controller.checkpoint_loaded is True
        assert controller.telemetry["activeFleetSize"] == target
        assert controller.telemetry["modelStatus"] == "trained"
        actions = controller.select_actions()
        assert len(actions) == target


def test_transition_failure_preserves_valid_controller(test_checkpoints, tmp_path):
    env = WarehouseEnvironment(num_robots=5, load_layout_tasks=True)
    adapter = MARLEnvironmentAdapter(environment=env)
    adapter.reset()
    controller = QMIXController(
        adapter,
        checkpoint_path=test_checkpoints[5],
        fleet_checkpoints=test_checkpoints,
    )

    # Record baseline state of initial valid controller
    initial_learner = controller.learner
    initial_ckpt = controller.checkpoint_path
    initial_actions = controller.select_actions()
    assert len(initial_actions) == 5

    # 1. Attempt transition to invalid fleet size 7
    with pytest.raises(ValueError, match="Unsupported fleet size"):
        controller.reconfigure_fleet(7)
    assert controller.learner is initial_learner
    assert controller.learner.num_agents == 5
    assert controller.checkpoint_path == initial_ckpt
    assert controller.checkpoint_loaded is True

    # 2. Attempt transition with corrupt checkpoint file
    corrupt_ckpt = tmp_path / "corrupt.pt"
    corrupt_ckpt.write_bytes(b"NOT_A_PYTORCH_FILE")
    with pytest.raises(ValueError, match="Corrupt or invalid"):
        controller.reconfigure_fleet(10, checkpoint_path=corrupt_ckpt)
    assert controller.learner is initial_learner
    assert controller.learner.num_agents == 5
    assert controller.checkpoint_loaded is True

    # 3. Attempt transition with mismatched checkpoint (N=5 ckpt for N=10)
    with pytest.raises((ValueError, CheckpointCompatibilityError)):
        controller.reconfigure_fleet(10, checkpoint_path=test_checkpoints[5])
    assert controller.learner is initial_learner
    assert controller.learner.num_agents == 5
    assert controller.checkpoint_loaded is True

    # 4. Attempt transition to N=10 with no checkpoint when allow_untrained=False
    empty_controller = QMIXController(
        adapter,
        checkpoint_path=test_checkpoints[5],
        fleet_checkpoints={},  # no mapping
    )
    with pytest.raises(ValueError, match="No trained checkpoint available"):
        empty_controller.reconfigure_fleet(10, allow_untrained=False)
    assert empty_controller.learner.num_agents == 5
    assert empty_controller.checkpoint_loaded is True

    # Verify controller can still execute inference seamlessly after all failures
    actions = controller.select_actions()
    assert len(actions) == 5
    assert all(a in range(4) for a in actions.values())


# ---------------------------------------------------------------------------
# 4. Invalid Checkpoint Cases
# ---------------------------------------------------------------------------


def test_missing_checkpoint_raises_file_not_found(tmp_path):
    env = WarehouseEnvironment(num_robots=5)
    adapter = MARLEnvironmentAdapter(environment=env)
    missing = tmp_path / "does_not_exist.pt"

    with pytest.raises(FileNotFoundError):
        QMIXController(adapter, checkpoint_path=missing)


def test_corrupt_checkpoint_raises_value_error(tmp_path):
    env = WarehouseEnvironment(num_robots=5)
    adapter = MARLEnvironmentAdapter(environment=env)
    corrupt = tmp_path / "bad.pt"
    corrupt.write_bytes(b"\x00\x01\x02\x03GARBAGE")

    with pytest.raises(ValueError, match="Corrupt or invalid"):
        QMIXController(adapter, checkpoint_path=corrupt)


def test_wrong_observation_dimension_rejected(tmp_path):
    ckpt = tmp_path / "wrong_obs.pt"
    _create_synthetic_qmix_checkpoint(ckpt, num_agents=5, observation_dim=50)

    env = WarehouseEnvironment(num_robots=5)
    adapter = MARLEnvironmentAdapter(environment=env)

    with pytest.raises((ValueError, CheckpointCompatibilityError), match="observation_dim=50"):
        QMIXController(adapter, checkpoint_path=ckpt)


def test_wrong_action_dimension_rejected(tmp_path):
    ckpt = tmp_path / "wrong_act.pt"
    _create_synthetic_qmix_checkpoint(ckpt, num_agents=5, action_dim=6)

    env = WarehouseEnvironment(num_robots=5)
    adapter = MARLEnvironmentAdapter(environment=env)

    with pytest.raises((ValueError, CheckpointCompatibilityError), match="action_dim=6"):
        QMIXController(adapter, checkpoint_path=ckpt)


def test_missing_mixer_rejected(tmp_path):
    ckpt = tmp_path / "no_mixer.pt"
    _create_synthetic_qmix_checkpoint(ckpt, num_agents=5, omit_mixer=True)

    env = WarehouseEnvironment(num_robots=5)
    adapter = MARLEnvironmentAdapter(environment=env)

    with pytest.raises((ValueError, CheckpointCompatibilityError), match="missing 'mixer'"):
        QMIXController(adapter, checkpoint_path=ckpt)


def test_unexpected_model_parameters_rejected(tmp_path):
    ckpt = tmp_path / "extra_keys.pt"
    _create_synthetic_qmix_checkpoint(
        ckpt,
        num_agents=5,
        extra_keys={"network.99.weight": torch.randn(10, 10)},
    )

    env = WarehouseEnvironment(num_robots=5)
    adapter = MARLEnvironmentAdapter(environment=env)

    with pytest.raises((ValueError, CheckpointCompatibilityError), match="unexpected parameter keys"):
        QMIXController(adapter, checkpoint_path=ckpt)


def test_tensor_shape_mismatch_rejected(tmp_path):
    ckpt = tmp_path / "corrupt_shape.pt"
    _create_synthetic_qmix_checkpoint(ckpt, num_agents=5, corrupt_shape=True)

    env = WarehouseEnvironment(num_robots=5)
    adapter = MARLEnvironmentAdapter(environment=env)

    with pytest.raises((ValueError, CheckpointCompatibilityError), match="shape mismatch"):
        QMIXController(adapter, checkpoint_path=ckpt)


def test_unsupported_checkpoint_version_rejected(tmp_path):
    ckpt = tmp_path / "future_ver.pt"
    _create_synthetic_qmix_checkpoint(ckpt, num_agents=5, schema_version=99)

    env = WarehouseEnvironment(num_robots=5)
    adapter = MARLEnvironmentAdapter(environment=env)

    with pytest.raises((ValueError, CheckpointCompatibilityError), match="schema_version 99"):
        QMIXController(adapter, checkpoint_path=ckpt)


# ---------------------------------------------------------------------------
# 5. Inference Validation Across All Fleet Sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fleet_size", [5, 10, 20])
def test_inference_validation_properties(test_checkpoints, fleet_size):
    ckpt_path = test_checkpoints[fleet_size]
    env = WarehouseEnvironment(num_robots=fleet_size, load_layout_tasks=True)
    adapter = MARLEnvironmentAdapter(environment=env)
    adapter.reset()
    controller = QMIXController(adapter, checkpoint_path=ckpt_path)

    # 1. Observation batch shape
    observations = controller._observations()
    assert observations.shape == (fleet_size, 456)
    assert torch.isfinite(observations).all()

    # 2. Agent Q-values shape and bounds
    with torch.no_grad():
        q_vals = controller.learner.agent_network(observations)
    assert q_vals.shape == (fleet_size, 4)
    assert torch.isfinite(q_vals).all()
    assert not torch.isnan(q_vals).any()

    # 3. Selected actions shape and bounds
    actions = controller.select_actions()
    assert len(actions) == fleet_size
    assert set(actions.keys()) == set(range(fleet_size))
    for act in actions.values():
        assert act in (0, 1, 2, 3)

    # 4. Joint Q-value Q_tot finite and valid
    best_q = q_vals.max(dim=-1).values.unsqueeze(0)  # (1, fleet_size)
    dummy_state = torch.zeros((1, 11 * fleet_size), dtype=torch.float32)
    with torch.no_grad():
        q_tot = controller.learner.mixer(best_q, dummy_state)
    assert q_tot.shape == (1, 1)
    assert torch.isfinite(q_tot).all()
    assert not torch.isnan(q_tot).any()

    # 5. Telemetry
    assert controller.telemetry["checkpointLoaded"] is True
    assert controller.telemetry["trained"] is True
    assert controller.telemetry["modelStatus"] == "trained"
    assert controller.telemetry["checkpointId"] == f"run_n{fleet_size}"


# ---------------------------------------------------------------------------
# 6. Verify Incompatible Load Never Falls Back to Random Weights
# ---------------------------------------------------------------------------


def test_incompatible_load_never_silently_falls_back(tmp_path):
    env = WarehouseEnvironment(num_robots=10, load_layout_tasks=True)
    adapter = MARLEnvironmentAdapter(environment=env)

    # Attempt to load N=5 checkpoint into N=10 controller
    ckpt_n5 = tmp_path / "n5_incompatible.pt"
    _create_synthetic_qmix_checkpoint(ckpt_n5, num_agents=5)

    with pytest.raises((ValueError, CheckpointCompatibilityError)):
        QMIXController(adapter, checkpoint_path=ckpt_n5)

    # Check that creating controller without checkpoint is explicitly untrained
    untrained_controller = QMIXController(adapter)
    assert untrained_controller.checkpoint_loaded is False
    assert untrained_controller.telemetry["trained"] is False
    assert untrained_controller.telemetry["modelStatus"] == "untrained"
    assert untrained_controller.telemetry["checkpointPath"] is None
    assert untrained_controller.telemetry["checkpointId"] is None


# ---------------------------------------------------------------------------
# 7. Backend Integration: Fleet size command handling
# ---------------------------------------------------------------------------


def test_backend_fleet_transition_integration(test_checkpoints):
    from src.api.warehouse_backend import _process_command_message

    env = WarehouseEnvironment(num_robots=5, load_layout_tasks=True)
    env.reset()
    adapter = MARLEnvironmentAdapter(environment=env)
    controller = QMIXController(
        adapter,
        checkpoint_path=test_checkpoints[5],
        fleet_checkpoints=test_checkpoints,
    )

    def policy_getter():
        return controller.telemetry

    # Successful transition via backend command 5 -> 10
    cmd_10 = '{"type":"set_fleet_size","size":10}'
    payload, _, _, _ = _process_command_message(
        cmd_10, env, controller, policy_getter, 0.1, 0.1, 1.0, False
    )
    assert payload["type"] == "state_update"
    assert payload["metrics"]["totalRobots"] == 10
    assert payload["policy"]["trained"] is True
    assert payload["policy"]["activeFleetSize"] == 10
    assert len(payload["robots"]) == 10

    # Failed transition with invalid size: controller and environment preserved
    cmd_bad = '{"type":"set_fleet_size","size":7}'
    error_payload, _, _, _ = _process_command_message(
        cmd_bad, env, controller, policy_getter, 0.1, 0.1, 1.0, False
    )
    assert error_payload["type"] == "command_error"
    assert "error" in error_payload
    # Environment and controller still at 10
    assert len(env.robots) == 10
    assert controller.learner.num_agents == 10
    assert controller.checkpoint_loaded is True
