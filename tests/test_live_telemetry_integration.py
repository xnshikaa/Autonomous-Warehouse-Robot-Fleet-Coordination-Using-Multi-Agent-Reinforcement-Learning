"""Integration tests for live FastAPI backend telemetry, QMIX policy execution, and fleet transitions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ai.environment import WarehouseEnvironment
from src.api.warehouse_backend import (
    _parse_command,
    _process_command_message,
    build_state_payload,
    create_app,
    CommandValidationError,
)
from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.marl.qmix_controller import QMIXController


@pytest.fixture
def trained_qmix_n5_checkpoint() -> str:
    root = Path(__file__).resolve().parents[1]
    ckpt = root / "artifacts" / "training" / "qmix_n5_v1" / "snapshots" / "episode_00600" / "checkpoint_state.pt"
    if not ckpt.is_file():
        pytest.skip(f"QMIX N=5 checkpoint not found at {ckpt}")
    return str(ckpt)


@pytest.fixture
def trained_qmix_n10_checkpoint() -> str:
    root = Path(__file__).resolve().parents[1]
    ckpt = root / "artifacts" / "training" / "qmix_n10_v1" / "qmix.pt"
    if not ckpt.is_file():
        ckpt = root / "artifacts" / "training" / "qmix_n10_v1" / "snapshots" / "episode_00500" / "checkpoint_state.pt"
    if not ckpt.is_file():
        pytest.skip(f"QMIX N=10 checkpoint not found at {ckpt}")
    return str(ckpt)


def test_command_parsing_and_validation() -> None:
    """Verify strict parsing and descriptive errors for visualizer WebSocket commands."""
    # Valid commands
    assert _parse_command('{"type": "reset"}') == {"type": "reset"}
    assert _parse_command('{"type": "pause"}') == {"type": "pause"}
    assert _parse_command('{"type": "resume"}') == {"type": "resume"}
    assert _parse_command('{"type": "set_speed", "speed": 2.5}') == {"type": "set_speed", "speed": 2.5}
    assert _parse_command('{"type": "set_fleet_size", "size": 10}') == {"type": "set_fleet_size", "size": 10}

    # Invalid JSON
    with pytest.raises(CommandValidationError) as exc:
        _parse_command("not a json")
    assert exc.value.code == "invalid_json"

    # Missing type
    with pytest.raises(CommandValidationError) as exc:
        _parse_command('{"key": "value"}')
    assert exc.value.code == "missing_type"

    # Unsupported command
    with pytest.raises(CommandValidationError) as exc:
        _parse_command('{"type": "fly_to_moon"}')
    assert exc.value.code == "unsupported_command"

    # Invalid speed
    with pytest.raises(CommandValidationError) as exc:
        _parse_command('{"type": "set_speed", "speed": -5.0}')
    assert exc.value.code == "invalid_speed"

    # Invalid fleet size
    with pytest.raises(CommandValidationError) as exc:
        _parse_command('{"type": "set_fleet_size", "size": -2}')
    assert exc.value.code == "invalid_fleet_size"

    with pytest.raises(CommandValidationError) as exc:
        _parse_command('{"type": "set_fleet_size", "size": 10, "checkpoint_path": "model.pt"}')
    assert exc.value.code == "invalid_checkpoint_source"


def test_fleet_command_keeps_trained_policy_when_target_checkpoint_is_missing(
    trained_qmix_n5_checkpoint: str, trained_qmix_n10_checkpoint: str, monkeypatch,
) -> None:
    monkeypatch.delenv("QMIX_CHECKPOINT", raising=False)
    monkeypatch.setenv("QMIX_CHECKPOINT_N5", trained_qmix_n5_checkpoint)
    monkeypatch.setenv("QMIX_CHECKPOINT_N10", trained_qmix_n10_checkpoint)
    monkeypatch.delenv("QMIX_CHECKPOINT_N20", raising=False)

    environment = WarehouseEnvironment(load_layout_tasks=True)
    environment.reset()
    app = create_app(environment=environment, policy_source="qmix")
    assert app.state.policy_source == "QMIX"
    controller = QMIXController(
        MARLEnvironmentAdapter(environment=environment), checkpoint_path=trained_qmix_n5_checkpoint
    )
    for size, expected_type, active in [(20, "command_error", 5), (10, "state_update", 10)]:
        state, _, _, _ = _process_command_message(
            json.dumps({"type": "set_fleet_size", "size": size}),
            environment, controller, lambda: controller.telemetry, 0.1, 0.1, 1.0, False,
        )
        assert state["type"] == expected_type
        assert controller.checkpoint_loaded
        assert controller.num_agents == len(environment.robots) == active


def test_live_backend_qmix_policy_execution(trained_qmix_n5_checkpoint: str) -> None:
    """Verify backend executes real QMIX policy without silent DEMO fallback."""
    env = WarehouseEnvironment(load_layout_tasks=True)
    env.reset()
    adapter = MARLEnvironmentAdapter(environment=env)
    controller = QMIXController(adapter, checkpoint_path=trained_qmix_n5_checkpoint, epsilon=0.0)

    policy_telemetry = controller.telemetry
    assert policy_telemetry["source"] == "QMIX"
    assert policy_telemetry["trained"] is True
    assert policy_telemetry["fleetSize"] == 5
    assert policy_telemetry["checkpointLoaded"] is True

    # Build telemetry payload
    payload = build_state_payload(
        env, policy_telemetry, update_interval_seconds=0.1, speed_multiplier=1.0, simulation_paused=False
    )
    assert payload["policy"]["source"] == "QMIX"
    assert payload["policy"]["trained"] is True
    assert payload["isDemoMode"] is False
    assert len(payload["robots"]) == 5

    # Verify controller step executes genuine network actions
    actions = controller.select_actions()
    assert len(actions) == 5
    assert all(a in {0, 1, 2, 3} for a in actions.values())


def test_command_processing_pause_resume_speed(trained_qmix_n5_checkpoint: str) -> None:
    """Verify pause, resume, and speed adjustment messages modify loop state."""
    env = WarehouseEnvironment(load_layout_tasks=True)
    env.reset()
    adapter = MARLEnvironmentAdapter(environment=env)
    controller = QMIXController(adapter, checkpoint_path=trained_qmix_n5_checkpoint)

    # Initial state
    delay = 0.1
    speed = 1.0
    paused = False

    # Pause
    res, delay, speed, paused = _process_command_message(
        '{"type": "pause"}', env, controller, lambda: controller.telemetry, 0.1, delay, speed, paused
    )
    assert paused is True
    assert res["simulationPaused"] is True

    # Resume
    res, delay, speed, paused = _process_command_message(
        '{"type": "resume"}', env, controller, lambda: controller.telemetry, 0.1, delay, speed, paused
    )
    assert paused is False
    assert res["simulationPaused"] is False

    # Set Speed
    res, delay, speed, paused = _process_command_message(
        '{"type": "set_speed", "speed": 4.0}', env, controller, lambda: controller.telemetry, 0.1, delay, speed, paused
    )
    assert speed == 4.0
    assert delay == pytest.approx(0.025)


def test_safe_fleet_transition_via_command(
    trained_qmix_n5_checkpoint: str, trained_qmix_n10_checkpoint: str
) -> None:
    """Verify set_fleet_size dynamically reconfigures QMIXController and warehouse environment."""
    env = WarehouseEnvironment(load_layout_tasks=True)
    env.reset()
    adapter = MARLEnvironmentAdapter(environment=env)
    controller = QMIXController(
        adapter,
        checkpoint_path=trained_qmix_n5_checkpoint,
        fleet_checkpoints={10: trained_qmix_n10_checkpoint},
    )

    assert controller.num_agents == 5
    assert len(env.robots) == 5

    # Send set_fleet_size command for size 10
    msg = json.dumps({"type": "set_fleet_size", "size": 10})
    res, delay, speed, paused = _process_command_message(
        msg, env, controller, lambda: controller.telemetry, 0.1, 0.1, 1.0, False
    )

    # Verify controller and environment updated atomically
    assert controller.num_agents == 10
    assert len(env.robots) == 10
    assert res["policy"]["fleetSize"] == 10
    assert res["policy"]["source"] == "QMIX"
    assert len(res["robots"]) == 10

    # Ensure actions work on new fleet size
    actions = controller.select_actions()
    assert len(actions) == 10
