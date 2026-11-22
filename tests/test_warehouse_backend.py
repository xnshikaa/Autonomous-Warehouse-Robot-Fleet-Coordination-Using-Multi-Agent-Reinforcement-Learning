import json

import pytest
from unittest.mock import patch

from src.ai.environment import WarehouseEnvironment
from src.api.warehouse_backend import (
    CommandValidationError,
    _next_policy_actions,
    _parse_command,
    _process_command_message,
    build_state_payload,
    create_app,
)


def _demo_policy():
    return {"source": "DEMO", "checkpointLoaded": False, "trained": False, "checkpointPath": None}


def _process(message, environment, *, delay=0.2, speed=1.0, paused=False):
    return _process_command_message(
        message,
        environment,
        None,
        _demo_policy,
        0.2,
        delay,
        speed,
        paused,
    )


@pytest.mark.parametrize("message", [
    "{",
    "",
    "[]",
    "{}",
    '{"type":"not_supported"}',
])
def test_invalid_json_empty_and_unsupported_commands_return_errors(message):
    environment = WarehouseEnvironment(load_layout_tasks=True)
    response, delay, speed, paused = _process(message, environment)

    assert response["type"] == "command_error"
    assert response["error"]["code"]
    assert (delay, speed, paused) == (0.2, 1.0, False)


@pytest.mark.parametrize("message", [
    '{"type":"set_fleet_size"}',
    '{"type":"set_fleet_size","size":"5"}',
    '{"type":"set_fleet_size","size":true}',
    '{"type":"set_fleet_size","size":0}',
    '{"type":"set_speed"}',
    '{"type":"set_speed","speed":"fast"}',
    '{"type":"set_speed","speed":true}',
    '{"type":"set_speed","speed":0}',
    '{"type":"set_speed","speed":NaN}',
])
def test_invalid_command_fields_return_errors(message):
    environment = WarehouseEnvironment(load_layout_tasks=True)
    response, delay, speed, paused = _process(message, environment)

    assert response["type"] == "command_error"
    assert response["error"]["code"]
    assert (delay, speed, paused) == (0.2, 1.0, False)


def test_malformed_message_does_not_prevent_a_following_valid_command():
    environment = WarehouseEnvironment(load_layout_tasks=True)
    first, delay, speed, paused = _process("not json", environment)
    second, delay, speed, paused = _process(
        '{"type":"set_speed","speed":2}',
        environment,
        delay=delay,
        speed=speed,
        paused=paused,
    )

    assert first["type"] == "command_error"
    assert second["type"] == "state_update"
    assert second["config"]["speedMultiplier"] == 2
    assert second["config"]["updateIntervalSeconds"] == 0.1


def test_pause_resume_reset_and_fleet_size_commands_return_current_state():
    environment = WarehouseEnvironment(load_layout_tasks=True)
    environment.reset()
    initial_episode = environment.episode

    paused, delay, speed, is_paused = _process('{"type":"pause"}', environment)
    resumed, delay, speed, is_paused = _process(
        '{"type":"resume"}', environment, delay=delay, speed=speed, paused=is_paused
    )
    reset, delay, speed, is_paused = _process(
        '{"type":"reset"}', environment, delay=delay, speed=speed, paused=is_paused
    )
    resized, _, _, _ = _process(
        '{"type":"set_fleet_size","size":5}',
        environment,
        delay=delay,
        speed=speed,
        paused=is_paused,
    )

    assert paused["simulationPaused"] is True
    assert resumed["simulationPaused"] is False
    assert reset["episode"] == initial_episode + 1
    assert reset["timestep"] == 0
    assert [robot["position"] for robot in reset["robots"]] == [
        list(position) for position in environment.get_active_layout().robot_starts[:5]
    ]
    assert [robot["taskId"] for robot in reset["robots"]] == ["T01", "T02", "T03", "T04", None]
    assert resized["metrics"]["totalRobots"] == 5
    assert [robot["id"] for robot in resized["robots"]] == ["R01", "R02", "R03", "R04", "R05"]


def test_default_backend_uses_moving_demo_controller_without_checkpoint():
    app = create_app()
    assert app.state.policy_source == "Rule-Based"

    environment = WarehouseEnvironment(load_layout_tasks=True)
    environment.reset()
    initial_positions = [robot.position for robot in environment.robots]
    for _ in range(12):
        environment.step(_next_policy_actions(environment))

    assert any(robot.position != initial for robot, initial in zip(environment.robots, initial_positions))
    assert len({robot.position for robot in environment.robots}) == len(environment.robots)


def test_demo_controller_replans_when_other_robots_temporarily_block_the_route():
    environment = WarehouseEnvironment(load_layout_tasks=True)
    environment.reset()

    for _ in range(3):
        previous_positions = [robot.position for robot in environment.robots]
        environment.step(_next_policy_actions(environment))

        assert all(
            robot.position != previous
            for robot, previous in zip(environment.robots, previous_positions)
        )
        assert not any(
            event.overridden and event.reason == "out_of_bounds"
            for event in environment.get_safety_events()
        )

    assert environment.movement_steps == 15
    assert len({robot.position for robot in environment.robots}) == 5


def test_default_state_snapshot_is_serializable_and_keeps_unavailable_values_null():
    environment = WarehouseEnvironment(load_layout_tasks=True)
    environment.reset()
    payload = build_state_payload(environment, _demo_policy(), 0.2, 1.0, False)

    assert len(payload["robots"]) == 5
    assert [robot["id"] for robot in payload["robots"]] == ["R01", "R02", "R03", "R04", "R05"]
    assert payload["timestep"] == 0
    assert payload["episode"] > 0
    assert json.loads(json.dumps(payload))["robots"][0]["pathSource"] == "BFS_ESTIMATE"
    assert payload["robots"][0]["rotation"] is None
    assert payload["robots"][0]["battery"] is None
    assert payload["simulationPaused"] is False
    assert payload["config"]["speedMultiplier"] == 1.0


def test_explicit_qmix_requires_checkpoint():
    with pytest.raises(ValueError, match="No QMIX checkpoint configured"):
        create_app(policy_source="qmix")


def test_auto_policy_uses_demo_when_checkpoint_is_missing():
    app = create_app(policy_source="auto", checkpoint_path="missing-qmix-checkpoint.pt")

    assert app.state.policy_source == "Rule-Based"


def test_auto_policy_uses_demo_when_checkpoint_is_incompatible():
    with patch("src.api.warehouse_backend.QMIXController", side_effect=ValueError("incompatible input shape")):
        app = create_app(policy_source="auto", checkpoint_path="incompatible.pt")

    assert app.state.policy_source == "Rule-Based"


def test_parse_command_returns_validated_data():
    assert _parse_command('{"type":"set_speed","speed":2}') == {
        "type": "set_speed", "speed": 2
    }
    with pytest.raises(CommandValidationError):
        _parse_command('{"type":"set_fleet_size","size":false}')
