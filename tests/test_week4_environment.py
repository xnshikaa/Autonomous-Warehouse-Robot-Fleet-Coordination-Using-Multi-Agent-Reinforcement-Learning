import json

import pytest

from src.ai.environment import WarehouseEnvironment
from src.api.warehouse_backend import build_state_payload, create_app
from src.warehouse.layouts import LayoutValidationError, WarehouseLayout, get_default_layout_library


def test_every_builtin_layout_has_one_schema_and_corridors():
    library = get_default_layout_library()
    assert library.ids() == ("cross_dock", "standard")
    for layout in library.all():
        assert layout.width == 20 and layout.height == 20
        assert layout.human_corridors
        assert layout.boundaries == {"min_x": 0, "max_x": 19, "min_y": 0, "max_y": 19}
        assert all(layout.is_within_bounds(cell) for cell in layout.human_corridors)


def test_malformed_layout_is_rejected():
    with pytest.raises(LayoutValidationError):
        WarehouseLayout(
            "bad", "Bad", 2, 2, frozenset({(0, 0)}), ((0, 0),), (), frozenset({(0, 0)})
        )


def test_active_layout_and_robot_state_access():
    environment = WarehouseEnvironment(num_robots=2, layout_id="cross_dock")
    environment.reset()
    assert environment.get_active_layout().layout_id == "cross_dock"
    assert environment.get_dimensions() == (20, 20)
    assert environment.get_robot_state(0).position == (0, 0)
    assert environment.is_within_bounds((19, 19))
    assert not environment.is_within_bounds((20, 19))


def test_target_cell_uses_real_action_mapping_and_boundary():
    environment = WarehouseEnvironment(num_robots=1)
    environment.reset()
    assert environment.get_target_cell(0, 0) is None
    assert environment.get_target_cell(0, 1) == (0, 1)
    assert environment.get_target_cell(0, 3) == (1, 0)
    assert environment.calculate_target_cell(environment.get_robot_state(0), 2) is None


def test_safe_movement_updates_state_before_observation_is_returned():
    environment = WarehouseEnvironment(num_robots=1)
    environment.reset()
    environment.add_task(1, (3, 0), (3, 1), assigned_robot=0)
    observations, _, terminated, state = environment.step({0: 3})
    assert environment.get_robot_state(0).position == (1, 0)
    assert state.robots[0].position == (1, 0)
    assert observations[0].robot_id == 0
    assert terminated is False


def test_boundary_and_obstacle_are_blocked_before_movement():
    environment = WarehouseEnvironment(num_robots=1)
    environment.reset()
    environment.step({0: 0})
    assert environment.get_robot_state(0).position == (0, 0)
    assert environment.get_safety_events()[0].reason == "out_of_bounds"

    environment.reset()
    environment.get_robot_state(0).position = (0, 2)
    environment.step({0: 3})
    assert environment.get_robot_state(0).position == (0, 2)
    assert environment.get_safety_events()[0].reason == "obstacle"


def test_corridor_override_has_required_telemetry_and_does_not_move_robot():
    environment = WarehouseEnvironment(num_robots=1)
    environment.reset()
    environment.get_robot_state(0).position = (0, 8)
    environment.step({0: 1})
    event = environment.get_safety_events()[0]
    assert event.robot_id == 0
    assert event.current_cell == (0, 8)
    assert event.proposed_action == 1
    assert event.target_cell == (0, 9)
    assert event.final_action is None
    assert event.overridden is True
    assert event.reason == "human_corridor"
    assert environment.get_robot_state(0).position == (0, 8)
    assert json.loads(json.dumps(event.to_dict()))["target_cell"] == [0, 9]


def test_multiple_robots_are_independent_and_reset_clears_episode_state():
    environment = WarehouseEnvironment(num_robots=2)
    environment.reset()
    environment.step({0: 3, 1: 2})
    assert environment.get_robot_state(0).position == (1, 0)
    assert environment.get_robot_state(1).position == (0, 1)
    assert environment.get_safety_events()[0].overridden is False
    assert environment.get_safety_events()[1].overridden is True
    environment.reset()
    assert environment.get_robot_state(0).position == (0, 0)
    assert not environment.get_safety_events()
    assert environment.safety_overrides == 0


def test_layout_switch_and_frontend_payload():
    environment = WarehouseEnvironment(num_robots=1)
    environment.set_layout("cross_dock")
    environment.reset()
    payload = build_state_payload(environment)
    json.dumps(payload)
    assert payload["config"]["layoutId"] == "cross_dock"
    assert payload["robots"][0]["id"] == "R01"
    assert payload["layout"]["human_corridors"]["cells"]


def test_backend_starts_with_robots_and_accepts_fleet_size_commands():
    environment = WarehouseEnvironment()
    create_app(environment, interval_seconds=0.01)
    assert len(environment.robots) == 5
    environment.set_num_robots(10)
    assert len(environment.robots) == 10
