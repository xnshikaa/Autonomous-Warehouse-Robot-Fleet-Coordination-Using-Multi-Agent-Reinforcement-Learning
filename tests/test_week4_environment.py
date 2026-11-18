import json
from collections import deque

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

    standard = library.get("standard")
    assert all((x, 9) not in standard.human_corridors for x in (0, 4, 9, 14, 19))
    assert (10, 9) in standard.human_corridors


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


def _safe_path(environment, start, goal):
    layout = environment.get_active_layout()
    blocked = set(layout.obstacles) | set(layout.human_corridors)
    queue = deque([start])
    parent = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return list(reversed(path))
        for dx, dy in environment.ACTION_DELTAS.values():
            nxt = (current[0] + dx, current[1] + dy)
            if (nxt not in parent and environment.is_within_bounds(nxt)
                    and environment.is_walkable(nxt) and nxt not in blocked):
                parent[nxt] = current
                queue.append(nxt)
    return None


def test_robot_can_complete_cross_corridor_delivery_without_entering_worker_zone():
    environment = WarehouseEnvironment(num_robots=1, max_steps=100)
    environment.reset()
    pickup, delivery = (2, 1), (2, 15)
    environment.add_task(1, pickup, delivery, assigned_robot=0)

    for goal in (pickup, delivery):
        start = environment.get_robot_state(0).position
        path = _safe_path(environment, start, goal)
        assert path is not None
        for current, nxt in zip(path, path[1:]):
            delta = (nxt[0] - current[0], nxt[1] - current[1])
            action = next(action for action, move in environment.ACTION_DELTAS.items() if move == delta)
            environment.step({0: action})
            assert nxt not in environment.get_active_layout().human_corridors

    assert environment.tasks[0].completed
    assert environment.safety_overrides == 0


def test_task_observation_and_reward_follow_carrying_phase():
    environment = WarehouseEnvironment(num_robots=1, max_steps=100)
    environment.reset()
    pickup, delivery = (3, 1), (3, 4)
    environment.add_task(1, pickup, delivery, assigned_robot=0)

    path = _safe_path(environment, (0, 0), pickup)
    for current, nxt in zip(path, path[1:]):
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        action = next(action for action, move in environment.ACTION_DELTAS.items() if move == delta)
        _, rewards, _, _ = environment.step({0: action})

    assert environment.get_robot_state(0).carrying_item
    assert environment.get_observations()[0].nearest_task_bearing == (0, 3)
    assert rewards[0] == pytest.approx(0.99)

    observations, rewards, _, _ = environment.step({0: 1})
    assert observations[0].nearest_task_bearing == (0, 2)
    assert rewards[0] == pytest.approx(0.99)


def test_progress_reward_uses_safe_path_distance_across_corridor():
    environment = WarehouseEnvironment(num_robots=1, max_steps=100)
    environment.reset()
    pickup, delivery = (2, 1), (2, 15)
    environment.add_task(1, pickup, delivery, assigned_robot=0)

    to_pickup = _safe_path(environment, (0, 0), pickup)
    for current, nxt in zip(to_pickup, to_pickup[1:]):
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        action = next(action for action, move in environment.ACTION_DELTAS.items() if move == delta)
        environment.step({0: action})

    route = _safe_path(environment, pickup, delivery)
    next_cell = route[1]
    delta = (next_cell[0] - pickup[0], next_cell[1] - pickup[1])
    action = next(action for action, move in environment.ACTION_DELTAS.items() if move == delta)
    _, rewards, _, _ = environment.step({0: action})

    assert environment._manhattan_distance(next_cell, delivery) > environment._manhattan_distance(pickup, delivery)
    assert environment._path_distance(next_cell, delivery) == environment._path_distance(pickup, delivery) - 1
    assert rewards[0] == pytest.approx(0.99)


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
    _, rewards, _, _ = environment.step({0: 3})
    assert environment.get_robot_state(0).position == (0, 2)
    assert environment.get_safety_events()[0].reason == "obstacle"
    assert rewards[0] == pytest.approx(-0.11)


def test_action_mask_excludes_boundaries_obstacles_and_worker_corridors():
    environment = WarehouseEnvironment(num_robots=1)
    environment.reset()
    assert environment.get_action_masks()[0] == [False, True, False, True]

    environment.get_robot_state(0).position = (2, 1)
    assert environment.get_action_masks()[0][1] is False

    environment.get_robot_state(0).position = (10, 8)
    assert environment.get_action_masks()[0][1] is False


def test_same_cell_conflict_allows_one_robot_to_progress_safely():
    environment = WarehouseEnvironment(num_robots=2)
    environment.reset()
    environment.get_robot_state(1).position = (2, 0)

    _, rewards, _, _ = environment.step({0: 3, 1: 2})

    assert environment.get_robot_state(0).position == (1, 0)
    assert environment.get_robot_state(1).position == (2, 0)
    assert environment.get_safety_events()[0].overridden is False
    assert environment.get_safety_events()[1].reason == "robot_collision"
    assert rewards[0] == pytest.approx(-0.01)
    assert rewards[1] == pytest.approx(-10.01)


def test_corridor_override_has_required_telemetry_and_does_not_move_robot():
    environment = WarehouseEnvironment(num_robots=1)
    environment.reset()
    environment.get_robot_state(0).position = (10, 8)
    _, rewards, _, _ = environment.step({0: 1})
    event = environment.get_safety_events()[0]
    assert event.robot_id == 0
    assert event.current_cell == (10, 8)
    assert event.proposed_action == 1
    assert event.target_cell == (10, 9)
    assert event.final_action is None
    assert event.overridden is True
    assert event.reason == "human_corridor"
    assert environment.get_robot_state(0).position == (10, 8)
    assert json.loads(json.dumps(event.to_dict()))["target_cell"] == [10, 9]
    assert rewards[0] == pytest.approx(-20.01)


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
