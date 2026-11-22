import logging

import pytest

from python_infra.verify_single_robot_mission import trial
from src.ai.environment import WarehouseEnvironment
from src.ai.live_navigation import WarehouseNavigator
from src.ai.single_robot_mission import MISSION_SEED


@pytest.mark.parametrize(
    ("positions", "actions", "expected"),
    [
        ([(0, 0), (1, 0)], {0: 3, 1: None}, [(0, 0), (1, 0)]),
        ([(0, 0), (1, 0)], {0: 3, 1: 2}, [(0, 0), (1, 0)]),
        ([(0, 0), (1, 0)], {0: 3, 1: 3}, [(0, 0), (2, 0)]),
        ([(0, 0), (2, 0)], {0: 3, 1: 2}, [(1, 0), (2, 0)]),
        ([(0, 0), (2, 0)], {0: 0, 1: None}, [(0, 0), (2, 0)]),
    ],
)
def test_live_resolver_protects_cells_and_waiting_robots(positions, actions, expected):
    env = WarehouseEnvironment(num_robots=2, conservative_motion=True)
    env.reset()
    for robot, position in zip(env.robots, positions, strict=True):
        robot.position = position
    env.step(actions)
    assert [robot.position for robot in env.robots] == expected
    for robot_id, action in actions.items():
        if action is None:
            assert env.robots[robot_id].last_executed_action is None
            assert not env.robots[robot_id].action_overridden


def test_live_navigator_routes_around_stationary_occupant():
    env = WarehouseEnvironment(num_robots=2, conservative_motion=True)
    env.reset()
    env.robots[0].position = (0, 0)
    env.robots[1].position = (1, 0)
    env.add_task(1, (2, 0), (2, 1), 0)
    navigator = WarehouseNavigator(env)
    assert navigator.get_actions() == {0: 1, 1: None}
    assert (1, 0) not in navigator.routes[0]


def test_one_hundred_seeded_missions_complete_and_return_to_idle():
    logger = logging.getLogger("mission_regression")
    env = WarehouseEnvironment(num_robots=1, conservative_motion=True, logger=logger)
    navigator = WarehouseNavigator(env)
    for seed in range(MISSION_SEED, MISSION_SEED + 100):
        row = trial(seed, env, env, navigator.get_actions)
        assert row["pickup_step"] < row["delivery_step"]
        assert row["idle"] and not row["stuck"]
        for key in ("physical_collisions", "shelf_or_wall_contacts", "corridor_intrusions",
                    "blocked_moves", "two_cell_cycles"):
            assert row[key] == 0, (seed, key)
