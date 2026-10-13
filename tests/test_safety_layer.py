import logging

import pytest

from src.ai.state import RobotState
from src.safety.corridor import CorridorModule
from src.safety.safety_layer import SafetyLayer


class MockMovementAdapter:
    """Test-only movement convention; not the project's final movement API.

    It mirrors the fixed action IDs from ``config/robots.json`` and uses a
    simple axis-aligned coordinate transform solely to exercise the injected
    target-cell boundary.  The real Member 1 movement provider replaces this
    adapter without changing ``SafetyLayer``.
    """

    ACTION_DELTAS = {
        0: (0, 1),   # forward
        1: (0, -1),  # backward
        2: (-1, 0),  # left
        3: (1, 0),   # right
    }

    def __init__(self, warehouse_size=(5, 5)):
        self.width, self.height = warehouse_size

    def get_target_cell(self, robot_state, action):
        dx, dy = self.ACTION_DELTAS[action]
        x, y = robot_state.position
        target = (x + dx, y + dy)

        # Test-only boundary assumption: an outside move has no target.
        if not (0 <= target[0] < self.width and 0 <= target[1] < self.height):
            return None
        return target


def make_layer(cells, *, blocked_action=None, logger=None):
    corridor = CorridorModule(
        cells=cells,
        warehouse_size=(5, 5),
    )
    return SafetyLayer(
        corridor,
        MockMovementAdapter(),
        blocked_action=blocked_action,
        logger=logger,
    )


def test_safe_movement_keeps_proposed_action():
    layer = make_layer(cells=[(4, 4)])
    robot = RobotState(robot_id=1, position=(1, 1))

    decision = layer.check_action(robot, 0)

    assert decision.safe_action == 0
    assert decision.was_overridden is False
    assert decision.target_cell == (1, 2)


def test_corridor_intrusion_is_blocked_without_mutating_robot_state():
    layer = make_layer(cells=[(1, 2)])
    robot = RobotState(robot_id=1, position=(1, 1))

    decision = layer.check_action(robot, 0)

    assert decision.safe_action is None
    assert decision.final_action is None
    assert decision.was_overridden is True
    assert decision.override_reason == "human_corridor"
    assert robot.position == (1, 1)


@pytest.mark.parametrize(
    ("action", "position", "target"),
    [
        (0, (1, 1), (1, 2)),  # forward
        (1, (1, 3), (1, 2)),  # backward
        (2, (2, 1), (1, 1)),  # left
        (3, (0, 1), (1, 1)),  # right
    ],
)
def test_all_fixed_action_ids_are_checked_by_the_safety_gate(action, position, target):
    layer = make_layer(cells=[target])
    robot = RobotState(robot_id=1, position=position)

    decision = layer.check_action(robot, action)

    assert decision.was_overridden is True
    assert decision.target_cell == target


def test_multiple_robots_are_checked_independently():
    layer = make_layer(cells=[(1, 2)])
    robot_in_danger = RobotState(robot_id=1, position=(1, 1))
    robot_in_safe_area = RobotState(robot_id=2, position=(2, 2))

    blocked = layer.check_action(robot_in_danger, 0)
    allowed = layer.check_action(robot_in_safe_area, 0)

    assert blocked.was_overridden is True
    assert allowed.was_overridden is False
    assert blocked.robot_id == 1
    assert allowed.robot_id == 2
    assert robot_in_danger.position == (1, 1)
    assert robot_in_safe_area.position == (2, 2)


def test_boundary_target_is_left_to_the_movement_provider():
    layer = make_layer(cells=[(0, 0)])
    robot = RobotState(robot_id=1, position=(0, 0))

    decision = layer.check_action(robot, 1)

    assert decision.target_cell is None
    assert decision.safe_action == 1
    assert decision.was_overridden is False


def test_policy_independence_blocks_unsafe_action_without_qmix():
    layer = make_layer(cells=[(1, 2)])
    robot = RobotState(robot_id=7, position=(1, 1))

    # A raw action ID is enough; no learner, Q-values, reward, or training
    # state participates in this hard constraint.
    decision = layer.check_action(robot, proposed_action=0)

    assert decision.was_overridden is True


def test_override_metadata_is_available_and_can_be_logged(caplog):
    logger = logging.getLogger("safety-test")
    layer = make_layer(cells=[(1, 2)], logger=logger)
    robot = RobotState(robot_id=3, position=(1, 1))

    with caplog.at_level(logging.WARNING, logger="safety-test"):
        decision = layer.check_action(robot, 0)

    assert decision.robot_id == 3
    assert decision.current_cell == (1, 1)
    assert decision.proposed_action == 0
    assert decision.target_cell == (1, 2)
    assert decision.final_action is None
    assert "Safety override" in caplog.text


def test_optional_project_defined_blocked_action_is_returned():
    layer = make_layer(cells=[(1, 2)], blocked_action=99)
    robot = RobotState(robot_id=1, position=(1, 1))

    decision = layer.check_action(robot, 0)

    assert decision.safe_action == 99
    assert decision.was_overridden is True
