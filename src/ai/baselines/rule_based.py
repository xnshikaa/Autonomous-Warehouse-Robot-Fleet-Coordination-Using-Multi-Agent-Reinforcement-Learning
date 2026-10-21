"""
Rule-Based Baseline Controller

Week 4 - Member 3

This controller provides a simple deterministic baseline for the
warehouse robot environment. It does not learn from experience.

Each robot moves toward the target associated with its assigned task.
The WarehouseEnvironment remains responsible for safety checks,
collision handling, rewards, and episode termination.
"""

from typing import Dict, Optional, Tuple

from src.ai.environment import WarehouseEnvironment
from src.ai.state import RobotState


class RuleBasedController:
    """
    Deterministic rule-based controller for warehouse robots.

    The controller selects movement actions that reduce the Manhattan
    distance between a robot and its current target.

    This provides a simple baseline that can later be compared with
    learning-based approaches such as IQL and QMIX.
    """

    def __init__(self, environment: WarehouseEnvironment):
        self.environment = environment

        # Remember which robots have reached
        # the pickup location for their task.
        self.picked_up_tasks = set()

    def get_actions(self) -> Dict[int, int]:
        """
        Generate one action for every robot.

        Returns:
            Dictionary mapping robot_id -> action.
        """

        actions: Dict[int, int] = {}

        for robot in self.environment.robots:
            actions[robot.robot_id] = self._choose_action(robot)

        return actions

    def _choose_action(self, robot: RobotState) -> int:
        """
        Select a deterministic movement action for one robot.
        """

        task = self.environment._get_robot_task(
            robot.robot_id
        )

        if task is None:
            return self._safe_default_action(robot)

        task_key = (
            robot.robot_id,
            task.task_id,
        )

        # Mark the task as picked up once the robot
        # reaches the pickup position.
        if robot.position == task.pickup_position:
            self.picked_up_tasks.add(task_key)

        # Before pickup, move toward pickup.
        # After pickup, continue toward delivery.
        if task_key in self.picked_up_tasks:
            target = task.delivery_position
        else:
            target = task.pickup_position

        return self._move_toward_target(
            robot,
            target,
    )

    def _move_toward_target(
        self,
        robot: RobotState,
        target: Tuple[int, int],
    ) -> int:
        """
        Choose an action that reduces Manhattan distance to the target.
        """

        current_x, current_y = robot.position
        target_x, target_y = target

        candidate_actions = []

        # Prefer horizontal movement first.
        if target_x > current_x:
            candidate_actions.append(
                self._find_action_for_delta(1, 0)
            )

        elif target_x < current_x:
            candidate_actions.append(
                self._find_action_for_delta(-1, 0)
            )

        # Then vertical movement.
        if target_y > current_y:
            candidate_actions.append(
                self._find_action_for_delta(0, 1)
            )

        elif target_y < current_y:
            candidate_actions.append(
                self._find_action_for_delta(0, -1)
            )

        # Remove unavailable actions.
        candidate_actions = [
            action
            for action in candidate_actions
            if action is not None
        ]

        # Select the first valid target cell.
        for action in candidate_actions:

            target_cell = self.environment.get_target_cell(
                robot,
                action,
            )

            if target_cell is not None:
                return action

        return self._safe_default_action(robot)

    def _find_action_for_delta(
        self,
        dx: int,
        dy: int,
    ) -> Optional[int]:
        """
        Find the environment action corresponding to a movement delta.
        """

        for action, delta in self.environment.ACTION_DELTAS.items():

            if delta == (dx, dy):
                return action

        return None

    def _safe_default_action(
        self,
        robot: RobotState,
    ) -> int:
        """
        Return a deterministic valid action when there is no active target.

        The environment has no explicit NO_OP policy action, so the
        controller chooses the first action whose target remains inside
        the warehouse.
        """

        for action in self.environment.get_action_space():

            if (
                self.environment.get_target_cell(
                    robot,
                    action,
                )
                is not None
            ):
                return action

        # Defensive fallback.
        return next(
            iter(
                self.environment.get_action_space()
            )
        )