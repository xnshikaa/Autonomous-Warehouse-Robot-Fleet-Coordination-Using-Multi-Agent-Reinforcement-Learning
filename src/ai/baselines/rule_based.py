"""Simple deterministic controller for baseline evaluation."""

from __future__ import annotations

from src.ai.environment import WarehouseEnvironment


class RuleBasedController:
    """Follow a shortest safe path toward each robot's assigned task."""

    def __init__(self, environment: WarehouseEnvironment):
        self.environment = environment

    def get_actions(self) -> dict[int, int]:
        actions: dict[int, int] = {}
        legal_by_robot = self.environment.get_action_masks()
        for robot in self.environment.robots:
            task = self.environment._get_robot_task(robot.robot_id)
            if task is None:
                actions[robot.robot_id] = 0
                continue
            target = (
                task.delivery_position
                if robot.carrying_item
                else task.pickup_position
            )
            safe_actions = [
                action for action, legal in enumerate(legal_by_robot[robot.robot_id])
                if legal
            ]
            actions[robot.robot_id] = min(
                safe_actions,
                key=lambda action: self.environment._path_distance(
                    self.environment.get_target_cell(robot, action), target
                ),
            )
        return actions
