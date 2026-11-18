"""Simple deterministic controller for baseline evaluation."""

from __future__ import annotations

from src.ai.environment import WarehouseEnvironment


class RuleBasedController:
    """Move each robot toward its assigned task using the environment action map."""

    def __init__(self, environment: WarehouseEnvironment):
        self.environment = environment

    def get_actions(self) -> dict[int, int]:
        actions: dict[int, int] = {}
        for robot in self.environment.robots:
            task = self.environment._get_robot_task(robot.robot_id)
            if task is None:
                actions[robot.robot_id] = 0
                continue
            target = (
                task.delivery_position
                if robot.position == task.pickup_position
                else task.pickup_position
            )
            candidates: list[int] = []
            dx = target[0] - robot.position[0]
            dy = target[1] - robot.position[1]
            if dx:
                candidates.append(3 if dx > 0 else 2)
            if dy:
                candidates.append(1 if dy > 0 else 0)
            candidates.extend(action for action in self.environment.ACTIONS if action not in candidates)
            actions[robot.robot_id] = next(
                (
                    action
                    for action in candidates
                    if self.environment.get_target_cell(robot, action) is not None
                    and self.environment.is_walkable(
                        self.environment.get_target_cell(robot, action)
                    )
                ),
                0,
            )
        return actions
