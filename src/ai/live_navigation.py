"""Reservation-aware navigation for the live rule-based controller."""
from collections import deque

from src.ai.environment import WarehouseEnvironment
from src.warehouse.layouts import Cell


class WarehouseNavigator:
    """Reserve starting cells and next destinations, replanning each tick."""

    def __init__(self, environment: WarehouseEnvironment):
        self.environment = environment
        self.routes: dict[int, list[Cell]] = {}
        self.waiting: set[int] = set()
        self._episode = -1
        self._detours: dict[tuple[int, int, bool], set[Cell]] = {}

    def _route(self, start: Cell, goal: Cell, blocked: set[Cell]) -> list[Cell]:
        if goal in blocked:
            return []
        queue = deque([start])
        parent: dict[Cell, Cell | None] = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                path = [cell]
                while (previous := parent[path[-1]]) is not None:
                    path.append(previous)
                return path[::-1]
            for dx, dy in self.environment.ACTION_DELTAS.values():
                neighbor = cell[0] + dx, cell[1] + dy
                if neighbor not in parent and neighbor not in blocked and self.environment.is_walkable(neighbor):
                    parent[neighbor] = cell
                    queue.append(neighbor)
        return []

    def get_actions(self) -> dict[int, int | None]:
        env = self.environment
        if self._episode != env.episode:
            self._episode = env.episode
            self._detours.clear()
        self.routes.clear()
        self.waiting.clear()
        positions = {robot.position for robot in env.robots}
        reserved: set[Cell] = set()
        corridor = set(env.get_active_layout().human_corridors)
        priority = env.episode_manager.current_step % env.num_robots
        ordered = env.robots[priority:] + env.robots[:priority]
        actions: dict[int, int | None] = {}
        for robot in ordered:
            robot_id = robot.robot_id
            actions[robot_id] = None
            task = env._get_robot_task(robot_id)
            if task is None:
                continue
            blocked = (positions | corridor | reserved) - {robot.position}
            goal = env._task_target(robot, task)
            route = self._route(robot.position, goal, blocked)
            phase = (robot_id, task.task_id, robot.carrying_item)
            visited = self._detours.setdefault(phase, set())
            if not route:
                # Yield into a free cell when the route is cut off. Keep the
                # assignment and never revisit a refuge during this blockage.
                self.waiting.add(robot_id)
                visited.add(robot.position)
                refuges = [
                    cell for action in env.ACTIONS
                    if (cell := env.get_target_cell(robot, action)) is not None
                    and cell not in blocked and cell not in visited and env.is_walkable(cell)
                ]
                if refuges:
                    route = [robot.position, min(refuges, key=lambda cell: (env._path_distance(cell, goal), cell))]
            else:
                visited.clear()
            self.routes[robot_id] = route
            if len(route) > 1:
                next_cell = route[1]
                actions[robot_id] = next(
                    action for action in env.ACTIONS if env.get_target_cell(robot, action) == next_cell
                )
                reserved.add(next_cell)
        return actions
