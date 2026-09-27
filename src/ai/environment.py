"""Stable, layout-aware multi-agent warehouse environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.ai.episode import EpisodeManager
from src.ai.reward_function import RewardFunction
from src.ai.state import GlobalState, RobotObservation, RobotState, TaskState
from src.infrastructure.config_loader import load_config
from src.infrastructure.logger import get_logger, log_safety_override
from src.safety.corridor import CorridorModule
from src.safety.safety_layer import SafetyLayer, SafetyOverrideEvent
from src.warehouse.layouts import Cell, LayoutLibrary, LayoutTask, WarehouseLayout, get_default_layout_library


class WarehouseEnvironment:
    """Python MARL environment using the existing four-action movement map.

    Action IDs remain compatible with the TypeScript simulator:
    ``0=UP (0,-1), 1=DOWN (0,+1), 2=LEFT (-1,0), 3=RIGHT (+1,0)``.
    """

    ACTIONS = {0: "forward", 1: "backward", 2: "left", 3: "right"}
    ACTION_DELTAS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    def __init__(self, num_robots: int = 5, max_steps: int = 100, *, layout_id: str | None = None,
                 layout_library: LayoutLibrary | None = None, load_layout_tasks: bool = False,
                 logger: Any | None = None):
        if isinstance(num_robots, bool) or num_robots <= 0:
            raise ValueError("num_robots must be greater than zero.")
        if isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps must be greater than zero.")
        self.num_robots, self.max_steps = num_robots, max_steps
        self.layout_library = layout_library or get_default_layout_library()
        self.load_layout_tasks = load_layout_tasks
        self.logger = logger or get_logger("warehouse_safety")
        self.robots: List[RobotState] = []
        self.tasks: List[TaskState] = []
        self.reward_function = RewardFunction()
        project_root = Path(__file__).resolve().parents[2]
        self._config = load_config(project_root / "config" / "environment.json")
        self.corridor_enabled = bool(self._config.get("human_corridors", {}).get("enabled", True))
        selected = layout_id or self._config.get("warehouse", {}).get("active_layout", "standard")
        self._configure_layout(selected)
        self.episode_manager = EpisodeManager(max_steps=max_steps)
        self.episode = 0
        self.safety_overrides = 0
        self._last_safety_events: list[SafetyOverrideEvent] = []
        self._safety_event_history: list[SafetyOverrideEvent] = []

    def _configure_layout(self, layout_id: str) -> None:
        layout = self.layout_library.get(layout_id)
        layout.validate(required_robots=self.num_robots)
        self._active_layout = layout
        self.warehouse_width, self.warehouse_height = layout.dimensions
        self.corridor = CorridorModule(layout.human_corridors, enabled=self.corridor_enabled,
                                       warehouse_size=layout.dimensions)
        self.safety_layer = SafetyLayer(self.corridor, self.get_target_cell)

    # Public integration points -------------------------------------------------
    def get_active_layout(self) -> WarehouseLayout:
        return self._active_layout

    def get_layout_ids(self) -> tuple[str, ...]:
        return self.layout_library.ids()

    def set_layout(self, layout_id: str) -> None:
        self._configure_layout(layout_id)
        self.robots, self.tasks = [], []
        self._last_safety_events, self._safety_event_history = [], []
        self.safety_overrides = 0

    def set_num_robots(self, num_robots: int) -> None:
        """Change the fleet size and start a clean episode."""
        if isinstance(num_robots, bool) or not isinstance(num_robots, int) or num_robots <= 0:
            raise ValueError("num_robots must be a positive integer.")
        self._active_layout.validate(required_robots=num_robots)
        self.num_robots = num_robots
        self.reset()

    def get_dimensions(self) -> tuple[int, int]:
        return self._active_layout.dimensions

    def is_within_bounds(self, cell: Sequence[int]) -> bool:
        return self._active_layout.is_within_bounds(cell)

    def is_walkable(self, cell: Sequence[int]) -> bool:
        return self._active_layout.is_walkable(cell)

    def get_robot_state(self, robot_id: int) -> RobotState:
        if not isinstance(robot_id, int) or isinstance(robot_id, bool) or not self.robots:
            raise KeyError(f"Unknown robot_id: {robot_id}")
        if robot_id < 0 or robot_id >= self.num_robots:
            raise KeyError(f"Unknown robot_id: {robot_id}")
        return self.robots[robot_id]

    def get_safety_events(self) -> tuple[SafetyOverrideEvent, ...]:
        return tuple(self._last_safety_events)

    def get_safety_event_history(self) -> tuple[SafetyOverrideEvent, ...]:
        return tuple(self._safety_event_history)

    def get_layout_task_locations(self) -> tuple[LayoutTask, ...]:
        return self._active_layout.task_locations

    # Lifecycle -----------------------------------------------------------------
    def reset(self):
        self._active_layout.validate(required_robots=self.num_robots)
        self.robots = [RobotState(i, self._active_layout.robot_starts[i]) for i in range(self.num_robots)]
        self.tasks = []
        if self.load_layout_tasks:
            for task in self._active_layout.task_locations:
                assigned_robot = task.assigned_robot
                if assigned_robot is None and task.task_id <= self.num_robots:
                    assigned_robot = task.task_id - 1
                self.add_task(task.task_id, task.pickup_position, task.delivery_position, assigned_robot)
        self.episode += 1
        self.safety_overrides = 0
        self._last_safety_events, self._safety_event_history = [], []
        self.episode_manager = EpisodeManager(max_steps=self.max_steps)
        return self.get_global_state(), self.get_observations()

    def add_task(self, task_id: int, pickup_position, delivery_position, assigned_robot: int | None = None):
        pickup, delivery = tuple(pickup_position), tuple(delivery_position)
        if not self.is_within_bounds(pickup) or not self.is_within_bounds(delivery):
            raise ValueError("task pickup and delivery positions must be in bounds.")
        if assigned_robot is not None and (not isinstance(assigned_robot, int) or assigned_robot < 0
                                            or assigned_robot >= self.num_robots):
            raise ValueError("assigned_robot must identify a configured robot.")
        self.tasks.append(TaskState(task_id, pickup, delivery, assigned_robot))

    def get_global_state(self) -> GlobalState:
        return GlobalState(robots=self.robots, tasks=self.tasks)

    # Observations and real movement implementation ----------------------------
    def _get_robot_task(self, robot_id: int):
        return next((task for task in self.tasks if task.assigned_robot == robot_id and not task.completed), None)

    def _occupancy_category(self, robot: RobotState, cell: Cell) -> int:
        if cell == robot.position:
            return 0
        if not self.is_within_bounds(cell):
            return 1
        if cell in self._active_layout.obstacles:
            return 2
        if any(other.position == cell for other in self.robots if other.robot_id != robot.robot_id):
            return 3
        if cell in self._active_layout.human_corridors:
            return 4
        return 0

    def get_observations(self) -> List[RobotObservation]:
        observations = []
        for robot in self.robots:
            task = self._get_robot_task(robot.robot_id)
            target = None if task is None else (task.delivery_position if robot.position == task.pickup_position else task.pickup_position)
            rx, ry = robot.position
            observations.append(RobotObservation(
                robot.robot_id,
                [[self._occupancy_category(robot, (rx + dx, ry + dy)) for dx in range(-1, 2)] for dy in range(-1, 2)],
                (0, 0) if target is None else (target[0] - rx, target[1] - ry),
                None if task is None else task.task_id,
            ))
        return observations

    def get_action_space(self) -> Dict[int, str]:
        return self.ACTIONS.copy()

    def _resolve_robot(self, robot_state_or_id):
        if isinstance(robot_state_or_id, int) and not isinstance(robot_state_or_id, bool):
            return self.get_robot_state(robot_state_or_id)
        if not hasattr(robot_state_or_id, "position"):
            raise ValueError("robot state must expose position.")
        return robot_state_or_id

    def get_target_cell(self, robot_state_or_id, action: int) -> Cell | None:
        """Single target-cell API derived from the real action semantics."""
        if action not in self.ACTION_DELTAS:
            return None
        robot = self._resolve_robot(robot_state_or_id)
        dx, dy = self.ACTION_DELTAS[action]
        target = robot.position[0] + dx, robot.position[1] + dy
        return target if self.is_within_bounds(target) else None

    calculate_target_cell = get_target_cell

    def _get_next_position(self, robot: RobotState, action: int) -> Cell:
        target = self.get_target_cell(robot, action)
        return robot.position if target is None or not self.is_walkable(target) else target

    @staticmethod
    def _manhattan_distance(a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Step ----------------------------------------------------------------------
    @staticmethod
    def _blocked_event(event: SafetyOverrideEvent, reason: str) -> SafetyOverrideEvent:
        return SafetyOverrideEvent(event.robot_id, event.current_cell, event.proposed_action,
                                   event.target_cell, None, True, reason,
                                   event.episode, event.timestep)

    def step(self, actions: Dict[int, int]):
        if len(actions) != self.num_robots:
            raise ValueError(f"Expected {self.num_robots} actions, received {len(actions)}.")
        for robot_id, action in actions.items():
            if not isinstance(robot_id, int) or robot_id < 0 or robot_id >= self.num_robots:
                raise ValueError(f"Invalid robot id: {robot_id}")
            if isinstance(action, bool) or action not in self.ACTIONS:
                raise ValueError(f"Invalid action: {action}")
        previous = {robot.robot_id: robot.position for robot in self.robots}
        previous_distances = {}
        for robot in self.robots:
            task = self._get_robot_task(robot.robot_id)
            target = None if task is None else (task.delivery_position if robot.position == task.pickup_position else task.pickup_position)
            previous_distances[robot.robot_id] = 0 if target is None else self._manhattan_distance(robot.position, target)

        # Safety checks happen for every robot before any position is mutated.
        self._last_safety_events = []
        safe_actions: dict[int, int | None] = {}
        danger = {robot.robot_id: False for robot in self.robots}
        for robot in self.robots:
            decision = self.safety_layer.check_action(robot, actions[robot.robot_id])
            event = SafetyOverrideEvent.from_decision(
                decision,
                episode=self.episode,
                timestep=self.episode_manager.current_step,
            )
            if event.final_action is not None:
                target = self.get_target_cell(robot, event.final_action)
                if target is None:
                    event = self._blocked_event(event, "out_of_bounds")
                elif not self.is_walkable(target):
                    event = self._blocked_event(event, "obstacle")
            safe_actions[robot.robot_id] = event.final_action
            self._last_safety_events.append(event)
            self._safety_event_history.append(event)
            if event.overridden:
                self.safety_overrides += 1
                danger[robot.robot_id] = True

        proposed = {robot.robot_id: robot.position if safe_actions[robot.robot_id] is None
                    else self._get_next_position(robot, safe_actions[robot.robot_id]) for robot in self.robots}
        target_map: dict[Cell, list[int]] = {}
        for robot_id, position in proposed.items():
            target_map.setdefault(position, []).append(robot_id)
        collision_robots = {robot_id for ids in target_map.values() if len(ids) > 1 for robot_id in ids}
        for index, a in enumerate(self.robots):
            for b in self.robots[index + 1:]:
                if (proposed[a.robot_id] == previous[b.robot_id] and proposed[b.robot_id] == previous[a.robot_id]
                        and proposed[a.robot_id] != previous[a.robot_id]):
                    collision_robots.update((a.robot_id, b.robot_id))
        # A robot must not move into a cell occupied by another robot whose
        # own movement was already blocked. Resolve this until no new blocked
        # robot can create an overlap.
        changed = True
        while changed:
            changed = False
            for robot in self.robots:
                if robot.robot_id in collision_robots:
                    continue
                if any(proposed[robot.robot_id] == previous[other.robot_id]
                       and other.robot_id in collision_robots for other in self.robots
                       if other.robot_id != robot.robot_id):
                    collision_robots.add(robot.robot_id)
                    changed = True
        for robot_id in sorted(collision_robots):
            if safe_actions[robot_id] is not None:
                event = self._blocked_event(self._last_safety_events[robot_id], "robot_collision")
                self._last_safety_events[robot_id] = event
                self._safety_event_history.append(event)
                safe_actions[robot_id] = None
                self.safety_overrides += 1
                danger[robot_id] = True

        for event in self._last_safety_events:
            if event.overridden:
                log_safety_override(
                    self.logger,
                    event.robot_id,
                    event.reason,
                    current_cell=event.current_cell,
                    proposed_action=event.proposed_action,
                    target_cell=event.target_cell,
                    final_action=event.final_action,
                    episode=event.episode,
                    timestep=event.timestep,
                )

        for robot in self.robots:
            blocked = robot.robot_id in collision_robots or safe_actions[robot.robot_id] is None
            event = self._last_safety_events[robot.robot_id]
            if robot.robot_id not in collision_robots:
                robot.position = proposed[robot.robot_id]
            robot.last_requested_action = actions[robot.robot_id]
            robot.last_executed_action = None if blocked else safe_actions[robot.robot_id]
            robot.action_overridden = event.overridden or robot.robot_id in collision_robots
            robot.blocked_reason = event.reason if blocked else None

        delivered = set()
        for task in self.tasks:
            if not task.completed and task.assigned_robot is not None and self.robots[task.assigned_robot].position == task.delivery_position:
                task.completed = True
                delivered.add(task.assigned_robot)
        self.episode_manager.step()
        rewards = {}
        for robot in self.robots:
            task = self._get_robot_task(robot.robot_id)
            target = None if task is None else (task.delivery_position if robot.position == task.pickup_position else task.pickup_position)
            rewards[robot.robot_id] = self.reward_function.calculate(
                previous_distances[robot.robot_id], 0 if target is None else self._manhattan_distance(robot.position, target),
                delivered=robot.robot_id in delivered, collision=robot.robot_id in collision_robots, danger=danger[robot.robot_id])
        terminated = self.episode_manager.is_terminated(sum(not task.completed for task in self.tasks))
        return self.get_observations(), rewards, terminated, self.get_global_state()
