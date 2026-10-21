from pathlib import Path
from typing import Dict, List

from src.ai.episode import EpisodeManager
from src.ai.reward_function import RewardFunction
from src.ai.state import GlobalState, RobotObservation, RobotState, TaskState
from src.infrastructure.config_loader import load_config
from src.safety.corridor import CorridorModule
from src.safety.safety_layer import SafetyLayer


class WarehouseEnvironment:
    """
    Basic multi-agent warehouse environment.

    Week 4 responsibilities:
    - Maintain robot and task state
    - Move robots using discrete actions
    - Apply the hard safety layer
    - Detect collisions
    - Detect task completion
    - Calculate rewards
    - Provide observations and global state
    """

    ACTIONS = {
        0: "forward",
        1: "backward",
        2: "left",
        3: "right",
    }

    ACTION_DELTAS = {
        0: (0, -1),   # UP
        1: (0, 1),    # DOWN
        2: (-1, 0),   # LEFT
        3: (1, 0),    # RIGHT
    }

    WAREHOUSE_WIDTH = 20
    WAREHOUSE_HEIGHT = 20

    def __init__(self, num_robots: int = 5, max_steps: int = 100):
        self.num_robots = num_robots
        self.max_steps = max_steps

        self.robots: List[RobotState] = []
        self.tasks: List[TaskState] = []

        self.episode_manager = EpisodeManager(
            max_steps=max_steps
        )

        self.reward_function = RewardFunction()

        self._load_environment_config()

        self.corridor = CorridorModule(
            cells=self.corridor_cells,
            enabled=self.corridor_enabled,
            warehouse_size=(
                self.warehouse_width,
                self.warehouse_height,
            ),
        )

        self.safety_layer = SafetyLayer(
            self.corridor,
            self.get_target_cell,
        )

        self.safety_overrides = 0

    def _load_environment_config(self):
        """Load the existing warehouse configuration."""

        project_root = Path(__file__).resolve().parents[2]

        config_path = (
            project_root
            / "config"
            / "environment.json"
        )

        config = load_config(config_path)

        warehouse_config = config.get(
            "warehouse",
            {}
        )

        self.warehouse_width = int(
            warehouse_config.get(
                "width",
                self.WAREHOUSE_WIDTH,
            )
        )

        self.warehouse_height = int(
            warehouse_config.get(
                "height",
                self.WAREHOUSE_HEIGHT,
            )
        )

        corridor_config = config.get(
            "human_corridors",
            {}
        )

        self.corridor_enabled = bool(
            corridor_config.get(
                "enabled",
                True,
            )
        )

        self.corridor_cells = corridor_config.get(
            "cells",
            []
        )

    def reset(self):
        """Reset the environment to its initial state."""

        self.robots = [
            RobotState(
                robot_id=robot_id,
                position=(0, robot_id),
            )
            for robot_id in range(self.num_robots)
        ]

        self.tasks = []

        self.safety_overrides = 0

        self.episode_manager = EpisodeManager(
            max_steps=self.max_steps
        )

        observations = self.get_observations()

        global_state = self.get_global_state()

        return global_state, observations

    def add_task(
        self,
        task_id: int,
        pickup_position,
        delivery_position,
        assigned_robot: int | None = None,
    ):
        """Add a task to the warehouse."""

        task = TaskState(
            task_id=task_id,
            pickup_position=tuple(
                pickup_position
            ),
            delivery_position=tuple(
                delivery_position
            ),
            assigned_robot=assigned_robot,
        )

        self.tasks.append(task)

    def get_global_state(self) -> GlobalState:
        """Return the current global environment state."""

        return GlobalState(
            robots=self.robots,
            tasks=self.tasks,
        )

    def get_observations(self) -> List[RobotObservation]:
        """
        Generate a basic local observation for every robot.

        The detailed 50-dimensional encoding is handled by
        ObservationEncoder in the MARL layer.
        """

        observations = []

        for robot in self.robots:

            current_task_id = None
            nearest_task_bearing = (0, 0)

            assigned_tasks = [
                task
                for task in self.tasks
                if (
                    task.assigned_robot
                    == robot.robot_id
                    and not task.completed
                )
            ]

            if assigned_tasks:

                task = assigned_tasks[0]

                current_task_id = task.task_id

                if robot.position == task.pickup_position:
                    target = task.delivery_position
                else:
                    target = task.pickup_position

                nearest_task_bearing = (
                    target[0] - robot.position[0],
                    target[1] - robot.position[1],
                )

            observations.append(
                RobotObservation(
                    robot_id=robot.robot_id,
                    occupancy_grid=[
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    nearest_task_bearing=nearest_task_bearing,
                    current_task_id=current_task_id,
                )
            )

        return observations

    def get_action_space(self) -> Dict[int, str]:
        """Return the discrete action space."""

        return self.ACTIONS.copy()

    def get_target_cell(
        self,
        robot_state: RobotState,
        action: int,
    ):
        """
        Return the cell targeted by a proposed action.

        This method is used by the SafetyLayer.
        """

        if action not in self.ACTION_DELTAS:
            return None

        dx, dy = self.ACTION_DELTAS[action]

        x, y = robot_state.position

        target = (
            x + dx,
            y + dy,
        )

        if not self._is_inside_warehouse(
            target
        ):
            return None

        return target

    def _get_next_position(
        self,
        robot_state: RobotState,
        action: int,
    ):
        """Calculate the next position for a robot."""

        dx, dy = self.ACTION_DELTAS[action]

        x, y = robot_state.position

        next_position = (
            x + dx,
            y + dy,
        )

        if not self._is_inside_warehouse(
            next_position
        ):
            return robot_state.position

        return next_position

    def _is_inside_warehouse(
        self,
        position,
    ) -> bool:
        """Check whether a position is inside the warehouse."""

        x, y = position

        return (
            0 <= x < self.warehouse_width
            and
            0 <= y < self.warehouse_height
        )

    @staticmethod
    def _manhattan_distance(
        position_a,
        position_b,
    ) -> int:
        """Calculate Manhattan distance between two cells."""

        return (
            abs(
                position_a[0]
                - position_b[0]
            )
            +
            abs(
                position_a[1]
                - position_b[1]
            )
        )

    def _get_robot_task(
        self,
        robot_id: int,
    ):
        """Return the active task assigned to a robot."""

        for task in self.tasks:

            if (
                task.assigned_robot == robot_id
                and not task.completed
            ):
                return task

        return None

    def step(
        self,
        actions: Dict[int, int],
    ):
        """
        Execute one environment step.

        Returns:
            observations,
            rewards,
            terminated,
            global_state
        """

        if len(actions) != self.num_robots:
            raise ValueError(
                f"Expected {self.num_robots} actions, "
                f"received {len(actions)}."
            )

        for robot_id, action in actions.items():

            if (
                robot_id < 0
                or robot_id >= self.num_robots
            ):
                raise ValueError(
                    f"Invalid robot id: {robot_id}"
                )

            if action not in self.ACTIONS:
                raise ValueError(
                    f"Invalid action: {action}"
                )

        previous_positions = {
            robot.robot_id: robot.position
            for robot in self.robots
        }

        previous_distances = {}

        for robot in self.robots:

            task = self._get_robot_task(
                robot.robot_id
            )

            if task is None:

                previous_distances[
                    robot.robot_id
                ] = 0

            else:

                if (
                    robot.position
                    == task.pickup_position
                ):
                    target = task.delivery_position
                else:
                    target = task.pickup_position

                previous_distances[
                    robot.robot_id
                ] = self._manhattan_distance(
                    robot.position,
                    target,
                )

        # --------------------------------------------------
        # SAFETY CHECK
        # --------------------------------------------------

        safe_actions = {}

        danger_flags = {
            robot.robot_id: False
            for robot in self.robots
        }

        for robot in self.robots:

            proposed_action = actions[
                robot.robot_id
            ]

            decision = (
                self.safety_layer.check_action(
                    robot,
                    proposed_action,
                )
            )

            safe_actions[
                robot.robot_id
            ] = decision.safe_action

            if decision.was_overridden:

                self.safety_overrides += 1

                danger_flags[
                    robot.robot_id
                ] = True

        # --------------------------------------------------
        # CALCULATE PROPOSED POSITIONS
        # --------------------------------------------------

        proposed_positions = {}

        for robot in self.robots:

            safe_action = safe_actions[
                robot.robot_id
            ]

            if safe_action is None:

                proposed_positions[
                    robot.robot_id
                ] = robot.position

            else:

                proposed_positions[
                    robot.robot_id
                ] = self._get_next_position(
                    robot,
                    safe_action,
                )

        # --------------------------------------------------
        # COLLISION DETECTION
        # --------------------------------------------------

        target_to_robots = {}

        for (
            robot_id,
            position,
        ) in proposed_positions.items():

            target_to_robots.setdefault(
                position,
                []
            ).append(robot_id)

        collision_robots = set()

        for robot_ids in target_to_robots.values():

            if len(robot_ids) > 1:

                collision_robots.update(
                    robot_ids
                )

        # --------------------------------------------------
        # DIRECT SWAP COLLISION
        # --------------------------------------------------

        for robot_a in self.robots:

            for robot_b in self.robots:

                if (
                    robot_a.robot_id
                    >= robot_b.robot_id
                ):
                    continue

                if (
                    proposed_positions[
                        robot_a.robot_id
                    ]
                    ==
                    previous_positions[
                        robot_b.robot_id
                    ]
                    and
                    proposed_positions[
                        robot_b.robot_id
                    ]
                    ==
                    previous_positions[
                        robot_a.robot_id
                    ]
                    and
                    proposed_positions[
                        robot_a.robot_id
                    ]
                    !=
                    previous_positions[
                        robot_a.robot_id
                    ]
                ):

                    collision_robots.add(
                        robot_a.robot_id
                    )

                    collision_robots.add(
                        robot_b.robot_id
                    )

        # --------------------------------------------------
        # APPLY MOVEMENT
        # --------------------------------------------------

        for robot in self.robots:

            if (
                robot.robot_id
                in collision_robots
            ):
                continue

            robot.position = (
                proposed_positions[
                    robot.robot_id
                ]
            )

        # --------------------------------------------------
        # TASK COMPLETION
        # --------------------------------------------------

        delivered_robots = set()

        for task in self.tasks:

            if (
                task.completed
                or task.assigned_robot is None
            ):
                continue

            robot = self.robots[
                task.assigned_robot
            ]

            if (
                robot.position
                == task.delivery_position
            ):

                task.completed = True

                delivered_robots.add(
                    robot.robot_id
                )

        # --------------------------------------------------
        # EPISODE STEP
        # --------------------------------------------------

        self.episode_manager.step()

        # --------------------------------------------------
        # REWARD CALCULATION
        # --------------------------------------------------

        rewards = {}

        for robot in self.robots:

            task = self._get_robot_task(
                robot.robot_id
            )

            if task is None:

                current_distance = 0

            else:

                if (
                    robot.position
                    == task.pickup_position
                ):
                    target = task.delivery_position
                else:
                    target = task.pickup_position

                current_distance = (
                    self._manhattan_distance(
                        robot.position,
                        target,
                    )
                )

            delivered = (
                robot.robot_id
                in delivered_robots
            )

            collision = (
                robot.robot_id
                in collision_robots
            )

            danger = danger_flags[
                robot.robot_id
            ]

            reward = (
                self.reward_function.calculate(
                    previous_distance=
                    previous_distances[
                        robot.robot_id
                    ],
                    current_distance=
                    current_distance,
                    delivered=delivered,
                    collision=collision,
                    danger=danger,
                )
            )

            rewards[
                robot.robot_id
            ] = reward

        # --------------------------------------------------
        # TERMINATION
        # --------------------------------------------------

        tasks_remaining = sum(
            1
            for task in self.tasks
            if not task.completed
        )

        terminated = (
            self.episode_manager.is_terminated(
                tasks_remaining
            )
        )

        observations = (
            self.get_observations()
        )

        global_state = (
            self.get_global_state()
        )

        return (
            observations,
            rewards,
            terminated,
            global_state,
        )