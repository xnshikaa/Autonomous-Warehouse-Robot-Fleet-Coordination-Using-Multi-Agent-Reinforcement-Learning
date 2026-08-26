from src.ai.reward_function import RewardFunction
from src.ai.episode import EpisodeManager
from src.ai.state import (
    RobotState,
    TaskState,
    GlobalState,
    RobotObservation
)


class WarehouseEnvironment:
    """
    Basic multi-agent warehouse RL environment.

    This module defines the interface between:
    - robot states
    - task states
    - observations
    - rewards
    - episode termination
    - robot actions
    """

    ACTIONS = {
        0: "forward",
        1: "backward",
        2: "left",
        3: "right"
    }

    def __init__(self, num_robots=5, max_steps=100):
        self.num_robots = num_robots
        self.max_steps = max_steps

        self.reward_function = RewardFunction()
        self.episode_manager = EpisodeManager(max_steps)

        self.robots = []
        self.tasks = []

    def reset(self):
        """
        Reset the environment.

        Returns:
            Global state and local observations.
        """

        self.episode_manager.reset()

        self.robots = [
            RobotState(
                robot_id=robot_id,
                position=(0, robot_id)
            )
            for robot_id in range(self.num_robots)
        ]

        self.tasks = []

        return self.get_global_state(), self.get_observations()

    def add_task(
        self,
        task_id,
        pickup_position,
        delivery_position,
        assigned_robot=None
    ):
        """Add a task to the environment."""

        task = TaskState(
            task_id=task_id,
            pickup_position=pickup_position,
            delivery_position=delivery_position,
            assigned_robot=assigned_robot
        )

        self.tasks.append(task)

    def get_global_state(self):
        """
        Return the global state used during centralized training.
        """

        return GlobalState(
            robots=self.robots.copy(),
            tasks=self.tasks.copy()
        )

    def get_observations(self):
        """
        Return local observations for every robot.
        """

        observations = []

        for robot in self.robots:

            assigned_task = next(
                (
                    task
                    for task in self.tasks
                    if task.assigned_robot == robot.robot_id
                    and not task.completed
                ),
                None
            )

            if assigned_task is not None:
                target = assigned_task.delivery_position

                dx = target[0] - robot.position[0]
                dy = target[1] - robot.position[1]

                bearing = (
                    0 if dx == 0 else (1 if dx > 0 else -1),
                    0 if dy == 0 else (1 if dy > 0 else -1)
                )

                task_id = assigned_task.task_id

            else:
                bearing = (0, 0)
                task_id = None

            observation = RobotObservation(
                robot_id=robot.robot_id,
                occupancy_grid=[
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                nearest_task_bearing=bearing,
                current_task_id=task_id
            )

            observations.append(observation)

        return observations

    def get_action_space(self):
        """Return the four available movement actions."""

        return self.ACTIONS.copy()

    def step(self, actions):
        """
        Perform one environment step.

        Args:
            actions:
                Dictionary mapping robot IDs to action IDs.

        Returns:
            observations, rewards, terminated, global_state
        """

        if len(actions) != self.num_robots:
            raise ValueError(
                "An action must be provided for every robot."
            )

        for robot_id, action in actions.items():

            if robot_id < 0 or robot_id >= self.num_robots:
                raise ValueError(
                    f"Invalid robot ID: {robot_id}"
                )

            if action not in self.ACTIONS:
                raise ValueError(
                    f"Invalid action: {action}"
                )

        self.episode_manager.step()

        rewards = {
            robot.robot_id: self.reward_function.calculate(
                previous_distance=0,
                current_distance=0
            )
            for robot in self.robots
        }

        tasks_remaining = sum(
            1 for task in self.tasks
            if not task.completed
        )

        terminated = self.episode_manager.is_terminated(
            tasks_remaining
        )

        return (
            self.get_observations(),
            rewards,
            terminated,
            self.get_global_state()
        )