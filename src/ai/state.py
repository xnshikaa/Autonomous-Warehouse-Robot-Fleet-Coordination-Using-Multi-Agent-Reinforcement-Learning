from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RobotState:
    """State information for one robot."""

    robot_id: int
    position: Tuple[int, int]
    last_requested_action: int | None = None
    last_executed_action: int | None = None
    action_overridden: bool = False
    blocked_reason: str | None = None


@dataclass
class TaskState:
    """State information for one warehouse task."""

    task_id: int
    pickup_position: Tuple[int, int]
    delivery_position: Tuple[int, int]
    assigned_robot: int | None = None
    completed: bool = False


@dataclass
class GlobalState:
    """
    Global state used during centralized training.

    Contains all robot positions and the complete task queue.
    """

    robots: List[RobotState]
    tasks: List[TaskState]


@dataclass
class RobotObservation:
    """
    Local observation available to an individual robot.

    Contains:
    - local occupancy grid
    - nearest task bearing
    - current task assignment
    """

    robot_id: int
    occupancy_grid: List[List[int]]
    nearest_task_bearing: Tuple[int, int]
    current_task_id: int | None
