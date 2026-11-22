"""Seeded one-robot missions using the simulator's real task queue."""
from random import Random

from src.ai.environment import WarehouseEnvironment
from src.warehouse.task_generation import generate_task_assignments

MISSION_SEED = 300001


def queue_mission(environment: WarehouseEnvironment, seed: int) -> None:
    if environment.num_robots != 1 or environment.tasks:
        raise ValueError("A single-robot mission requires an empty one-robot task queue.")
    task, = generate_task_assignments(environment, Random(seed))
    environment.add_task(task.task_id, task.pickup_position, task.delivery_position, task.assigned_robot)
