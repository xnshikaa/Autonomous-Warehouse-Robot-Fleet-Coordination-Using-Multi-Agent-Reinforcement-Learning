import random

from src.ai.environment import WarehouseEnvironment
from src.warehouse.task_generation import generate_task_assignments


def test_generated_tasks_are_seeded_safe_unique_and_reachable():
    environment = WarehouseEnvironment(num_robots=5)
    environment.reset()
    first = generate_task_assignments(environment, random.Random(123))
    second = generate_task_assignments(environment, random.Random(123))

    assert first == second
    assert len(first) == 5
    layout = environment.get_active_layout()
    task_cells = []
    for task in first:
        for cell in (task.pickup_position, task.delivery_position):
            assert layout.is_walkable(cell)
            assert cell not in layout.human_corridors
            task_cells.append(cell)
        start = environment.robots[task.assigned_robot].position
        environment._path_distance(start, task.pickup_position)
        environment._path_distance(task.pickup_position, task.delivery_position)
        assert start in environment._distance_maps[task.pickup_position]
        assert task.pickup_position in environment._distance_maps[task.delivery_position]
    assert len(task_cells) == len(set(task_cells))


def test_different_seeds_generate_different_task_coordinates():
    environment = WarehouseEnvironment(num_robots=5)
    environment.reset()
    first = generate_task_assignments(environment, random.Random(123))
    second = generate_task_assignments(environment, random.Random(456))

    assert [(t.pickup_position, t.delivery_position) for t in first] != [
        (t.pickup_position, t.delivery_position) for t in second
    ]
