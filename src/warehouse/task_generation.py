"""Seeded generation of reachable warehouse pickup/delivery tasks."""

from __future__ import annotations

from random import Random

from src.warehouse.layouts import LayoutTask


def generate_task_assignments(environment, rng: Random) -> list[LayoutTask]:
    """Sample distinct, safe, reachable tasks for the current robot fleet."""
    layout = environment.get_active_layout()
    starts = {robot.position for robot in environment.robots}
    safe_cells = set(layout.walkable_cells) - set(layout.human_corridors) - starts
    split_y = layout.height // 2
    pickup_cells = [cell for cell in safe_cells if cell[1] < split_y]
    delivery_cells = [cell for cell in safe_cells if cell[1] >= split_y]
    if not pickup_cells or not delivery_cells:
        raise ValueError(f"Layout {layout.layout_id!r} has no safe pickup or delivery cells.")

    used_cells: set[tuple[int, int]] = set()
    tasks: list[LayoutTask] = []
    max_path = layout.width + layout.height
    for robot in environment.robots:
        pickup_order = rng.sample(pickup_cells, len(pickup_cells))
        delivery_order = rng.sample(delivery_cells, len(delivery_cells))
        selected = None
        for pickup in pickup_order:
            if pickup in used_cells:
                continue
            environment._path_distance(robot.position, pickup)
            start_distances = environment._distance_maps[pickup]
            start_distance = start_distances.get(robot.position)
            if start_distance is None or start_distance < 2 or start_distance > max_path:
                continue

            for delivery in delivery_order:
                if delivery in used_cells or delivery == pickup:
                    continue
                environment._path_distance(pickup, delivery)
                delivery_distances = environment._distance_maps[delivery]
                delivery_distance = delivery_distances.get(pickup)
                if delivery_distance is not None and 6 <= delivery_distance <= max_path:
                    selected = pickup, delivery
                    break
            if selected:
                break

        if selected is None:
            raise ValueError(
                f"Could not generate a reachable task for robot {robot.robot_id} "
                f"in layout {layout.layout_id!r}."
            )
        pickup, delivery = selected
        used_cells.update((pickup, delivery))
        tasks.append(LayoutTask(len(tasks) + 1, pickup, delivery, robot.robot_id))

    return tasks
