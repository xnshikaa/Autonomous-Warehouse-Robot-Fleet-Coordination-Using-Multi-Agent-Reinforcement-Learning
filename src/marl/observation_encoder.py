import math
from collections import deque
from functools import lru_cache


class ObservationEncoder:
    """
    Converts warehouse robot state into a per-robot policy observation.

    Layout:
        0-44  : 3x3 occupancy grid, 5-category one-hot
        45-47 : normalized task delta row, delta column, distance
        48    : has_task
        49    : carrying_item
        50-51 : normalized absolute robot position (optional)
        52+   : static map, row-major, 0 free / 0.5 worker zone / 1 shelf (optional)
        final 4: remaining path distance after each action (optional)
    """

    EMPTY_FLOOR = 0
    WALL_OR_BOUNDARY = 1
    SHELF = 2
    ROBOT = 3
    HUMAN_CORRIDOR = 4

    def __init__(self, grid_width=20, grid_height=20, include_static_map=False,
                 include_path_costs=False):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.include_static_map = include_static_map
        self.include_path_costs = include_path_costs

    @property
    def observation_dim(self):
        return (
            50
            + (2 + self.grid_width * self.grid_height if self.include_static_map else 0)
            + (4 if self.include_path_costs else 0)
        )

    @staticmethod
    @lru_cache(maxsize=512)
    def _distance_map(grid_width, grid_height, target, blocked):
        distances = {target: 0}
        queue = deque([target])
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
                nx, ny = neighbor
                if (0 <= nx < grid_width and 0 <= ny < grid_height
                        and neighbor not in blocked and neighbor not in distances):
                    distances[neighbor] = distances[(x, y)] + 1
                    queue.append(neighbor)
        return distances

    def _path_costs(self, robot, target, layout):
        """Return normalized shortest-path distances after UP/DOWN/LEFT/RIGHT."""
        if target is None or layout is None:
            return [1.0] * 4
        blocked = frozenset(layout.obstacles | layout.human_corridors)
        distances = self._distance_map(
            self.grid_width, self.grid_height, target, blocked
        )

        x, y = robot.position
        action_targets = ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))
        scale = max(1, self.grid_width + self.grid_height)
        return [
            min(1.0, distances[cell] / scale) if cell in distances else 1.0
            for cell in action_targets
        ]

    def encode(self, robot, robots, tasks, shelves=None, layout=None):
        """
        Encode one robot's observation.

        Args:
            robot: RobotState
            robots: list of RobotState
            tasks: list of TaskState
            shelves: optional iterable of (x, y) coordinates

        Returns:
            List containing exactly 50 float values.
        """

        if layout is not None:
            shelves = getattr(layout, "obstacles", ())
            human_corridors = set(getattr(layout, "human_corridors", ()))
        else:
            human_corridors = set()
        shelves = set(shelves or [])

        robot_positions = {
            other_robot.position
            for other_robot in robots
        }

        vector = []

        rx, ry = robot.position

        # 1. 3x3 occupancy window = 45 values
        for dy in range(-1, 2):
            for dx in range(-1, 2):

                cx = rx + dx
                cy = ry + dy

                category = self.EMPTY_FLOOR

                # Own center cell is represented as empty floor.
                if dx == 0 and dy == 0:
                    category = self.EMPTY_FLOOR

                elif (
                    cx < 0
                    or cx >= self.grid_width
                    or cy < 0
                    or cy >= self.grid_height
                ):
                    category = self.WALL_OR_BOUNDARY

                elif (cx, cy) in shelves:
                    category = self.SHELF

                elif (cx, cy) in robot_positions:
                    category = self.ROBOT

                elif (cx, cy) in human_corridors:
                    category = self.HUMAN_CORRIDOR

                one_hot = [0.0] * 5
                one_hot[category] = 1.0

                vector.extend(one_hot)

        # 2. Task bearing and distance = 3 values
        norm_delta_row = 0.0
        norm_delta_col = 0.0
        norm_distance = 0.0
        target = None

        assigned_task = next(
            (
                task
                for task in tasks
                if task.assigned_robot == robot.robot_id
                and not task.completed
            ),
            None
        )

        if assigned_task is not None:

            carrying_item = getattr(robot, "carrying_item", False) or (robot.position == assigned_task.pickup_position)

            if carrying_item:
                target_x, target_y = assigned_task.delivery_position
            else:
                target_x, target_y = assigned_task.pickup_position
            target = (target_x, target_y)

            delta_row = target_y - ry
            delta_col = target_x - rx
            distance = math.hypot(delta_row, delta_col)
            if distance > 0:
                # Keep direction features near unit scale so the small
                # normalized deltas do not get drowned out by binary occupancy.
                norm_delta_row = delta_row / distance
                norm_delta_col = delta_col / distance
            norm_distance = distance / max(
                1.0,
                math.hypot(self.grid_height - 1, self.grid_width - 1),
            )

        vector.extend([
            float(norm_delta_row),
            float(norm_delta_col),
            float(norm_distance)
        ])

        # 3. Task status flags = 2 values
        has_task = 1.0 if assigned_task is not None else 0.0
        carrying_item_val = 1.0 if (assigned_task is not None and carrying_item) else 0.0

        vector.extend([
            has_task,
            carrying_item_val
        ])

        if self.include_static_map:
            vector.extend((
                rx / max(1, self.grid_width - 1),
                ry / max(1, self.grid_height - 1),
            ))
            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    cell = (x, y)
                    if cell in shelves:
                        vector.append(1.0)
                    elif cell in human_corridors:
                        vector.append(0.5)
                    else:
                        vector.append(0.0)

        if self.include_path_costs:
            vector.extend(self._path_costs(robot, target if assigned_task else None, layout))

        if len(vector) != self.observation_dim:
            raise RuntimeError(
                f"Observation must contain {self.observation_dim} values, got {len(vector)}."
            )

        return vector
