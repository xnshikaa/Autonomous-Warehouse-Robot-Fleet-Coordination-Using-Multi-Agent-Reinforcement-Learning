import math


class ObservationEncoder:
    """
    Converts warehouse robot state into the project's
    standard 50-dimensional QMIX observation vector.

    Layout:
        0-44  : 3x3 occupancy grid, 5-category one-hot
        45-47 : normalized task delta row, delta column, distance
        48    : has_task
        49    : carrying_item
    """

    EMPTY_FLOOR = 0
    WALL_OR_BOUNDARY = 1
    SHELF = 2
    ROBOT = 3
    HUMAN_CORRIDOR = 4

    def __init__(self, grid_width=20, grid_height=20):
        self.grid_width = grid_width
        self.grid_height = grid_height

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
            grid_width, grid_height = layout.dimensions
            shelves = set(layout.obstacles)
            corridor_cells = set(layout.human_corridors)
        else:
            grid_width, grid_height = self.grid_width, self.grid_height
            shelves = set(shelves or [])
            corridor_cells = {(x, y) for x in range(8, 11) for y in range(8, 11)}

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
                    or cx >= grid_width
                    or cy < 0
                    or cy >= grid_height
                ):
                    category = self.WALL_OR_BOUNDARY

                elif (cx, cy) in shelves:
                    category = self.SHELF

                elif (cx, cy) in robot_positions:
                    category = self.ROBOT

                elif (cx, cy) in corridor_cells:
                    category = self.HUMAN_CORRIDOR

                one_hot = [0.0] * 5
                one_hot[category] = 1.0

                vector.extend(one_hot)

        # 2. Task bearing and distance = 3 values
        norm_delta_row = 0.0
        norm_delta_col = 0.0
        norm_distance = 0.0

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

            norm_delta_row = (
                target_y - ry
            ) / max(1, grid_height - 1)

            norm_delta_col = (
                target_x - rx
            ) / max(1, grid_width - 1)

            norm_distance = math.sqrt(
                norm_delta_row ** 2 +
                norm_delta_col ** 2
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

        if len(vector) != 50:
            raise RuntimeError(
                f"Observation must contain 50 values, got {len(vector)}."
            )

        return vector
