from types import SimpleNamespace
from collections import deque

from src.ai.baselines.rule_based import RuleBasedController
from src.ai.state import RobotState


class OpenGrid:
    ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}
    DELTAS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    def __init__(self, robot, task, obstacles=()):
        self.robots = [robot]
        self.task = task
        self.obstacles = set(obstacles)

    def _get_robot_task(self, robot_id):
        return self.task

    def get_target_cell(self, robot, action):
        dx, dy = self.DELTAS[action]
        return robot.position[0] + dx, robot.position[1] + dy

    @staticmethod
    def is_walkable(cell):
        return True

    def get_action_masks(self):
        return [[
            (0 <= self.get_target_cell(robot, action)[0] < 5
             and 0 <= self.get_target_cell(robot, action)[1] < 5
             and self.get_target_cell(robot, action) not in self.obstacles)
            for action in self.ACTIONS
        ] for robot in self.robots]

    def _path_distance(self, start, goal):
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            cell, distance = queue.popleft()
            if cell == goal:
                return distance
            for dx, dy in self.DELTAS.values():
                neighbor = cell[0] + dx, cell[1] + dy
                if (0 <= neighbor[0] < 5 and 0 <= neighbor[1] < 5
                        and neighbor not in self.obstacles and neighbor not in seen):
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return 100


def test_rule_based_controller_targets_delivery_after_pickup():
    robot = RobotState(robot_id=0, position=(2, 1), carrying_item=True)
    task = SimpleNamespace(
        pickup_position=(2, 0),
        delivery_position=(2, 4),
    )
    controller = RuleBasedController(OpenGrid(robot, task))

    assert controller.get_actions() == {0: 1}


def test_rule_based_controller_routes_around_obstacle():
    robot = RobotState(robot_id=0, position=(1, 1))
    task = SimpleNamespace(
        pickup_position=(3, 1),
        delivery_position=(4, 4),
    )
    environment = OpenGrid(robot, task, obstacles={(2, 1)})
    controller = RuleBasedController(environment)

    action = controller.get_actions()[0]

    assert action != 3
    assert environment.get_action_masks()[0][action]
