import math

from src.ai.state import RobotState, TaskState
from src.marl.observation_encoder import ObservationEncoder


def test_observation_has_50_dimensions():
    encoder = ObservationEncoder(
        grid_width=20,
        grid_height=20
    )

    robot = RobotState(
        robot_id=0,
        position=(5, 5)
    )

    observation = encoder.encode(
        robot=robot,
        robots=[robot],
        tasks=[]
    )

    assert len(observation) == 50
    assert all(isinstance(value, float) for value in observation)


def test_observation_with_task():
    encoder = ObservationEncoder(
        grid_width=20,
        grid_height=20
    )

    robot = RobotState(
        robot_id=0,
        position=(5, 5)
    )

    task = TaskState(
        task_id=1,
        pickup_position=(10, 5),
        delivery_position=(15, 15),
        assigned_robot=0
    )

    observation = encoder.encode(
        robot=robot,
        robots=[robot],
        tasks=[task]
    )

    assert len(observation) == 50

    # has_task is index 48
    assert observation[48] == 1.0

    # carrying_item is index 49
    assert observation[49] == 0.0


def test_task_bearing_is_unit_direction_and_distance_is_separately_scaled():
    encoder = ObservationEncoder(grid_width=20, grid_height=20)
    robot = RobotState(robot_id=0, position=(5, 5))
    task = TaskState(
        task_id=1,
        pickup_position=(15, 5),
        delivery_position=(15, 15),
        assigned_robot=0,
    )

    observation = encoder.encode(robot, [robot], [task])

    assert observation[45:48] == [0.0, 1.0, 10 / math.hypot(19, 19)]


def test_occupancy_grid_is_one_hot():
    encoder = ObservationEncoder(
        grid_width=20,
        grid_height=20
    )

    robot = RobotState(
        robot_id=0,
        position=(5, 5)
    )

    observation = encoder.encode(
        robot=robot,
        robots=[robot],
        tasks=[]
    )

    occupancy = observation[:45]

    # 9 cells × 5 categories.
    for index in range(0, 45, 5):
        cell = occupancy[index:index + 5]

        assert sum(cell) == 1.0
        assert all(value in (0.0, 1.0) for value in cell)


def test_map_aware_observation_includes_pose_and_static_warehouse_map():
    from src.ai.environment import WarehouseEnvironment

    environment = WarehouseEnvironment(num_robots=1)
    environment.reset()
    layout = environment.get_active_layout()
    encoder = ObservationEncoder(
        layout.width, layout.height, include_static_map=True
    )
    robot = environment.robots[0]
    observation = encoder.encode(robot, environment.robots, [], layout=layout)

    assert len(observation) == 452
    assert observation[50:52] == [0.0, 0.0]
    map_offset = 52
    shelf_x, shelf_y = next(iter(layout.obstacles))
    corridor_x, corridor_y = next(iter(layout.human_corridors))
    assert observation[map_offset + shelf_y * layout.width + shelf_x] == 1.0
    assert observation[map_offset + corridor_y * layout.width + corridor_x] == 0.5


def test_path_cost_features_prefer_shortest_safe_route():
    from src.warehouse.layouts import WarehouseLayout

    layout = WarehouseLayout(
        layout_id="path-cost-test",
        name="Path cost test",
        width=5,
        height=5,
        obstacles=frozenset({(2, 1)}),
        robot_starts=((1, 2),),
        task_locations=(),
        human_corridors=frozenset(),
    )
    robot = RobotState(robot_id=0, position=(1, 2))
    task = TaskState(1, (3, 2), (4, 4), assigned_robot=0)
    encoder = ObservationEncoder(5, 5, include_static_map=True, include_path_costs=True)

    observation = encoder.encode(robot, [robot], [task], layout=layout)

    assert len(observation) == 81
    costs = observation[-4:]  # UP, DOWN, LEFT, RIGHT.
    assert costs[3] < costs[2]
