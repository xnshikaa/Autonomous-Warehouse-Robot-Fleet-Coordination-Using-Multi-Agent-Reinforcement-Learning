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