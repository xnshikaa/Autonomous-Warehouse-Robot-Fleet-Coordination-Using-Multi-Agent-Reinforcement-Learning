from src.ai.state import (
    RobotState,
    TaskState,
    GlobalState,
    RobotObservation
)


def test_robot_state():
    robot = RobotState(
        robot_id=1,
        position=(2, 3)
    )

    assert robot.robot_id == 1
    assert robot.position == (2, 3)


def test_task_state():
    task = TaskState(
        task_id=10,
        pickup_position=(1, 2),
        delivery_position=(5, 6)
    )

    assert task.task_id == 10
    assert task.pickup_position == (1, 2)
    assert task.delivery_position == (5, 6)
    assert task.assigned_robot is None
    assert task.completed is False


def test_global_state():
    robot = RobotState(
        robot_id=1,
        position=(2, 3)
    )

    task = TaskState(
        task_id=10,
        pickup_position=(1, 2),
        delivery_position=(5, 6)
    )

    state = GlobalState(
        robots=[robot],
        tasks=[task]
    )

    assert len(state.robots) == 1
    assert len(state.tasks) == 1
    assert state.robots[0].robot_id == 1
    assert state.tasks[0].task_id == 10


def test_robot_observation():
    observation = RobotObservation(
        robot_id=1,
        occupancy_grid=[
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 0]
        ],
        nearest_task_bearing=(1, 0),
        current_task_id=10
    )

    assert observation.robot_id == 1
    assert observation.occupancy_grid[1][1] == 1
    assert observation.nearest_task_bearing == (1, 0)
    assert observation.current_task_id == 10


def test_robot_without_task():
    observation = RobotObservation(
        robot_id=2,
        occupancy_grid=[
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        nearest_task_bearing=(0, 0),
        current_task_id=None
    )

    assert observation.current_task_id is None