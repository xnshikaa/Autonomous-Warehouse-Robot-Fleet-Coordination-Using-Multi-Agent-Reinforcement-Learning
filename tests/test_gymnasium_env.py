import numpy as np

from src.marl.gymnasium_env import WarehouseGymEnv


def test_gymnasium_reset():
    env = WarehouseGymEnv()

    observation, info = env.reset()

    assert observation.shape == (5, 50)
    assert observation.dtype == np.float32
    assert isinstance(info, dict)

    env.close()


def test_gymnasium_action_and_observation_spaces():
    env = WarehouseGymEnv()

    assert env.action_space.n == 4
    assert env.observation_space.shape == (50,)

    env.close()


def test_gymnasium_step():
    env = WarehouseGymEnv()

    observation, info = env.reset()

    env.adapter.environment.add_task(
        task_id=1,
        pickup_position=(1, 0),
        delivery_position=(2, 0),
        assigned_robot=0,
    )

    next_observation, reward, terminated, truncated, info = env.step(
        [3, 0, 0, 0, 0]
    )

    assert next_observation.shape == (5, 50)
    assert next_observation.dtype == np.float32
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False

    robot_0 = env.adapter.environment.robots[0]

    assert robot_0.position == (1, 0)

    env.close()