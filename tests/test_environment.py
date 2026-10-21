from src.ai.environment import WarehouseEnvironment


def test_environment_creation():
    environment = WarehouseEnvironment(
        num_robots=5,
        max_steps=100
    )

    assert environment.num_robots == 5
    assert environment.max_steps == 100


def test_environment_reset():
    environment = WarehouseEnvironment(
        num_robots=5,
        max_steps=100
    )

    global_state, observations = environment.reset()

    assert len(global_state.robots) == 5
    assert len(global_state.tasks) == 0
    assert len(observations) == 5


def test_action_space():
    environment = WarehouseEnvironment()

    actions = environment.get_action_space()

    assert actions[0] == "forward"
    assert actions[1] == "backward"
    assert actions[2] == "left"
    assert actions[3] == "right"


def test_add_task():
    environment = WarehouseEnvironment()

    environment.reset()

    environment.add_task(
        task_id=1,
        pickup_position=(2, 2),
        delivery_position=(5, 5),
        assigned_robot=0
    )

    assert len(environment.tasks) == 1
    assert environment.tasks[0].task_id == 1
    assert environment.tasks[0].assigned_robot == 0


def test_robot_observation_with_task():
    environment = WarehouseEnvironment()

    environment.reset()

    environment.add_task(
        task_id=1,
        pickup_position=(2, 2),
        delivery_position=(5, 5),
        assigned_robot=0
    )

    observations = environment.get_observations()

    robot_zero_observation = observations[0]

    assert robot_zero_observation.current_task_id == 1
    assert robot_zero_observation.nearest_task_bearing != (0, 0)


def test_environment_step():
    environment = WarehouseEnvironment(
        num_robots=5,
        max_steps=100
    )

    environment.reset()

    # Add an incomplete task so the episode continues
    environment.add_task(
        task_id=1,
        pickup_position=(2, 2),
        delivery_position=(5, 5),
        assigned_robot=0
    )

    actions = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }

    observations, rewards, terminated, state = environment.step(
        actions
    )

    assert len(observations) == 5
    assert len(rewards) == 5
    assert terminated is False
    assert len(state.robots) == 5


def test_invalid_number_of_actions():
    environment = WarehouseEnvironment(
        num_robots=5,
        max_steps=100
    )

    environment.reset()

    actions = {
        0: 0,
        1: 0
    }

    try:
        environment.step(actions)
        assert False
    except ValueError:
        assert True


def test_invalid_action():
    environment = WarehouseEnvironment(
        num_robots=5,
        max_steps=100
    )

    environment.reset()

    actions = {
        0: 99,
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }

    try:
        environment.step(actions)
        assert False
    except ValueError:
        assert True