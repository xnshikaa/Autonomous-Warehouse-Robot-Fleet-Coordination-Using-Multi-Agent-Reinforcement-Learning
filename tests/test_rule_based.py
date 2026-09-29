from src.ai.environment import WarehouseEnvironment
from src.ai.baselines.rule_based import RuleBasedController


def create_environment_with_tasks(
    num_robots=5,
    max_steps=100,
):
    """
    Create a deterministic warehouse environment
    with one task assigned to each robot.
    """

    environment = WarehouseEnvironment(
        num_robots=num_robots,
        max_steps=max_steps,
    )

    environment.reset()

    for robot in environment.robots:

        robot_id = robot.robot_id

        environment.add_task(
            task_id=robot_id,
            pickup_position=(
                2 + robot_id,
                robot_id,
            ),
            delivery_position=(
                5 + robot_id,
                robot_id,
            ),
            assigned_robot=robot_id,
        )

    return environment


def test_rule_based_controller_creation():
    environment = WarehouseEnvironment(
        num_robots=5,
        max_steps=100,
    )

    controller = RuleBasedController(
        environment
    )

    assert controller.environment is environment


def test_rule_based_generates_action_for_every_robot():
    environment = create_environment_with_tasks()

    controller = RuleBasedController(
        environment
    )

    actions = controller.get_actions()

    assert len(actions) == environment.num_robots

    for robot in environment.robots:
        assert robot.robot_id in actions


def test_rule_based_actions_are_valid():
    environment = create_environment_with_tasks()

    controller = RuleBasedController(
        environment
    )

    actions = controller.get_actions()

    valid_actions = environment.get_action_space()

    for action in actions.values():
        assert action in valid_actions


def test_rule_based_moves_robot_toward_pickup():
    environment = WarehouseEnvironment(
        num_robots=1,
        max_steps=100,
    )

    environment.reset()

    environment.add_task(
        task_id=1,
        pickup_position=(2, 0),
        delivery_position=(5, 0),
        assigned_robot=0,
    )

    controller = RuleBasedController(
        environment
    )

    robot = environment.robots[0]

    distance_before = (
        environment._manhattan_distance(
            robot.position,
            (2, 0),
        )
    )

    actions = controller.get_actions()

    environment.step(actions)

    distance_after = (
        environment._manhattan_distance(
            robot.position,
            (2, 0),
        )
    )

    assert distance_after < distance_before


def test_rule_based_environment_step():
    environment = create_environment_with_tasks()

    controller = RuleBasedController(
        environment
    )

    actions = controller.get_actions()

    observations, rewards, terminated, state = (
        environment.step(actions)
    )

    assert len(observations) == environment.num_robots
    assert len(rewards) == environment.num_robots
    assert len(state.robots) == environment.num_robots
    assert isinstance(terminated, bool)


def test_rule_based_completes_simple_task():
    environment = WarehouseEnvironment(
        num_robots=1,
        max_steps=20,
    )

    environment.reset()

    environment.add_task(
        task_id=1,
        pickup_position=(2, 0),
        delivery_position=(5, 0),
        assigned_robot=0,
    )

    controller = RuleBasedController(
        environment
    )

    terminated = False

    while not terminated:

        actions = controller.get_actions()

        (
            observations,
            rewards,
            terminated,
            state,
        ) = environment.step(actions)

    assert environment.tasks[0].completed is True


def test_rule_based_is_deterministic():
    environment_one = create_environment_with_tasks(
        num_robots=1
    )

    environment_two = create_environment_with_tasks(
        num_robots=1
    )

    controller_one = RuleBasedController(
        environment_one
    )

    controller_two = RuleBasedController(
        environment_two
    )

    actions_one = controller_one.get_actions()
    actions_two = controller_two.get_actions()

    assert actions_one == actions_two