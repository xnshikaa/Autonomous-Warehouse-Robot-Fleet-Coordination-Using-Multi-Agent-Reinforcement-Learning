from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.ai.environment import WarehouseEnvironment


def test_adapter_reset():
    adapter = MARLEnvironmentAdapter(num_agents=5, max_steps=100)

    observations, global_state = adapter.reset()

    assert len(observations) == 5
    assert len(global_state.robots) == 5


def test_adapter_action_space():
    adapter = MARLEnvironmentAdapter(num_agents=5)

    action_space = adapter.get_action_space()

    assert len(action_space) == 4
    assert action_space[0] == "forward"
    assert action_space[1] == "backward"
    assert action_space[2] == "left"
    assert action_space[3] == "right"


def test_adapter_step():
    adapter = MARLEnvironmentAdapter(num_agents=5, max_steps=100)

    adapter.reset()

    actions = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }

    observations, rewards, terminated, global_state = adapter.step(actions)

    assert len(observations) == 5
    assert len(rewards) == 5
    assert isinstance(terminated, bool)
    assert len(global_state.robots) == 5


def test_adapter_can_wrap_the_live_environment_instance():
    environment = WarehouseEnvironment(num_robots=2)
    adapter = MARLEnvironmentAdapter(environment=environment)

    assert adapter.environment is environment
    assert adapter.get_num_agents() == 2
    observations, _ = adapter.reset()
    assert set(observations) == {0, 1}


def test_adapter_fleet_resize_stays_in_sync():
    adapter = MARLEnvironmentAdapter(num_agents=2)
    adapter.set_num_agents(3)
    assert adapter.get_num_agents() == 3
    assert len(adapter.environment.robots) == 3
