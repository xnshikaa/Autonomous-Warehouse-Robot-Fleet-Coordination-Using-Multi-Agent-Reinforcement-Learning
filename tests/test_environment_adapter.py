from src.marl.environment_adapter import MARLEnvironmentAdapter


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