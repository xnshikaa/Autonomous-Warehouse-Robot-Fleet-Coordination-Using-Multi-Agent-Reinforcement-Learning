from src.marl.spec import MARLEnvironmentSpec


def test_default_marl_spec():
    spec = MARLEnvironmentSpec()

    assert spec.num_agents == 5
    assert spec.observation_dim == 50
    assert spec.action_dim == 4
    assert spec.max_steps == 100


def test_agent_ids():
    spec = MARLEnvironmentSpec(num_agents=5)

    assert spec.agent_ids == [0, 1, 2, 3, 4]


def test_action_ids():
    spec = MARLEnvironmentSpec()

    assert spec.action_ids == [0, 1, 2, 3]