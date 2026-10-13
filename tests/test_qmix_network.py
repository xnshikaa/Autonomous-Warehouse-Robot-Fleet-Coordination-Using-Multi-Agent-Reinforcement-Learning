import torch

from src.marl.qmix_network import AgentQNetwork, QMIXMixer


def test_agent_q_network():
    network = AgentQNetwork(
        observation_dim=50,
        action_dim=4,
        hidden_dim=64,
    )

    observations = torch.randn(2, 5, 50)

    q_values = network(observations)

    assert q_values.shape == (2, 5, 4)
    assert torch.isfinite(q_values).all()


def test_qmix_mixer():
    mixer = QMIXMixer(
        num_agents=5,
        state_dim=250,
        mixing_hidden_dim=32,
    )

    agent_q_values = torch.randn(2, 5)
    global_state = torch.randn(2, 250)

    q_total = mixer(
        agent_q_values,
        global_state,
    )

    assert q_total.shape == (2, 1)
    assert torch.isfinite(q_total).all()


def test_qmix_end_to_end():
    agent_network = AgentQNetwork(
        observation_dim=50,
        action_dim=4,
        hidden_dim=64,
    )

    mixer = QMIXMixer(
        num_agents=5,
        state_dim=250,
        mixing_hidden_dim=32,
    )

    observations = torch.randn(2, 5, 50)
    global_state = torch.randn(2, 250)

    q_values = agent_network(observations)

    selected_q_values = q_values.max(
        dim=2
    ).values

    q_total = mixer(
        selected_q_values,
        global_state,
    )

    assert q_values.shape == (2, 5, 4)
    assert selected_q_values.shape == (2, 5)
    assert q_total.shape == (2, 1)