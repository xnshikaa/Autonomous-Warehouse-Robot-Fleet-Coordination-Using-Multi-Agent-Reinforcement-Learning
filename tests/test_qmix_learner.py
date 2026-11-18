import torch

from src.marl.qmix_learner import QMIXLearner


def test_qmix_action_selection():
    learner = QMIXLearner(
        num_agents=5,
        observation_dim=50,
        action_dim=4,
        state_dim=250,
    )

    observations = torch.randn(5, 50)

    actions = learner.select_actions(
        observations,
        epsilon=0.0,
    )

    assert actions.shape == (5,)
    assert torch.all(
        (actions >= 0)
        & (actions < 4)
    )


def test_qmix_action_selection_respects_legal_action_masks():
    learner = QMIXLearner(
        num_agents=2, observation_dim=50, action_dim=4, state_dim=22
    )
    observations = torch.zeros(2, 50)
    masks = torch.tensor([[False, True, False, False], [False, False, True, False]])

    for epsilon in (0.0, 1.0):
        actions = learner.select_actions(
            observations, epsilon=epsilon, action_masks=masks
        )
        assert actions.tolist() == [1, 2]


def test_qmix_training_update():
    learner = QMIXLearner(
        num_agents=5,
        observation_dim=50,
        action_dim=4,
        state_dim=250,
    )

    observations = torch.randn(2, 5, 50)
    next_observations = torch.randn(
        2, 5, 50
    )

    actions = torch.randint(
        0,
        4,
        (2, 5),
    )

    rewards = torch.tensor(
        [1.0, 0.5]
    )

    global_states = torch.randn(
        2,
        250,
    )

    next_global_states = torch.randn(
        2,
        250,
    )

    terminated = torch.tensor(
        [0.0, 1.0]
    )

    result = learner.train_batch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        global_states=global_states,
        next_global_states=next_global_states,
        terminated=terminated,
    )

    assert "loss" in result
    assert "q_total_mean" in result
    assert "target_mean" in result

    assert result["loss"] >= 0.0


def test_qmix_huber_loss_limits_large_td_residuals():
    learner = QMIXLearner(
        num_agents=1,
        observation_dim=50,
        action_dim=4,
        state_dim=50,
    )
    observations = torch.zeros(1, 1, 50)
    global_states = torch.zeros(1, 50)

    result = learner.train_batch(
        observations=observations,
        actions=torch.zeros(1, 1, dtype=torch.long),
        rewards=torch.tensor([1_000_000.0]),
        next_observations=observations,
        global_states=global_states,
        next_global_states=global_states,
        terminated=torch.ones(1),
    )

    assert result["loss"] < 1_100_000.0

    assert torch.isfinite(
        torch.tensor(result["loss"])
    )


def test_target_network_update():
    learner = QMIXLearner(
        num_agents=5,
        observation_dim=50,
        action_dim=4,
        state_dim=250,
    )

    learner.update_target_networks()

    for parameter, target_parameter in zip(
        learner.agent_network.parameters(),
        learner.target_agent_network.parameters(),
    ):
        assert torch.equal(
            parameter,
            target_parameter,
        )

    for parameter, target_parameter in zip(
        learner.mixer.parameters(),
        learner.target_mixer.parameters(),
    ):
        assert torch.equal(
            parameter,
            target_parameter,
        )
