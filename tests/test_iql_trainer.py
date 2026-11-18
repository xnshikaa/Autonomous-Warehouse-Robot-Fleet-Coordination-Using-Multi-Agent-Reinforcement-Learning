import pytest

from src.marl.iql_trainer import IQLTrainer


def test_iql_trainer_initialization():
    trainer = IQLTrainer(
        episodes=2,
        max_steps=5,
        seed=42,
    )

    assert trainer.num_robots == 5
    assert trainer.observation_dim == 50
    assert trainer.action_dim == 4
    assert len(trainer.learners) == 5
    assert trainer.epsilon == 1.0


def test_iql_trainer_configures_requested_fleet_and_horizon():
    trainer = IQLTrainer(num_robots=10, episodes=1, max_steps=3, seed=42)

    assert trainer.env.num_agents == 10
    assert trainer.env.adapter.environment.max_steps == 3
    assert len(trainer.learners) == 10


def test_iql_training_starts_with_seeded_task_observations():
    trainer = IQLTrainer(episodes=1, max_steps=1, seed=42)
    captured = []
    select_actions = trainer._select_actions

    def capture(observations):
        captured.append(observations.copy())
        return select_actions(observations)

    trainer._select_actions = capture
    trainer.train()

    assert captured[0].shape == (5, 50)
    assert (captured[0][:, 48] == 1.0).all()
    assert len(trainer.env.adapter.environment.tasks) == 5


def test_iql_training():
    trainer = IQLTrainer(
        episodes=2,
        max_steps=5,
        seed=42,
    )

    history = trainer.train()

    assert len(history) == 2

    for result in history:
        assert "episode" in result
        assert "steps" in result
        assert "epsilon" in result
        assert "reward" in result
        assert "loss" in result

        assert result["steps"] > 0
        assert isinstance(result["reward"], float)
        assert isinstance(result["loss"], float)


def test_iql_evaluation():
    trainer = IQLTrainer(
        episodes=1,
        max_steps=5,
        seed=42,
    )

    trainer.train()

    result = trainer.evaluate(
        episodes=2
    )

    assert "episodes" in result
    assert "average_reward" in result
    assert "max_reward" in result
    assert "min_reward" in result

    assert result["episodes"] == 2.0
