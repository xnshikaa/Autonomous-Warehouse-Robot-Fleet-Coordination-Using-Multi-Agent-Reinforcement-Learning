import pytest
import json

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

    def capture(observations, action_masks=None):
        captured.append(observations.copy())
        assert action_masks is not None
        return select_actions(observations, action_masks)

    trainer._select_actions = capture
    trainer.train()

    assert captured[0].shape == (5, 50)
    assert (captured[0][:, 48] == 1.0).all()
    assert len(trainer.env.adapter.environment.tasks) == 5


def test_iql_can_train_with_qmix_observation_and_action_mask_parity():
    trainer = IQLTrainer(
        episodes=1,
        max_steps=2,
        seed=21,
        include_static_map=True,
        include_path_costs=True,
    )
    assert trainer.observation_dim == 456
    history = trainer.train()
    assert len(history) == 1
    assert all(learner.observation_dim == 456 for learner in trainer.learners)


def test_iql_checkpoint_saves_observation_and_seed_metadata(tmp_path):
    trainer = IQLTrainer(
        episodes=1, max_steps=1, seed=123,
        include_static_map=True, include_path_costs=True,
    )
    trainer.train()
    trainer.save_agents(str(tmp_path))
    metadata = json.loads((tmp_path / "iql_manifest.json").read_text())
    assert metadata["seed"] == 123
    assert metadata["observation_dim"] == 456
    assert metadata["include_path_costs"] is True


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
