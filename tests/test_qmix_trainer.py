import numpy as np
import pytest
import torch

from src.marl.qmix_trainer import QMIXTrainer


def test_qmix_trainer_smoke():
    trainer = QMIXTrainer(
        episodes=2,
        max_steps=5,
        batch_size=4,
        replay_warmup_size=4,
        seed=42,
    )

    history = trainer.train()

    assert len(history) == 2

    for episode in history:
        assert "episode" in episode
        assert "steps" in episode
        assert "epsilon" in episode
        assert "reward" in episode
        assert "loss" in episode

        assert episode["steps"] > 0
        assert episode["loss"] >= 0.0

    assert len(trainer.replay_buffer) > 0


def test_qmix_trainer_reproducible_setup():
    trainer = QMIXTrainer(
        episodes=1,
        max_steps=2,
        seed=42,
    )

    assert trainer.seed == 42
    assert trainer.num_agents == 5
    assert trainer.observation_dim == 50
    assert trainer.action_dim == 4
    assert trainer.state_dim == 250


def test_global_state_dimension():
    trainer = QMIXTrainer(
        episodes=1,
        max_steps=1,
        seed=42,
    )

    observation = np.zeros(
        (5, 50),
        dtype=np.float32,
    )

    state = trainer._build_global_state(observation)

    assert state.shape == (250,)
    assert state.dtype == np.float32


def test_qmix_trainer_mlflow_and_checkpoint_integration(tmp_path):
    checkpoint_dir = str(tmp_path / "checkpoints")

    trainer = QMIXTrainer(
        num_agents=5,
        episodes=3,
        max_steps=10,
        batch_size=4,
        replay_warmup_size=4,
        seed=42,
        use_mlflow=True,
        experiment_name="test_qmix_trainer_experiment",
        save_checkpoints=True,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=2,
    )

    history = trainer.train()
    assert len(history) == 3
    assert trainer.run_id is not None
    assert trainer.checkpoint_manager is not None

    # Verify latest checkpoint exists and can be loaded
    checkpoint = trainer.load_checkpoint("latest")
    assert "agent_network" in checkpoint["checkpoint_data"]
    assert "mixer" in checkpoint["checkpoint_data"]
    assert checkpoint["metadata"]["step"] >= 2


def test_qmix_trainer_checkpoint_resume(tmp_path):
    checkpoint_dir = str(tmp_path / "resume_checkpoints")

    # Initial training run
    trainer1 = QMIXTrainer(
        num_agents=5,
        episodes=4,
        max_steps=10,
        batch_size=4,
        replay_warmup_size=4,
        seed=42,
        save_checkpoints=True,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=2,
    )
    trainer1.train()

    weights_before = trainer1.learner.agent_network.network[0].weight.clone()

    # Create new trainer instance and load checkpoint
    trainer2 = QMIXTrainer(
        num_agents=5,
        episodes=2,
        max_steps=10,
        batch_size=4,
        replay_warmup_size=4,
        seed=100,
        save_checkpoints=True,
        checkpoint_dir=checkpoint_dir,
    )
    trainer2.load_checkpoint("latest")

    weights_after = trainer2.learner.agent_network.network[0].weight.clone()
    assert torch.equal(weights_before, weights_after)

    # Resume training
    history2 = trainer2.train()
    assert len(history2) == 2