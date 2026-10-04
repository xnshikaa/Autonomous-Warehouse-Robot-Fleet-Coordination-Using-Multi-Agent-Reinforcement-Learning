from src.marl.qmix_trainer import QMIXTrainer


def test_qmix_trainer_smoke():
    trainer = QMIXTrainer(
        episodes=2,
        max_steps=5,
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

    import numpy as np

    observation = np.zeros(
        (5, 50),
        dtype=np.float32,
    )

    state = trainer._build_global_state(
        observation
    )

    assert state.shape == (250,)
    assert state.dtype == np.float32