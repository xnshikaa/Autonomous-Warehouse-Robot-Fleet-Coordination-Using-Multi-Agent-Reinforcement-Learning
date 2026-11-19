import pytest
import torch

from src.marl.qmix_trainer import QMIXTrainer


def test_qmix_file_checkpoint_resume_restores_replay_and_episode_schedule(tmp_path):
    original = QMIXTrainer(
        num_agents=10, episodes=1000, seed=650010,
        learning_rate=1e-4, target_update_interval=200,
    )
    checkpoint = tmp_path / "qmix_resume.pt"
    torch.save(original._get_checkpoint_payload(1000, 0.05), checkpoint)

    resumed = QMIXTrainer(
        num_agents=10, episodes=2000, seed=650010,
        learning_rate=1e-4, target_update_interval=200,
    )
    restored = resumed.load_checkpoint(str(checkpoint))

    assert restored["source"] == str(checkpoint)
    assert resumed.episode_offset == 1000
    assert resumed.resume_epsilon == 0.05
    assert resumed._epsilon_for_episode(1000) == pytest.approx(0.05)
    assert resumed.learner.update_count == original.learner.update_count


def test_qmix_resume_rejects_incompatible_fleet_and_reward_configuration(tmp_path):
    source = QMIXTrainer(
        num_agents=10, episodes=1, seed=650010,
        learning_rate=1e-4, target_update_interval=200,
    )
    checkpoint = tmp_path / "qmix_resume.pt"
    torch.save(source._get_checkpoint_payload(1000, 0.05), checkpoint)

    different_fleet = QMIXTrainer(num_agents=5, episodes=1, seed=650010)
    with pytest.raises(ValueError, match="incompatible"):
        different_fleet.load_checkpoint(str(checkpoint))

    different_reward = QMIXTrainer(
        num_agents=10, episodes=1, seed=650010,
        learning_rate=1e-4, target_update_interval=200,
        reward_overrides={"collision_penalty": 0.0},
    )
    with pytest.raises(ValueError, match="reward overrides"):
        different_reward.load_checkpoint(str(checkpoint))
