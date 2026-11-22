import pytest
import torch

from src.marl.qmix_trainer import QMIXTrainer
from python_infra.generate_checkpoint_inventory import audit_qmix_checkpoint, sha256_file


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


def test_warm_start_loads_snapshot_weights_without_training_state(tmp_path):
    source = QMIXTrainer(num_agents=5, episodes=1, seed=42)
    payload = source._get_checkpoint_payload(600, 0.43)
    payload.pop("replay_buffer")
    payload["update_count"] = 900
    checkpoint = tmp_path / "warm_start.pt"
    torch.save(payload, checkpoint)

    fresh = QMIXTrainer(num_agents=5, episodes=20, epsilon_start=0.2, seed=43)
    with pytest.raises(ValueError, match="no replay buffer"):
        fresh.load_checkpoint(str(checkpoint))
    fresh.load_checkpoint(str(checkpoint), warm_start=True)

    for online, target, key in (
        (fresh.learner.agent_network, fresh.learner.target_agent_network, "agent_network"),
        (fresh.learner.mixer, fresh.learner.target_mixer, "mixer"),
    ):
        for name, expected in payload[key].items():
            assert torch.equal(online.state_dict()[name], expected)
            assert torch.equal(target.state_dict()[name], expected)
    assert len(fresh.replay_buffer) == 0
    assert fresh.learner.update_count == 0
    assert fresh.episode_offset == 0
    assert fresh._epsilon_for_episode(0) == pytest.approx(0.2)


def test_inventory_counts_only_the_evaluated_checkpoint_path(tmp_path):
    trainer = QMIXTrainer(num_agents=5, episodes=1, seed=42)
    payload = trainer._get_checkpoint_payload(100, 0.5)
    payload["update_count"] = 200
    selected = tmp_path / "snapshot.pt"
    final = tmp_path / "qmix.pt"
    torch.save(payload, selected)
    final.write_bytes(selected.read_bytes())
    evaluated = {(selected.name, frozenset({sha256_file(selected)}))}

    assert audit_qmix_checkpoint(selected, evaluated, tmp_path)["status"] == "evaluated"
    assert audit_qmix_checkpoint(final, evaluated, tmp_path)["status"] == "trained but unevaluated"
