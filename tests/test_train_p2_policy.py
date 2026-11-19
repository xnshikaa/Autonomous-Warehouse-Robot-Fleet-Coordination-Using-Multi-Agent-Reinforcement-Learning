import json

import torch

from src.marl.train_p2_policy import _save_checkpoint_snapshot


class _FakeQMIXTrainer:
    def _get_checkpoint_payload(self, episode, epsilon):
        return {"episode": episode, "epsilon": epsilon, "agent_network": {}, "replay_buffer": {"large": True}}


class _FakeIQLTrainer:
    def save_agents(self, directory):
        from pathlib import Path

        path = Path(directory)
        (path / "iql_robot_0.pt").write_bytes(b"weights")
        (path / "iql_manifest.json").write_text(json.dumps({"observation_dim": 456}))


def test_qmix_snapshot_is_inference_loadable_and_omits_replay_memory(tmp_path):
    snapshot = _save_checkpoint_snapshot(
        "QMIX", _FakeQMIXTrainer(), {"episode": 100, "epsilon": 0.2},
        tmp_path / "snapshots", checkpoint_interval=100, episodes=500,
    )
    payload = torch.load(snapshot / "checkpoint_state.pt", map_location="cpu", weights_only=False)
    assert payload["episode"] == 100
    assert "agent_network" in payload
    assert "replay_buffer" not in payload


def test_iql_snapshot_contains_weights_and_encoder_manifest(tmp_path):
    snapshot = _save_checkpoint_snapshot(
        "IQL", _FakeIQLTrainer(), {"episode": 500, "epsilon": 0.1},
        tmp_path / "snapshots", checkpoint_interval=100, episodes=500,
    )
    assert (snapshot / "iql_robot_0.pt").read_bytes() == b"weights"
    assert json.loads((snapshot / "iql_manifest.json").read_text())["observation_dim"] == 456


def test_checkpoint_snapshot_skips_non_interval_episode(tmp_path):
    assert _save_checkpoint_snapshot(
        "QMIX", _FakeQMIXTrainer(), {"episode": 101, "epsilon": 0.2},
        tmp_path / "snapshots", checkpoint_interval=100, episodes=500,
    ) is None
