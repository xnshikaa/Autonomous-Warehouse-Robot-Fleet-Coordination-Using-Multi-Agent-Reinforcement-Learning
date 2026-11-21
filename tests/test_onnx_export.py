"""Unit tests for ONNX export and PyTorch-to-ONNX Runtime parity."""

from pathlib import Path
import pytest
import torch

from src.marl.export_onnx import export_agent_to_onnx, verify_onnx_parity
from src.marl.qmix_network import AgentQNetwork


def test_onnx_export_and_parity_from_module(tmp_path):
    model = AgentQNetwork(observation_dim=456, action_dim=4, hidden_dim=64)
    onnx_file = tmp_path / "agent.onnx"

    exported = export_agent_to_onnx(model, onnx_file, observation_dim=456, action_dim=4)
    assert exported.is_file()
    assert onnx_file.with_suffix(".json").is_file()

    parity = verify_onnx_parity(model, exported, atol=1e-5)
    assert parity["verified"] is True
    assert parity["max_absolute_difference"] < 1e-5
    assert set(parity["batches_tested"]) == {1, 5, 10, 20}


def test_onnx_export_from_checkpoint_dict(tmp_path):
    model = AgentQNetwork(observation_dim=456, action_dim=4, hidden_dim=64)
    ckpt = {
        "agent_network": model.state_dict(),
        "config": {"observation_dim": 456, "action_dim": 4},
    }
    onnx_file = tmp_path / "from_dict.onnx"

    exported = export_agent_to_onnx(ckpt, onnx_file)
    assert exported.is_file()

    parity = verify_onnx_parity(ckpt, exported, atol=1e-5)
    assert parity["verified"] is True
