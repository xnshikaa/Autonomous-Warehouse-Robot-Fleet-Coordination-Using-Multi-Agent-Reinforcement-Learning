"""ONNX export and numerical parity verification for QMIX Agent Q-Networks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from src.marl.checkpoint_validator import (
    CHECKPOINT_SCHEMA_VERSION,
    extract_or_infer_metadata,
)
from src.marl.qmix_network import AgentQNetwork

logger = logging.getLogger(__name__)


def export_agent_to_onnx(
    model_or_checkpoint: Union[nn.Module, str, Path, Dict[str, Any]],
    output_path: str | Path,
    observation_dim: int = 456,
    action_dim: int = 4,
    hidden_dim: int = 64,
    opset_version: int = 14,
    metadata_props: Optional[Dict[str, str]] = None,
) -> Path:
    """Export an AgentQNetwork PyTorch model or checkpoint to ONNX format."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve and load PyTorch model weights
    if isinstance(model_or_checkpoint, nn.Module):
        agent_net = model_or_checkpoint
    else:
        if isinstance(model_or_checkpoint, (str, Path)):
            path = Path(model_or_checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
        else:
            payload = model_or_checkpoint

        agent_state = payload.get("agent_network") or payload.get("model_state_dict")
        if agent_state is None:
            raise ValueError("Payload missing 'agent_network' weights for ONNX export.")

        # Inspect dimensions from state dict
        obs_dim = agent_state["network.0.weight"].shape[1]
        act_dim = agent_state["network.4.weight"].shape[0]
        h_dim = agent_state["network.0.weight"].shape[0]

        agent_net = AgentQNetwork(observation_dim=obs_dim, action_dim=act_dim, hidden_dim=h_dim)
        agent_net.load_state_dict(agent_state, strict=True)

    agent_net.eval()

    # 2. Export using dynamic batch dimension
    dummy_input = torch.randn(1, observation_dim, dtype=torch.float32)

    torch.onnx.export(
        agent_net,
        dummy_input,
        str(out_path),
        input_names=["observation"],
        output_names=["q_values"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "q_values": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # 3. Embed metadata properties into ONNX model
    onnx_model = onnx.load(str(out_path))
    props = {
        "model_architecture": "AgentQNetwork",
        "observation_dim": str(observation_dim),
        "action_dim": str(action_dim),
        "schema_version": str(CHECKPOINT_SCHEMA_VERSION),
    }
    if metadata_props:
        props.update(metadata_props)

    for k, v in props.items():
        meta = onnx_model.metadata_props.add()
        meta.key = k
        meta.value = str(v)

    onnx.save(onnx_model, str(out_path))

    # Also save adjacent JSON manifest for quick reference
    manifest_path = out_path.with_suffix(".json")
    with open(manifest_path, "w") as f:
        json.dump(props, f, indent=2)

    logger.info(f"Exported ONNX model to {out_path} with observation_dim={observation_dim}")
    return out_path


def verify_onnx_parity(
    model_or_checkpoint: Union[nn.Module, str, Path, Dict[str, Any]],
    onnx_path: str | Path,
    num_samples: int = 50,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Verify numerical parity between PyTorch agent network and ONNX Runtime output."""
    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    # Load ONNX session
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    obs_dim = session.get_inputs()[0].shape[1]
    if isinstance(obs_dim, str) or obs_dim is None:
        obs_dim = 456

    # Load PyTorch model
    if isinstance(model_or_checkpoint, nn.Module):
        pytorch_model = model_or_checkpoint
    else:
        if isinstance(model_or_checkpoint, (str, Path)):
            payload = torch.load(model_or_checkpoint, map_location="cpu", weights_only=False)
        else:
            payload = model_or_checkpoint
        agent_state = payload.get("agent_network") or payload.get("model_state_dict")
        pytorch_model = AgentQNetwork(
            observation_dim=obs_dim,
            action_dim=4,
            hidden_dim=64,
        )
        pytorch_model.load_state_dict(agent_state, strict=True)
    pytorch_model.eval()

    # Generate test inputs across varying batch sizes
    test_batches = [1, 5, 10, 20]
    max_diff = 0.0

    for b in test_batches:
        test_obs = np.random.randn(b, obs_dim).astype(np.float32)

        # PyTorch inference
        with torch.no_grad():
            torch_out = pytorch_model(torch.from_numpy(test_obs)).numpy()

        # ONNX Runtime inference
        onnx_out = session.run([output_name], {input_name: test_obs})[0]

        diff = float(np.max(np.abs(torch_out - onnx_out)))
        if diff > max_diff:
            max_diff = diff

        assert torch_out.shape == (b, 4)
        assert onnx_out.shape == (b, 4)
        assert np.allclose(torch_out, onnx_out, atol=atol), (
            f"Parity mismatch for batch_size={b}: max_diff={diff:.2e} exceeds atol={atol}"
        )

        # Verify action equivalence (argmax)
        torch_actions = np.argmax(torch_out, axis=-1)
        onnx_actions = np.argmax(onnx_out, axis=-1)
        assert np.array_equal(torch_actions, onnx_actions), (
            f"Action selection mismatch between PyTorch and ONNX for batch_size={b}"
        )

    return {
        "verified": True,
        "max_absolute_difference": max_diff,
        "tolerance": atol,
        "batches_tested": test_batches,
        "onnx_model": str(path),
    }
