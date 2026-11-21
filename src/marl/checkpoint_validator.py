"""Validation and metadata management for QMIX checkpoints across fleet sizes."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.marl.qmix_network import AgentQNetwork, QMIXMixer

CHECKPOINT_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}
SUPPORTED_FLEET_SIZES = (5, 10, 20)

AGENT_NETWORK_ARCH = "AgentQNetwork"
MIXER_ARCH = "QMIXMixer"
DEFAULT_AGENT_HIDDEN_DIM = 64
DEFAULT_MIXING_HIDDEN_DIM = 32
DEFAULT_ACTION_DIM = 4


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint is incompatible with the target architecture or configuration."""

    def __init__(
        self,
        message: str,
        expected: Optional[Dict[str, Any]] = None,
        actual: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.expected = expected or {}
        self.actual = actual or {}


@dataclass
class CheckpointMetadata:
    """Standardized metadata schema for MARL/QMIX checkpoints."""

    schema_version: int
    algorithm: str
    num_agents: int
    observation_dim: int
    action_dim: int
    state_dim: int
    agent_network_arch: str
    mixer_arch: str
    agent_hidden_dim: int
    mixing_hidden_dim: int
    action_space: str
    run_id: Optional[str] = None
    step: Optional[int] = None
    episode: Optional[int] = None
    timestamp: Optional[str] = None
    is_best: Optional[bool] = None
    best_metric_val: Optional[float] = None
    checkpoint_sha256: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_file_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a checkpoint file."""
    path = Path(file_path)
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_checkpoint_metadata(
    num_agents: int,
    observation_dim: int,
    action_dim: int = DEFAULT_ACTION_DIM,
    state_dim: Optional[int] = None,
    agent_hidden_dim: int = DEFAULT_AGENT_HIDDEN_DIM,
    mixing_hidden_dim: int = DEFAULT_MIXING_HIDDEN_DIM,
    run_id: Optional[str] = None,
    step: Optional[int] = None,
    episode: Optional[int] = None,
    is_best: bool = False,
    best_metric_val: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a full metadata dictionary for serialization."""
    effective_state_dim = state_dim if state_dim is not None else 11 * num_agents
    metadata = CheckpointMetadata(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        algorithm="QMIX",
        num_agents=int(num_agents),
        observation_dim=int(observation_dim),
        action_dim=int(action_dim),
        state_dim=int(effective_state_dim),
        agent_network_arch=AGENT_NETWORK_ARCH,
        mixer_arch=MIXER_ARCH,
        agent_hidden_dim=int(agent_hidden_dim),
        mixing_hidden_dim=int(mixing_hidden_dim),
        action_space=f"Discrete({action_dim})",
        run_id=str(run_id) if run_id else None,
        step=int(step) if step is not None else None,
        episode=int(episode) if episode is not None else None,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        is_best=bool(is_best),
        best_metric_val=float(best_metric_val) if best_metric_val is not None else None,
        config=config or {},
    )
    return metadata.to_dict()


def extract_or_infer_metadata(
    payload: Dict[str, Any],
    file_path: Optional[str | Path] = None,
) -> CheckpointMetadata:
    """Extract metadata or safely infer it from tensor shapes when unambiguous."""
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError(
            f"Checkpoint payload must be a dictionary, got {type(payload).__name__}"
        )

    # 1. Check for explicit metadata dictionary
    meta_dict = payload.get("metadata")
    if isinstance(meta_dict, dict):
        version = meta_dict.get("schema_version", 1)
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise CheckpointCompatibilityError(
                f"Unsupported checkpoint schema_version {version}. Supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        try:
            return CheckpointMetadata(
                schema_version=int(version),
                algorithm=meta_dict.get("algorithm", "QMIX"),
                num_agents=int(meta_dict["num_agents"]),
                observation_dim=int(meta_dict["observation_dim"]),
                action_dim=int(meta_dict.get("action_dim", DEFAULT_ACTION_DIM)),
                state_dim=int(meta_dict["state_dim"]),
                agent_network_arch=meta_dict.get("agent_network_arch", AGENT_NETWORK_ARCH),
                mixer_arch=meta_dict.get("mixer_arch", MIXER_ARCH),
                agent_hidden_dim=int(meta_dict.get("agent_hidden_dim", DEFAULT_AGENT_HIDDEN_DIM)),
                mixing_hidden_dim=int(meta_dict.get("mixing_hidden_dim", DEFAULT_MIXING_HIDDEN_DIM)),
                action_space=meta_dict.get("action_space", f"Discrete({meta_dict.get('action_dim', DEFAULT_ACTION_DIM)})"),
                run_id=meta_dict.get("run_id"),
                step=meta_dict.get("step"),
                episode=meta_dict.get("episode"),
                timestamp=meta_dict.get("timestamp"),
                is_best=meta_dict.get("is_best"),
                best_metric_val=meta_dict.get("best_metric_val"),
                checkpoint_sha256=meta_dict.get("checkpoint_sha256") or (calculate_file_sha256(file_path) if file_path else None),
                config=meta_dict.get("config") or payload.get("config"),
            )
        except (KeyError, ValueError, TypeError) as err:
            raise CheckpointCompatibilityError(
                f"Checkpoint metadata is corrupt or missing required fields: {err}"
            ) from err

    # 2. Check for legacy 'config' dictionary
    config = payload.get("config")
    if isinstance(config, dict) and "num_agents" in config and "observation_dim" in config:
        num_agents = int(config["num_agents"])
        obs_dim = int(config["observation_dim"])
        action_dim = int(config.get("action_dim", DEFAULT_ACTION_DIM))
        state_dim = int(config.get("state_dim", 11 * num_agents))
        return CheckpointMetadata(
            schema_version=1,
            algorithm="QMIX",
            num_agents=num_agents,
            observation_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            agent_network_arch=AGENT_NETWORK_ARCH,
            mixer_arch=MIXER_ARCH,
            agent_hidden_dim=DEFAULT_AGENT_HIDDEN_DIM,
            mixing_hidden_dim=DEFAULT_MIXING_HIDDEN_DIM,
            action_space=f"Discrete({action_dim})",
            step=payload.get("update_count"),
            episode=payload.get("episode"),
            checkpoint_sha256=calculate_file_sha256(file_path) if file_path else None,
            config=config,
        )

    # 3. Safe unambiguous inference from tensor shapes
    agent_state = payload.get("agent_network") or payload.get("model_state_dict")
    mixer_state = payload.get("mixer")

    if not agent_state:
        raise CheckpointCompatibilityError(
            "Checkpoint has no metadata, no config, and lacks 'agent_network' tensors. "
            "Cannot determine architecture compatibility."
        )

    try:
        # Agent network inference
        w0 = agent_state["network.0.weight"]
        w4 = agent_state["network.4.weight"]
        agent_hidden_dim = w0.shape[0]
        obs_dim = w0.shape[1]
        action_dim = w4.shape[0]

        if mixer_state is not None:
            # Mixer tensor inference
            hb1_w = mixer_state["hyper_b1.weight"]
            mixing_hidden_dim = hb1_w.shape[0]
            state_dim = hb1_w.shape[1]

            hw1_0_w = mixer_state["hyper_w1.0.weight"]
            w1_out = hw1_0_w.shape[0]

            if mixing_hidden_dim <= 0 or w1_out % mixing_hidden_dim != 0:
                raise CheckpointCompatibilityError(
                    f"Ambiguous or corrupt mixer hyper_w1 dimension: {w1_out} is not divisible by {mixing_hidden_dim}."
                )
            num_agents = w1_out // mixing_hidden_dim
        else:
            num_agents = 5  # cannot infer fleet size without mixer
            mixing_hidden_dim = DEFAULT_MIXING_HIDDEN_DIM
            state_dim = 11 * num_agents

        return CheckpointMetadata(
            schema_version=1,
            algorithm="QMIX",
            num_agents=num_agents,
            observation_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            agent_network_arch=AGENT_NETWORK_ARCH,
            mixer_arch=MIXER_ARCH,
            agent_hidden_dim=agent_hidden_dim,
            mixing_hidden_dim=mixing_hidden_dim,
            action_space=f"Discrete({action_dim})",
            checkpoint_sha256=calculate_file_sha256(file_path) if file_path else None,
            config={},
        )
    except (KeyError, IndexError, AttributeError) as err:
        raise CheckpointCompatibilityError(
            f"Failed to infer architecture from checkpoint tensor shapes: {err}. "
            "Ensure checkpoint contains standard AgentQNetwork and QMIXMixer state dicts."
        ) from err


def validate_checkpoint_architecture(
    metadata: CheckpointMetadata,
    expected_num_agents: int,
    expected_observation_dim: int,
    expected_action_dim: int = DEFAULT_ACTION_DIM,
    expected_state_dim: Optional[int] = None,
    expected_agent_hidden_dim: int = DEFAULT_AGENT_HIDDEN_DIM,
    expected_mixing_hidden_dim: int = DEFAULT_MIXING_HIDDEN_DIM,
) -> None:
    """Validate that metadata matches the required target architecture exactly."""
    effective_expected_state_dim = (
        expected_state_dim if expected_state_dim is not None else 11 * expected_num_agents
    )
    expected = {
        "num_agents": expected_num_agents,
        "observation_dim": expected_observation_dim,
        "action_dim": expected_action_dim,
        "state_dim": effective_expected_state_dim,
        "agent_hidden_dim": expected_agent_hidden_dim,
        "mixing_hidden_dim": expected_mixing_hidden_dim,
    }
    actual = {
        "num_agents": metadata.num_agents,
        "observation_dim": metadata.observation_dim,
        "action_dim": metadata.action_dim,
        "state_dim": metadata.state_dim,
        "agent_hidden_dim": metadata.agent_hidden_dim,
        "mixing_hidden_dim": metadata.mixing_hidden_dim,
    }

    mismatches = []
    if metadata.num_agents != expected_num_agents:
        mismatches.append(
            f"fleet size num_agents={metadata.num_agents} does not match expected {expected_num_agents}"
        )
    if metadata.observation_dim != expected_observation_dim:
        mismatches.append(
            f"observation_dim={metadata.observation_dim} does not match expected {expected_observation_dim} "
            f"(current policy requires {expected_observation_dim}-D map-aware input)"
        )
    if metadata.action_dim != expected_action_dim:
        mismatches.append(
            f"action_dim={metadata.action_dim} does not match expected {expected_action_dim}"
        )
    if metadata.state_dim != effective_expected_state_dim:
        mismatches.append(
            f"global state_dim={metadata.state_dim} does not match expected {effective_expected_state_dim}"
        )
    if metadata.agent_hidden_dim != expected_agent_hidden_dim:
        mismatches.append(
            f"agent_hidden_dim={metadata.agent_hidden_dim} does not match expected {expected_agent_hidden_dim}"
        )
    if metadata.mixing_hidden_dim != expected_mixing_hidden_dim:
        mismatches.append(
            f"mixing_hidden_dim={metadata.mixing_hidden_dim} does not match expected {expected_mixing_hidden_dim}"
        )

    if mismatches:
        raise CheckpointCompatibilityError(
            f"incompatible checkpoint architecture: {'; '.join(mismatches)}.",
            expected=expected,
            actual=actual,
        )


def validate_checkpoint_tensors(
    payload: Dict[str, Any],
    expected_num_agents: int,
    expected_observation_dim: int,
    expected_action_dim: int = DEFAULT_ACTION_DIM,
    expected_state_dim: Optional[int] = None,
    expected_agent_hidden_dim: int = DEFAULT_AGENT_HIDDEN_DIM,
    expected_mixing_hidden_dim: int = DEFAULT_MIXING_HIDDEN_DIM,
    require_mixer: bool = True,
) -> None:
    """Strictly validate tensor keys and shapes for agent network and mixer."""
    effective_state_dim = (
        expected_state_dim if expected_state_dim is not None else 11 * expected_num_agents
    )

    # 1. Agent Network Validation
    agent_state = payload.get("agent_network") or payload.get("model_state_dict")
    if agent_state is None:
        raise CheckpointCompatibilityError("QMIX checkpoint is missing 'agent_network' parameters.")

    expected_agent_shapes = {
        "network.0.weight": (expected_agent_hidden_dim, expected_observation_dim),
        "network.0.bias": (expected_agent_hidden_dim,),
        "network.2.weight": (expected_agent_hidden_dim, expected_agent_hidden_dim),
        "network.2.bias": (expected_agent_hidden_dim,),
        "network.4.weight": (expected_action_dim, expected_agent_hidden_dim),
        "network.4.bias": (expected_action_dim,),
    }

    agent_keys = set(agent_state.keys())
    expected_agent_keys = set(expected_agent_shapes.keys())

    missing_agent_keys = expected_agent_keys - agent_keys
    if missing_agent_keys:
        raise CheckpointCompatibilityError(
            f"Agent network state is missing required parameter keys: {sorted(missing_agent_keys)}"
        )

    unexpected_agent_keys = agent_keys - expected_agent_keys
    if unexpected_agent_keys:
        raise CheckpointCompatibilityError(
            f"Agent network state contains unexpected parameter keys: {sorted(unexpected_agent_keys)}"
        )

    for key, expected_shape in expected_agent_shapes.items():
        actual_shape = tuple(agent_state[key].shape)
        if actual_shape != expected_shape:
            raise CheckpointCompatibilityError(
                f"Agent network tensor '{key}' shape mismatch: expected {expected_shape}, got {actual_shape}",
                expected={"key": key, "shape": expected_shape},
                actual={"key": key, "shape": actual_shape},
            )

    # 2. Mixer Validation
    mixer_state = payload.get("mixer")
    if mixer_state is None:
        if require_mixer:
            raise CheckpointCompatibilityError(
                "QMIX checkpoint is missing 'mixer' parameters. A complete QMIX policy requires a trained mixer."
            )
        return

    hw1_dim = expected_mixing_hidden_dim * expected_num_agents
    expected_mixer_shapes = {
        "hyper_w1.0.weight": (hw1_dim, effective_state_dim),
        "hyper_w1.0.bias": (hw1_dim,),
        "hyper_w1.2.weight": (hw1_dim, hw1_dim),
        "hyper_w1.2.bias": (hw1_dim,),
        "hyper_b1.weight": (expected_mixing_hidden_dim, effective_state_dim),
        "hyper_b1.bias": (expected_mixing_hidden_dim,),
        "hyper_w2.0.weight": (expected_mixing_hidden_dim, effective_state_dim),
        "hyper_w2.0.bias": (expected_mixing_hidden_dim,),
        "hyper_w2.2.weight": (expected_mixing_hidden_dim, expected_mixing_hidden_dim),
        "hyper_w2.2.bias": (expected_mixing_hidden_dim,),
        "hyper_b2.0.weight": (expected_mixing_hidden_dim, effective_state_dim),
        "hyper_b2.0.bias": (expected_mixing_hidden_dim,),
        "hyper_b2.2.weight": (1, expected_mixing_hidden_dim),
        "hyper_b2.2.bias": (1,),
    }

    mixer_keys = set(mixer_state.keys())
    expected_mixer_keys = set(expected_mixer_shapes.keys())

    missing_mixer_keys = expected_mixer_keys - mixer_keys
    if missing_mixer_keys:
        raise CheckpointCompatibilityError(
            f"Mixer state is missing required parameter keys: {sorted(missing_mixer_keys)}"
        )

    unexpected_mixer_keys = mixer_keys - expected_mixer_keys
    if unexpected_mixer_keys:
        raise CheckpointCompatibilityError(
            f"Mixer state contains unexpected parameter keys: {sorted(unexpected_mixer_keys)}"
        )

    for key, expected_shape in expected_mixer_shapes.items():
        actual_shape = tuple(mixer_state[key].shape)
        if actual_shape != expected_shape:
            raise CheckpointCompatibilityError(
                f"Mixer tensor '{key}' shape mismatch for fleet size N={expected_num_agents}: "
                f"expected {expected_shape}, got {actual_shape}",
                expected={"key": key, "shape": expected_shape, "num_agents": expected_num_agents},
                actual={"key": key, "shape": actual_shape},
            )


def verify_controller_inference(
    learner: Any,
    num_agents: int,
    observation_dim: int,
    action_dim: int = DEFAULT_ACTION_DIM,
    state_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Perform a dry-run inference step to verify model outputs and joint Q-values are valid."""
    effective_state_dim = state_dim if state_dim is not None else 11 * num_agents

    dummy_observations = torch.zeros(
        (num_agents, observation_dim), dtype=torch.float32
    )

    with torch.no_grad():
        q_values = learner.agent_network(dummy_observations)

    if q_values.shape != (num_agents, action_dim):
        raise CheckpointCompatibilityError(
            f"Inference validation failed: agent Q-values shape {tuple(q_values.shape)} "
            f"does not match expected ({num_agents}, {action_dim})"
        )

    if not torch.isfinite(q_values).all():
        raise CheckpointCompatibilityError(
            "Inference validation failed: agent Q-values contain NaN or infinite values."
        )

    selected_actions = learner.select_actions(dummy_observations, epsilon=0.0)

    if selected_actions.shape != (num_agents,):
        raise CheckpointCompatibilityError(
            f"Inference validation failed: selected actions shape {tuple(selected_actions.shape)} "
            f"does not match expected ({num_agents},)"
        )

    for act in selected_actions.tolist():
        if act not in range(action_dim):
            raise CheckpointCompatibilityError(
                f"Inference validation failed: action ID {act} outside valid space [0, {action_dim - 1}]"
            )

    # Test mixer forward pass
    best_q = q_values.max(dim=-1).values.unsqueeze(0)  # (1, num_agents)
    dummy_state = torch.zeros((1, effective_state_dim), dtype=torch.float32)

    with torch.no_grad():
        q_tot = learner.mixer(best_q, dummy_state)

    if q_tot.shape != (1, 1):
        raise CheckpointCompatibilityError(
            f"Inference validation failed: mixer output shape {tuple(q_tot.shape)} "
            f"does not match expected (1, 1)"
        )

    if not torch.isfinite(q_tot).all():
        raise CheckpointCompatibilityError(
            "Inference validation failed: joint Q-value Q_tot contains NaN or infinite values."
        )

    return {
        "num_agents": num_agents,
        "q_values_shape": tuple(q_values.shape),
        "actions_shape": tuple(selected_actions.shape),
        "q_tot_finite": True,
        "valid": True,
    }
