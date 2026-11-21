"""Inference bridge between the warehouse adapter and the in-repo QMIX learner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.marl.checkpoint_validator import (
    CHECKPOINT_SCHEMA_VERSION,
    SUPPORTED_FLEET_SIZES,
    CheckpointCompatibilityError,
    CheckpointMetadata,
    extract_or_infer_metadata,
    validate_checkpoint_architecture,
    validate_checkpoint_tensors,
    verify_controller_inference,
)
from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.marl.observation_encoder import ObservationEncoder
from src.marl.qmix_learner import QMIXLearner


class QMIXController:
    """Select live actions with the repository's QMIX individual network.

    The controller deliberately delegates movement, safety, collisions, and
    rewards to ``WarehouseEnvironment.step`` through the adapter. With no
    checkpoint it still exercises the real QMIX inference path, but its
    weights are initialised and therefore are not learned warehouse policy.
    """

    ACTION_DIM = 4

    def __init__(
        self,
        adapter: MARLEnvironmentAdapter,
        checkpoint_path: str | Path | None = None,
        fleet_checkpoints: Optional[Dict[int, str | Path]] = None,
        epsilon: float = 0.0,
    ):
        self.adapter = adapter
        self.epsilon = float(epsilon)
        self.fleet_checkpoints: Dict[int, str | Path] = dict(fleet_checkpoints or {})
        self.checkpoint_path: Optional[str] = None
        self.checkpoint_id: Optional[str] = None
        self.checkpoint_metadata: Optional[CheckpointMetadata] = None
        self.checkpoint_loaded = False
        self.encoder: Optional[ObservationEncoder] = None
        self.learner = self._new_learner()

        initial_fleet = self.adapter.get_num_agents()
        if checkpoint_path is not None:
            self.fleet_checkpoints[initial_fleet] = checkpoint_path
            self.load_checkpoint(checkpoint_path)
        elif initial_fleet in self.fleet_checkpoints:
            self.load_checkpoint(self.fleet_checkpoints[initial_fleet])

    def _new_learner(self, num_agents: Optional[int] = None) -> QMIXLearner:
        agents = num_agents if num_agents is not None else self.adapter.get_num_agents()
        layout = self.adapter.environment.get_active_layout()
        self.encoder = ObservationEncoder(
            layout.width,
            layout.height,
            include_static_map=True,
            include_path_costs=True,
        )
        return QMIXLearner(
            num_agents=agents,
            observation_dim=self.encoder.observation_dim,
            action_dim=self.ACTION_DIM,
            state_dim=11 * agents,
        )

    def reconfigure_fleet(
        self,
        new_fleet_size: int,
        checkpoint_path: Optional[str | Path] = None,
        allow_untrained: bool = True,
    ) -> Dict[str, Any]:
        """Atomically transition the controller to a new fleet size.

        Steps:
        1. Validate requested fleet size against SUPPORTED_FLEET_SIZES.
        2. Resolve checkpoint path from explicit arg, fleet mapping, or environment.
        3. Build candidate encoder and candidate learner in isolated state.
        4. Validate checkpoint metadata and tensor shapes strictly.
        5. Load compatible parameters into candidate learner.
        6. Verify candidate inference (shapes, action ranges, finite joint Q).
        7. Atomically commit candidate state to self on complete success.
        If any step fails, self remains completely unchanged.
        """
        if not isinstance(new_fleet_size, int) or isinstance(new_fleet_size, bool) or new_fleet_size <= 0:
            raise ValueError(f"Fleet size must be a positive integer, got {new_fleet_size}")

        if new_fleet_size not in SUPPORTED_FLEET_SIZES:
            raise ValueError(
                f"Unsupported fleet size: {new_fleet_size}. "
                f"Supported fleet sizes are: {SUPPORTED_FLEET_SIZES}."
            )

        # Resolve checkpoint target
        target_path: Optional[Path] = None
        if checkpoint_path is not None:
            target_path = Path(checkpoint_path)
        elif new_fleet_size in self.fleet_checkpoints:
            target_path = Path(self.fleet_checkpoints[new_fleet_size])
        elif f"QMIX_CHECKPOINT_N{new_fleet_size}" in os.environ:
            target_path = Path(os.environ[f"QMIX_CHECKPOINT_N{new_fleet_size}"])

        if target_path is not None and not target_path.is_file():
            raise FileNotFoundError(f"QMIX checkpoint not found: {target_path}")

        # Construct candidate architecture in isolation
        layout = self.adapter.environment.get_active_layout()
        candidate_encoder = ObservationEncoder(
            layout.width,
            layout.height,
            include_static_map=True,
            include_path_costs=True,
        )
        candidate_learner = QMIXLearner(
            num_agents=new_fleet_size,
            observation_dim=candidate_encoder.observation_dim,
            action_dim=self.ACTION_DIM,
            state_dim=11 * new_fleet_size,
        )

        candidate_loaded = False
        candidate_id = None
        candidate_meta = None

        if target_path is not None:
            try:
                payload = torch.load(target_path, map_location="cpu", weights_only=False)
            except Exception as error:
                raise ValueError(f"Corrupt or invalid QMIX checkpoint file: {target_path} ({error})") from error

            if not isinstance(payload, dict):
                raise ValueError("QMIX checkpoint must contain a state dictionary payload")

            # Extract and validate metadata
            meta = extract_or_infer_metadata(payload, file_path=target_path)
            validate_checkpoint_architecture(
                meta,
                expected_num_agents=new_fleet_size,
                expected_observation_dim=candidate_encoder.observation_dim,
                expected_action_dim=self.ACTION_DIM,
                expected_state_dim=11 * new_fleet_size,
            )

            # Validate tensor names and shapes strictly
            validate_checkpoint_tensors(
                payload,
                expected_num_agents=new_fleet_size,
                expected_observation_dim=candidate_encoder.observation_dim,
                expected_action_dim=self.ACTION_DIM,
                expected_state_dim=11 * new_fleet_size,
                require_mixer=True,
            )

            agent_state = payload.get("agent_network") or payload.get("model_state_dict")
            if agent_state is None:
                raise ValueError("QMIX checkpoint is missing 'agent_network'")
            mixer_state = payload.get("mixer")
            if mixer_state is None:
                raise ValueError("QMIX checkpoint is missing 'mixer'")

            candidate_learner.agent_network.load_state_dict(agent_state, strict=True)
            candidate_learner.mixer.load_state_dict(mixer_state, strict=True)
            candidate_learner.agent_network.eval()
            candidate_learner.mixer.eval()

            candidate_loaded = True
            candidate_meta = meta
            candidate_id = meta.run_id or meta.checkpoint_sha256 or target_path.name
        else:
            if not allow_untrained:
                raise ValueError(
                    f"No trained checkpoint available for fleet size N={new_fleet_size}. "
                    "Provide a compatible checkpoint or permit untrained mode explicitly."
                )

        # Verify candidate inference before committing
        verify_controller_inference(
            candidate_learner,
            num_agents=new_fleet_size,
            observation_dim=candidate_encoder.observation_dim,
            action_dim=self.ACTION_DIM,
            state_dim=11 * new_fleet_size,
        )

        # Atomic commit to self
        self.encoder = candidate_encoder
        self.learner = candidate_learner
        self.checkpoint_loaded = candidate_loaded
        self.checkpoint_path = str(target_path) if target_path else None
        self.checkpoint_id = candidate_id
        self.checkpoint_metadata = candidate_meta
        if target_path:
            self.fleet_checkpoints[new_fleet_size] = str(target_path)

        return self.telemetry

    def reset_for_fleet(self) -> None:
        """Rebuild network dimensions after changing the fleet size in the adapter."""
        target_size = self.adapter.get_num_agents()
        if target_size in SUPPORTED_FLEET_SIZES:
            self.reconfigure_fleet(target_size, allow_untrained=True)
        else:
            # Fallback for unconstrained test setups
            self.checkpoint_loaded = False
            self.checkpoint_path = None
            self.checkpoint_id = None
            self.checkpoint_metadata = None
            self.learner = self._new_learner(target_size)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load a checkpoint produced by the supported QMIX learner format.

        Validates metadata and tensor shapes strictly. An incompatible checkpoint
        is rejected without altering the controller state.
        """
        num_agents = self.learner.num_agents
        self.reconfigure_fleet(
            new_fleet_size=num_agents,
            checkpoint_path=checkpoint_path,
            allow_untrained=False,
        )

    def _observations(self) -> torch.Tensor:
        environment = self.adapter.environment
        layout = environment.get_active_layout()
        if self.encoder is None or (
            self.encoder.grid_width,
            self.encoder.grid_height,
        ) != layout.dimensions:
            self.encoder = ObservationEncoder(
                layout.width,
                layout.height,
                include_static_map=True,
                include_path_costs=True,
            )
        encoded = [
            self.encoder.encode(
                robot,
                environment.robots,
                environment.tasks,
                layout=layout,
            )
            for robot in environment.robots
        ]
        return torch.as_tensor(np.asarray(encoded, dtype=np.float32))

    def select_actions(self) -> dict[int, int]:
        """Return one QMIX action for each currently active robot."""
        observations = self._observations()
        action_masks = torch.as_tensor(
            self.adapter.environment.get_action_masks(), dtype=torch.bool
        )
        actions = self.learner.select_actions(
            observations, epsilon=self.epsilon, action_masks=action_masks
        )
        return {
            robot_id: int(action)
            for robot_id, action in zip(
                (robot.robot_id for robot in self.adapter.environment.robots),
                actions.tolist(),
            )
        }

    @property
    def telemetry(self) -> dict[str, object]:
        return {
            "source": "QMIX",
            "activeFleetSize": self.learner.num_agents,
            "checkpointLoaded": self.checkpoint_loaded,
            "trained": self.checkpoint_loaded,
            "checkpointPath": self.checkpoint_path,
            "checkpointId": self.checkpoint_id,
            "modelStatus": "trained" if self.checkpoint_loaded else "untrained",
            "architecture": {
                "numAgents": self.learner.num_agents,
                "observationDim": self.learner.observation_dim,
                "actionDim": self.ACTION_DIM,
                "stateDim": self.learner.state_dim,
            },
        }
