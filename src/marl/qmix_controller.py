"""Inference bridge between the warehouse adapter and the in-repo QMIX learner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

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

    OBSERVATION_DIM = 50
    ACTION_DIM = 4

    def __init__(
        self,
        adapter: MARLEnvironmentAdapter,
        checkpoint_path: str | Path | None = None,
        epsilon: float = 0.0,
    ):
        self.adapter = adapter
        self.epsilon = float(epsilon)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.checkpoint_loaded = False
        self.learner = self._new_learner()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def _new_learner(self) -> QMIXLearner:
        num_agents = self.adapter.get_num_agents()
        return QMIXLearner(
            num_agents=num_agents,
            observation_dim=self.OBSERVATION_DIM,
            action_dim=self.ACTION_DIM,
            state_dim=self.OBSERVATION_DIM * num_agents,
        )

    def reset_for_fleet(self) -> None:
        """Rebuild network dimensions after changing the fleet size."""

        self.checkpoint_loaded = False
        self.learner = self._new_learner()
        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load a checkpoint produced by the supported QMIX learner format.

        Supported payload:
        ``{"agent_network": state_dict, "mixer": state_dict}``.
        ``model_state_dict`` is accepted as an alias for ``agent_network``
        for simple inference checkpoints.
        """

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"QMIX checkpoint not found: {path}")

        payload: Any = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("QMIX checkpoint must contain a state dictionary payload")

        agent_state = payload.get("agent_network") or payload.get("model_state_dict")
        if agent_state is None:
            raise ValueError("QMIX checkpoint is missing 'agent_network'")
        self.learner.agent_network.load_state_dict(agent_state)

        mixer_state = payload.get("mixer")
        if mixer_state is not None:
            self.learner.mixer.load_state_dict(mixer_state)

        self.learner.agent_network.eval()
        self.learner.mixer.eval()
        self.checkpoint_path = str(path)
        self.checkpoint_loaded = True

    def _observations(self) -> torch.Tensor:
        environment = self.adapter.environment
        layout = environment.get_active_layout()
        encoder = ObservationEncoder(
            grid_width=layout.width,
            grid_height=layout.height,
        )
        encoded = [
            encoder.encode(
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
        actions = self.learner.select_actions(observations, epsilon=self.epsilon)
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
            "checkpointLoaded": self.checkpoint_loaded,
            "trained": self.checkpoint_loaded,
            "checkpointPath": self.checkpoint_path,
        }
