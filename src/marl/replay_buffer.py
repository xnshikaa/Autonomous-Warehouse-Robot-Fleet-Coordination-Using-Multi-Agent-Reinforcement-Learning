"""
Minimal Experience Replay Buffer for QMIX.

Stores individual multi-agent transitions and samples mini-batches
of experience for off-policy QMIX training.
"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch


class ReplayBuffer:
    """
    Circular experience replay buffer for cooperative multi-agent reinforcement learning.

    Stores fixed-size transitions containing:
    - observations: (num_agents, observation_dim)
    - global_state: (state_dim,)
    - actions: (num_agents,)
    - reward: scalar float
    - next_observations: (num_agents, observation_dim)
    - next_global_state: (state_dim,)
    - terminated: scalar float/bool
    """

    def __init__(
        self,
        capacity: int = 5000,
        num_agents: int = 5,
        observation_dim: int = 50,
        state_dim: int = 250,
        seed: int | None = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer.")

        self.capacity = capacity
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.state_dim = state_dim

        self.storage: list[dict[str, np.ndarray]] = []
        self.position = 0
        self.rng = random.Random(seed)

    def set_seed(self, seed: int | None = None) -> None:
        """Set random seed for reproducible mini-batch sampling."""
        self.rng = random.Random(seed)

    def push(
        self,
        observations: np.ndarray | torch.Tensor,
        global_state: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor | list[int],
        reward: float | int,
        next_observations: np.ndarray | torch.Tensor,
        next_global_state: np.ndarray | torch.Tensor,
        terminated: bool | float | int,
    ) -> None:
        """Add a single transition to the replay buffer."""

        obs_arr = np.asarray(observations, dtype=np.float32)
        state_arr = np.asarray(global_state, dtype=np.float32)
        actions_arr = np.asarray(actions, dtype=np.int64)
        next_obs_arr = np.asarray(next_observations, dtype=np.float32)
        next_state_arr = np.asarray(next_global_state, dtype=np.float32)
        term_val = float(terminated)

        # Validate shapes
        if obs_arr.shape != (self.num_agents, self.observation_dim):
            raise ValueError(
                f"Expected observations shape ({self.num_agents}, {self.observation_dim}), "
                f"got {obs_arr.shape}"
            )

        if state_arr.shape != (self.state_dim,):
            raise ValueError(
                f"Expected global_state shape ({self.state_dim},), got {state_arr.shape}"
            )

        if actions_arr.shape != (self.num_agents,):
            raise ValueError(
                f"Expected actions shape ({self.num_agents},), got {actions_arr.shape}"
            )

        if next_obs_arr.shape != (self.num_agents, self.observation_dim):
            raise ValueError(
                f"Expected next_observations shape ({self.num_agents}, {self.observation_dim}), "
                f"got {next_obs_arr.shape}"
            )

        if next_state_arr.shape != (self.state_dim,):
            raise ValueError(
                f"Expected next_global_state shape ({self.state_dim},), got {next_state_arr.shape}"
            )

        transition = {
            "observations": obs_arr,
            "global_states": state_arr,
            "actions": actions_arr,
            "rewards": np.float32(reward),
            "next_observations": next_obs_arr,
            "next_global_states": next_state_arr,
            "terminated": np.float32(term_val),
        }

        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer contains enough samples to draw a mini-batch."""
        return len(self.storage) >= batch_size

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        """
        Sample a mini-batch of transitions as PyTorch Tensors.

        Args:
            batch_size: Number of transitions to sample.
            device: Target torch device ('cpu' default).

        Returns:
            Dictionary mapping transition keys to torch.Tensors with shape [batch_size, ...]
        """
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with size {len(self.storage)}."
            )

        indices = self.rng.sample(range(len(self.storage)), batch_size)
        samples = [self.storage[i] for i in indices]

        obs_batch = np.stack([s["observations"] for s in samples], axis=0)
        state_batch = np.stack([s["global_states"] for s in samples], axis=0)
        actions_batch = np.stack([s["actions"] for s in samples], axis=0)
        rewards_batch = np.array([s["rewards"] for s in samples], dtype=np.float32)
        next_obs_batch = np.stack([s["next_observations"] for s in samples], axis=0)
        next_state_batch = np.stack([s["next_global_states"] for s in samples], axis=0)
        terminated_batch = np.array([s["terminated"] for s in samples], dtype=np.float32)

        return {
            "observations": torch.tensor(obs_batch, dtype=torch.float32, device=device),
            "global_states": torch.tensor(state_batch, dtype=torch.float32, device=device),
            "actions": torch.tensor(actions_batch, dtype=torch.long, device=device),
            "rewards": torch.tensor(rewards_batch, dtype=torch.float32, device=device),
            "next_observations": torch.tensor(next_obs_batch, dtype=torch.float32, device=device),
            "next_global_states": torch.tensor(next_state_batch, dtype=torch.float32, device=device),
            "terminated": torch.tensor(terminated_batch, dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return len(self.storage)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize buffer state for checkpointing."""
        return {
            "capacity": self.capacity,
            "num_agents": self.num_agents,
            "observation_dim": self.observation_dim,
            "state_dim": self.state_dim,
            "position": self.position,
            "storage": self.storage,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore buffer state from checkpoint."""
        self.capacity = state_dict.get("capacity", self.capacity)
        self.num_agents = state_dict.get("num_agents", self.num_agents)
        self.observation_dim = state_dict.get("observation_dim", self.observation_dim)
        self.state_dim = state_dict.get("state_dim", self.state_dim)
        self.position = state_dict.get("position", 0)
        self.storage = state_dict.get("storage", [])
