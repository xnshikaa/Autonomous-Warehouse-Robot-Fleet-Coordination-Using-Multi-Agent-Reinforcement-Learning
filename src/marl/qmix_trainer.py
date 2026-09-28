import random

import numpy as np
import torch

from src.marl.gymnasium_env import WarehouseGymEnv
from src.marl.qmix_learner import QMIXLearner


class QMIXTrainer:
    """
    Minimal CPU-friendly QMIX training loop.

    This is a Week 4 foundation for validating that the
    warehouse environment, QMIX learner, and training loop
    can operate together.
    """

    def __init__(
        self,
        num_agents: int = 5,
        observation_dim: int = 50,
        action_dim: int = 4,
        state_dim: int = 250,
        max_steps: int = 20,
        episodes: int = 5,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        seed: int = 42,
    ):
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_steps = max_steps
        self.episodes = episodes

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        self.seed = seed

        self._set_seed(seed)

        self.env = WarehouseGymEnv(
            num_agents=num_agents,
            max_steps=max_steps,
        )

        self.learner = QMIXLearner(
            num_agents=num_agents,
            observation_dim=observation_dim,
            action_dim=action_dim,
            state_dim=state_dim,
        )

    @staticmethod
    def _set_seed(seed: int):
        """Set deterministic seeds for the smoke test."""

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _epsilon_for_episode(self, episode: int) -> float:
        """Linearly decay epsilon across training episodes."""

        if self.episodes <= 1:
            return self.epsilon_end

        progress = episode / (self.episodes - 1)

        return (
            self.epsilon_start
            + progress
            * (
                self.epsilon_end
                - self.epsilon_start
            )
        )

    def _build_global_state(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        """
        Build the current global-state vector.

        Week 4 uses the concatenated 50-D observations
        from all agents as a deterministic 250-D state
        representation.

        This is a baseline representation for the
        initial QMIX integration.
        """

        state = observation.reshape(-1)

        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"Expected global state dimension "
                f"{self.state_dim}, "
                f"received {state.shape[0]}"
            )

        return state.astype(np.float32)

    def train(self):
        """
        Run the minimal QMIX training loop.

        Returns:
            List of episode-level training metrics.
        """

        history = []

        for episode in range(self.episodes):

            observation, _ = self.env.reset()

            # Add one simple deterministic task for
            # the training smoke test.
            self.env.adapter.environment.add_task(
                task_id=1,
                pickup_position=(1, 0),
                delivery_position=(2, 0),
                assigned_robot=0,
            )

            epsilon = self._epsilon_for_episode(
                episode
            )

            episode_reward = 0.0
            losses = []

            for step in range(self.max_steps):

                observation_tensor = torch.tensor(
                    observation,
                    dtype=torch.float32,
                )

                actions = (
                    self.learner.select_actions(
                        observation_tensor,
                        epsilon=epsilon,
                    )
                )

                next_observation, reward, terminated, truncated, _ = (
                    self.env.step(
                        actions.tolist()
                    )
                )

                global_state = (
                    self._build_global_state(
                        observation
                    )
                )

                next_global_state = (
                    self._build_global_state(
                        next_observation
                    )
                )

                batch_observations = (
                    torch.tensor(
                        observation,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )

                batch_next_observations = (
                    torch.tensor(
                        next_observation,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )

                batch_actions = (
                    actions
                    .long()
                    .unsqueeze(0)
                )

                batch_rewards = torch.tensor(
                    [reward],
                    dtype=torch.float32,
                )

                batch_global_states = (
                    torch.tensor(
                        global_state,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )

                batch_next_global_states = (
                    torch.tensor(
                        next_global_state,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )

                batch_terminated = torch.tensor(
                    [float(terminated or truncated)],
                    dtype=torch.float32,
                )

                result = (
                    self.learner.train_batch(
                        observations=batch_observations,
                        actions=batch_actions,
                        rewards=batch_rewards,
                        next_observations=batch_next_observations,
                        global_states=batch_global_states,
                        next_global_states=batch_next_global_states,
                        terminated=batch_terminated,
                    )
                )

                losses.append(
                    result["loss"]
                )

                episode_reward += float(
                    reward
                )

                observation = next_observation

                if terminated or truncated:
                    break

            history.append(
                {
                    "episode": episode + 1,
                    "steps": step + 1,
                    "epsilon": epsilon,
                    "reward": episode_reward,
                    "loss": float(
                        np.mean(losses)
                    ),
                }
            )

        return history