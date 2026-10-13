import random
from typing import Dict, List, Optional

import numpy as np
import torch

from src.marl.gymnasium_env import WarehouseGymEnv
from src.marl.iql_learner import IQLLearner


class IQLTrainer:
    """
    Independent Q-Learning trainer.

    Each robot has its own independent Q-learning agent.
    All agents interact with the existing Week 4 WarehouseGymEnv.
    """

    def __init__(
        self,
        num_robots: int = 5,
        observation_dim: int = 50,
        action_dim: int = 4,
        episodes: int = 100,
        max_steps: int = 100,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        target_update_frequency: int = 10,
        seed: Optional[int] = None,
    ):
        if num_robots <= 0:
            raise ValueError("num_robots must be greater than zero.")

        if observation_dim <= 0:
            raise ValueError(
                "observation_dim must be greater than zero."
            )

        if action_dim <= 0:
            raise ValueError(
                "action_dim must be greater than zero."
            )

        if episodes <= 0:
            raise ValueError(
                "episodes must be greater than zero."
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        if not 0.0 <= epsilon_start <= 1.0:
            raise ValueError(
                "epsilon_start must be between 0 and 1."
            )

        if not 0.0 <= epsilon_end <= 1.0:
            raise ValueError(
                "epsilon_end must be between 0 and 1."
            )

        if epsilon_end > epsilon_start:
            raise ValueError(
                "epsilon_end cannot be greater than epsilon_start."
            )

        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError(
                "epsilon_decay must be in the range (0, 1]."
            )

        if target_update_frequency <= 0:
            raise ValueError(
                "target_update_frequency must be greater than zero."
            )

        self.num_robots = num_robots
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.episodes = episodes
        self.max_steps = max_steps

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.target_update_frequency = target_update_frequency

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Use the existing Week 4 environment exactly as it is.
        self.env = WarehouseGymEnv()

        # One independent learner per robot.
        self.learners = [
            IQLLearner(
                observation_dim=observation_dim,
                action_dim=action_dim,
                learning_rate=learning_rate,
                gamma=gamma,
            )
            for _ in range(num_robots)
        ]

        self.epsilon = epsilon_start

    def _select_actions(
        self,
        observations: np.ndarray,
    ) -> List[int]:
        """
        Select one action independently for each robot.
        """

        observations = np.asarray(
            observations,
            dtype=np.float32,
        )

        if observations.shape != (
            self.num_robots,
            self.observation_dim,
        ):
            raise ValueError(
                "Unexpected observation shape. "
                f"Expected "
                f"({self.num_robots}, {self.observation_dim}), "
                f"received {observations.shape}."
            )

        actions = []

        for robot_id in range(self.num_robots):
            observation = torch.as_tensor(
                observations[robot_id],
                dtype=torch.float32,
            )

            action = self.learners[robot_id].select_action(
                observation,
                epsilon=self.epsilon,
            )

            actions.append(action)

        return actions

    def train(self) -> List[Dict[str, float]]:
        """
        Train all independent Q-learning agents.

        Returns:
            List containing one training record per episode.
        """

        history = []

        for episode in range(1, self.episodes + 1):

            observations, _ = self.env.reset()

            episode_reward = 0.0
            episode_losses = []
            steps = 0

            for step in range(1, self.max_steps + 1):

                actions = self._select_actions(
                    observations
                )

                (
                    next_observations,
                    team_reward,
                    terminated,
                    truncated,
                    _,
                ) = self.env.step(actions)

                episode_reward += float(team_reward)

                episode_done = (
                    terminated or truncated
                )

                # Every robot learns independently.
                for robot_id in range(self.num_robots):

                    result = self.learners[
                        robot_id
                    ].train_step(
                        observation=observations[
                            robot_id
                        ],
                        action=actions[
                            robot_id
                        ],
                        reward=float(team_reward),
                        next_observation=next_observations[
                            robot_id
                        ],
                        terminated=episode_done,
                    )

                    episode_losses.append(
                        result["loss"]
                    )

                observations = next_observations
                steps = step

                if episode_done:
                    break

            # Periodically update target networks.
            if (
                episode
                % self.target_update_frequency
                == 0
            ):
                for learner in self.learners:
                    learner.update_target_network()

            # Decay epsilon after every episode.
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay,
            )

            average_loss = (
                sum(episode_losses)
                / len(episode_losses)
                if episode_losses
                else 0.0
            )

            history.append(
                {
                    "episode": episode,
                    "steps": steps,
                    "epsilon": self.epsilon,
                    "reward": episode_reward,
                    "loss": average_loss,
                }
            )

        return history

    def evaluate(
        self,
        episodes: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate trained IQL agents without exploration.
        """

        if episodes <= 0:
            raise ValueError(
                "episodes must be greater than zero."
            )

        previous_epsilon = self.epsilon

        # Disable exploration during evaluation.
        self.epsilon = 0.0

        rewards = []

        try:
            for _ in range(episodes):

                observations, _ = self.env.reset()

                episode_reward = 0.0

                for _ in range(self.max_steps):

                    actions = self._select_actions(
                        observations
                    )

                    (
                        next_observations,
                        team_reward,
                        terminated,
                        truncated,
                        _,
                    ) = self.env.step(actions)

                    episode_reward += float(
                        team_reward
                    )

                    observations = next_observations

                    if terminated or truncated:
                        break

                rewards.append(
                    episode_reward
                )

        finally:
            # Restore training epsilon.
            self.epsilon = previous_epsilon

        return {
            "episodes": float(episodes),
            "average_reward": (
                float(np.mean(rewards))
                if rewards
                else 0.0
            ),
            "max_reward": (
                float(np.max(rewards))
                if rewards
                else 0.0
            ),
            "min_reward": (
                float(np.min(rewards))
                if rewards
                else 0.0
            ),
        }

    def save_agents(
        self,
        directory: str,
    ) -> None:
        """
        Save all independent robot learners.
        """

        from pathlib import Path

        save_directory = Path(directory)
        save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for robot_id, learner in enumerate(
            self.learners
        ):
            learner.save(
                str(
                    save_directory
                    / f"iql_robot_{robot_id}.pt"
                )
            )

    def load_agents(
        self,
        directory: str,
    ) -> None:
        """
        Load all independent robot learners.
        """

        from pathlib import Path

        load_directory = Path(directory)

        for robot_id, learner in enumerate(
            self.learners
        ):
            learner.load(
                str(
                    load_directory
                    / f"iql_robot_{robot_id}.pt"
                )
            )