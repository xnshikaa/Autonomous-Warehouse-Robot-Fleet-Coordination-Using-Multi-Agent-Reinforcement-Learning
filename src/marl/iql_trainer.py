import random
import json
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
        observation_dim: Optional[int] = None,
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
        include_static_map: bool = False,
        include_path_costs: bool = False,
        reward_overrides: Optional[Dict[str, float]] = None,
        episode_callback=None,
    ):
        if num_robots <= 0:
            raise ValueError("num_robots must be greater than zero.")

        if observation_dim is not None and observation_dim <= 0:
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
        self.action_dim = action_dim
        self.seed = 42 if seed is None else seed
        self.episodes = episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.include_static_map = include_static_map
        self.include_path_costs = include_path_costs
        self.reward_overrides = dict(reward_overrides or {})
        self.episode_callback = episode_callback
        self.training_episodes_completed = 0

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.target_update_frequency = target_update_frequency

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Use the existing Week 4 environment exactly as it is.
        self.env = WarehouseGymEnv(
            num_agents=num_robots,
            max_steps=max_steps,
            include_static_map=include_static_map,
            include_path_costs=include_path_costs,
            reward_overrides=self.reward_overrides,
        )
        self.observation_dim = self.env.observation_dim
        if observation_dim is not None and observation_dim != self.observation_dim:
            raise ValueError(
                f"observation_dim must match environment encoder ({self.observation_dim}), got {observation_dim}"
            )

        # One independent learner per robot.
        self.learners = [
            IQLLearner(
                observation_dim=self.observation_dim,
                action_dim=action_dim,
                learning_rate=learning_rate,
                gamma=gamma,
            )
            for _ in range(num_robots)
        ]

        self.epsilon = epsilon_start

    def _initialize_episode(self, episode_seed: int):
        """Reset the warehouse and add a reproducible set of safe tasks."""
        observations, _ = self.env.reset()
        environment = self.env.adapter.environment
        if not environment.tasks:
            from src.warehouse.task_generation import generate_task_assignments

            for task in generate_task_assignments(
                environment, random.Random(episode_seed)
            ):
                environment.add_task(
                    task.task_id,
                    task.pickup_position,
                    task.delivery_position,
                    task.assigned_robot,
                )
            observations = self.env._encode_observations()
        return observations

    def _select_actions(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray | None = None,
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
                action_mask=None if action_masks is None else action_masks[robot_id],
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

            observations = self._initialize_episode(self.seed + episode)

            episode_reward = 0.0
            episode_losses = []
            reward_component_totals: Dict[str, float] = {}
            safety_override_events = 0
            collision_intent_groups = 0
            steps = 0

            for step in range(1, self.max_steps + 1):

                action_masks = np.asarray(
                    self.env.adapter.environment.get_action_masks(), dtype=np.bool_
                )
                actions = self._select_actions(observations, action_masks)

                (
                    next_observations,
                    team_reward,
                    terminated,
                    truncated,
                    info,
                ) = self.env.step(actions)
                for name, value in info["team_reward_components"].items():
                    reward_component_totals[name] = reward_component_totals.get(name, 0.0) + float(value)
                safety_override_events += int(info["safety_overrides_step"])
                collision_intent_groups += int(info["collision_intent_groups_step"])
                next_action_masks = np.asarray(
                    self.env.adapter.environment.get_action_masks(), dtype=np.bool_
                )

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
                        next_action_mask=next_action_masks[robot_id],
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
                    "safety_override_events": safety_override_events,
                    "collision_intent_groups": collision_intent_groups,
                    "team_reward_component_totals": reward_component_totals,
                }
            )
            self.training_episodes_completed += 1
            if self.episode_callback is not None:
                self.episode_callback(self, history[-1])

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
            for episode in range(episodes):

                observations = self._initialize_episode(
                    self.seed + 1_000_000 + episode
                )

                episode_reward = 0.0

                for _ in range(self.max_steps):

                    action_masks = np.asarray(
                        self.env.adapter.environment.get_action_masks(), dtype=np.bool_
                    )
                    actions = self._select_actions(observations, action_masks)

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

        metadata = {
            "schema_version": 1,
            "algorithm": "IQL",
            "num_robots": self.num_robots,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "include_static_map": self.include_static_map,
            "include_path_costs": self.include_path_costs,
            "seed": self.seed,
            "episodes_completed": self.training_episodes_completed,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "target_update_frequency": self.target_update_frequency,
            "td_loss": self.learners[0].td_loss,
            "gradient_clip_norm": self.learners[0].gradient_clip_norm,
            "reward_overrides": self.reward_overrides,
        }
        (save_directory / "iql_manifest.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
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
