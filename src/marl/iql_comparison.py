from typing import Dict, List

import numpy as np
import torch

from src.marl.gymnasium_env import WarehouseGymEnv
from src.marl.iql_learner import IQLLearner


class IQLComparison:
    """
    Compare trained Independent Q-Learning (IQL) agents
    against a rule-based baseline.

    Both controllers are evaluated on the same warehouse
    task so that their performance can be compared fairly.
    """

    def __init__(
        self,
        episodes: int = 10,
        max_steps: int = 100,
        seed: int = 42,
    ):
        if episodes <= 0:
            raise ValueError(
                "episodes must be greater than zero."
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        self.episodes = episodes
        self.max_steps = max_steps
        self.seed = seed

        self.num_robots = 5
        self.observation_dim = 50
        self.action_dim = 4

    def create_iql_learners(self) -> List[IQLLearner]:
        """
        Create one independent Q-learning learner
        for each robot.
        """

        return [
            IQLLearner(
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
            )
            for _ in range(self.num_robots)
        ]

    def _create_iql_learners(self) -> List[IQLLearner]:
        """
        Backward-compatible alias used by the test suite.
        """

        return self.create_iql_learners()

    def _add_test_task(
        self,
        env: WarehouseGymEnv,
    ) -> None:
        """
        Add a deterministic test task so both controllers
        solve the same warehouse problem.
        """

        env.adapter.environment.add_task(
            task_id=1,
            pickup_position=(1, 0),
            delivery_position=(2, 0),
            assigned_robot=0,
        )

    def _run_iql_episode(
        self,
        learners: List[IQLLearner],
        seed: int,
    ) -> Dict[str, float]:
        """
        Run one evaluation episode using trained IQL agents.
        """

        torch.manual_seed(seed)
        np.random.seed(seed)

        env = WarehouseGymEnv()

        observations, _ = env.reset()

        # Add the same task used by the rule-based baseline.
        self._add_test_task(env)

        total_reward = 0.0
        steps = 0

        for step in range(self.max_steps):

            actions = []

            for robot_id in range(self.num_robots):

                observation = torch.tensor(
                    observations[robot_id],
                    dtype=torch.float32,
                )

                action = learners[robot_id].select_action(
                    observation,
                    epsilon=0.0,
                )

                actions.append(int(action))

            (
                next_observations,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(actions)

            total_reward += float(reward)

            observations = next_observations
            steps = step + 1

            if terminated or truncated:
                break

        return {
            "reward": float(total_reward),
            "steps": float(steps),
        }

    def _run_rule_based_episode(
        self,
        seed: int,
    ) -> Dict[str, float]:
        """
        Run one evaluation episode using the rule-based
        baseline.

        The baseline currently uses action 0 for all robots.
        """

        torch.manual_seed(seed)
        np.random.seed(seed)

        env = WarehouseGymEnv()

        observations, _ = env.reset()

        # Add exactly the same task as the IQL episode.
        self._add_test_task(env)

        total_reward = 0.0
        steps = 0

        for step in range(self.max_steps):

            # Rule-based baseline.
            actions = [0] * self.num_robots

            (
                next_observations,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(actions)

            total_reward += float(reward)

            observations = next_observations
            steps = step + 1

            if terminated or truncated:
                break

        return {
            "reward": float(total_reward),
            "steps": float(steps),
        }

    def compare(
        self,
        learners: List[IQLLearner],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare trained IQL agents against the
        rule-based baseline.
        """

        if len(learners) != self.num_robots:
            raise ValueError(
                f"Expected {self.num_robots} learners, "
                f"received {len(learners)}."
            )

        iql_rewards = []
        iql_steps = []

        rule_rewards = []
        rule_steps = []

        for episode in range(self.episodes):

            episode_seed = self.seed + episode

            iql_result = self._run_iql_episode(
                learners=learners,
                seed=episode_seed,
            )

            rule_result = self._run_rule_based_episode(
                seed=episode_seed,
            )

            iql_rewards.append(
                iql_result["reward"]
            )

            iql_steps.append(
                iql_result["steps"]
            )

            rule_rewards.append(
                rule_result["reward"]
            )

            rule_steps.append(
                rule_result["steps"]
            )

        iql_average_reward = float(
            np.mean(iql_rewards)
        )

        rule_average_reward = float(
            np.mean(rule_rewards)
        )

        iql_average_steps = float(
            np.mean(iql_steps)
        )

        rule_average_steps = float(
            np.mean(rule_steps)
        )

        return {
            "iql": {
                "average_reward": iql_average_reward,
                "average_steps": iql_average_steps,
            },
            "rule_based": {
                "average_reward": rule_average_reward,
                "average_steps": rule_average_steps,
            },
            "difference": {
                "reward_difference": (
                    iql_average_reward
                    - rule_average_reward
                ),
                "step_difference": (
                    iql_average_steps
                    - rule_average_steps
                ),
            },
        }

    def print_comparison(
        self,
        results: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Print a readable comparison report.
        """

        print()
        print("IQL vs Rule-Based Comparison")
        print("=" * 40)

        print(
            f"IQL average reward: "
            f"{results['iql']['average_reward']:.3f}"
        )

        print(
            f"Rule-based average reward: "
            f"{results['rule_based']['average_reward']:.3f}"
        )

        print(
            f"Reward difference: "
            f"{results['difference']['reward_difference']:.3f}"
        )

        print(
            f"IQL average steps: "
            f"{results['iql']['average_steps']:.2f}"
        )

        print(
            f"Rule-based average steps: "
            f"{results['rule_based']['average_steps']:.2f}"
        )

        print(
            f"Step difference: "
            f"{results['difference']['step_difference']:.2f}"
        )

        print("=" * 40)