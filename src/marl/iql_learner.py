import random
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from src.marl.iql_network import IQLNetwork


class IQLLearner:
    """
    Independent Q-Learning learner for a single robot.

    Each robot learns independently using its own Q-network.
    """

    def __init__(
        self,
        observation_dim: int = 50,
        action_dim: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in the range (0, 1].")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gradient_clip_norm = 10.0
        self.td_loss = "smooth_l1"

        self.device = torch.device(device)

        self.q_network = IQLNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_network = IQLNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )

        # TD targets can spike under shared-reward multi-agent learning. Huber
        # loss and clipped gradients limit one bad transition's influence.
        self.loss_function = nn.SmoothL1Loss()

    def select_action(
        self,
        observation: torch.Tensor,
        epsilon: float = 0.0,
        action_mask=None,
    ) -> int:
        """
        Select an action using epsilon-greedy policy.
        """

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")

        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(
                observation,
                dtype=torch.float32,
            )

        valid_actions = list(range(self.action_dim))
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool).reshape(-1)
            if mask.numel() != self.action_dim or not bool(mask.any()):
                raise ValueError("action_mask must identify at least one valid action")
            valid_actions = torch.nonzero(mask, as_tuple=False).flatten().tolist()

        if random.random() < epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            observation = observation.to(self.device)
            q_values = self.q_network(observation)
            if action_mask is not None:
                q_values = q_values.masked_fill(~mask.to(self.device), -torch.inf)

            return int(torch.argmax(q_values).item())

    def train_step(
        self,
        observation: torch.Tensor,
        action: int,
        reward: float,
        next_observation: torch.Tensor,
        terminated: bool,
        next_action_mask=None,
    ) -> Dict[str, float]:
        """
        Perform one Q-learning update.
        """

        if not isinstance(action, int):
            raise ValueError("action must be an integer.")

        if action < 0 or action >= self.action_dim:
            raise ValueError(
                f"action must be between 0 and {self.action_dim - 1}."
            )

        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )

        next_observation = torch.as_tensor(
            next_observation,
            dtype=torch.float32,
            device=self.device,
        )

        reward_tensor = torch.tensor(
            float(reward),
            dtype=torch.float32,
            device=self.device,
        )

        q_values = self.q_network(observation)

        current_q = q_values[action]

        with torch.no_grad():
            next_q_values = self.target_network(next_observation)
            if next_action_mask is not None:
                mask = torch.as_tensor(next_action_mask, dtype=torch.bool, device=self.device).reshape(-1)
                if mask.numel() != self.action_dim or not bool(mask.any()):
                    raise ValueError("next_action_mask must identify at least one valid action")
                next_q_values = next_q_values.masked_fill(~mask, -torch.inf)
            max_next_q = torch.max(next_q_values)

            if terminated:
                target_q = reward_tensor
            else:
                target_q = reward_tensor + self.gamma * max_next_q

        loss = self.loss_function(
            current_q,
            target_q,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), max_norm=self.gradient_clip_norm
        )
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "q_value": float(current_q.detach().item()),
            "target_q": float(target_q.detach().item()),
        }

    def update_target_network(self) -> None:
        """
        Copy the online network weights into the target network.
        """

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    def save(self, path: str) -> None:
        """
        Save the learner's model weights.
        """

        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load previously saved learner weights.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.q_network.load_state_dict(
            checkpoint["q_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
