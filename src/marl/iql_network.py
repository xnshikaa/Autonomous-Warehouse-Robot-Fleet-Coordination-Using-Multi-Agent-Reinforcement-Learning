import torch
import torch.nn as nn


class IQLNetwork(nn.Module):
    """
    Independent Q-Learning network for a single robot.

    Each robot has its own Q-network in IQL.
    Input:
        observation -> 50 features

    Output:
        Q-value for each action -> 4 actions
    """

    def __init__(
        self,
        observation_dim: int = 50,
        action_dim: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()

        if observation_dim <= 0:
            raise ValueError("observation_dim must be greater than zero.")

        if action_dim <= 0:
            raise ValueError("action_dim must be greater than zero.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for the given observation.

        Supports:
            [observation_dim]
            [batch_size, observation_dim]

        Returns:
            Q-values with shape:
                [action_dim]
                or
                [batch_size, action_dim]
        """

        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation must be a torch.Tensor.")

        if observation.ndim not in (1, 2):
            raise ValueError(
                "observation must have shape "
                "[observation_dim] or [batch_size, observation_dim]."
            )

        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"received {observation.shape[-1]}."
            )

        return self.network(observation.float())