import torch
import torch.nn as nn


class AgentQNetwork(nn.Module):
    """
    Individual agent Q-network.

    Each robot receives a 50-dimensional local observation
    and produces one Q-value for each of the 4 actions.
    """

    def __init__(
        self,
        observation_dim: int = 50,
        action_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation:
                Tensor with shape (..., observation_dim)

        Returns:
            Q-values with shape (..., action_dim)
        """
        return self.network(observation)


class QMIXMixer(nn.Module):
    """
    QMIX mixing network.

    Combines individual agent Q-values into a single
    joint Q-value Q_tot using the global state.

    The mixing weights are generated from the global state
    and constrained to be non-negative, preserving the
    QMIX monotonicity property.
    """

    def __init__(
        self,
        num_agents: int = 5,
        state_dim: int = 250,
        mixing_hidden_dim: int = 32,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_hidden_dim = mixing_hidden_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(
                state_dim,
                mixing_hidden_dim * num_agents,
            ),
            nn.ReLU(),
            nn.Linear(
                mixing_hidden_dim * num_agents,
                mixing_hidden_dim * num_agents,
            ),
        )

        self.hyper_b1 = nn.Linear(
            state_dim,
            mixing_hidden_dim,
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(
                state_dim,
                mixing_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                mixing_hidden_dim,
                mixing_hidden_dim,
            ),
        )

        self.hyper_b2 = nn.Sequential(
            nn.Linear(
                state_dim,
                mixing_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                mixing_hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        agent_q_values: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            agent_q_values:
                Tensor of shape
                (batch_size, num_agents)

            global_state:
                Tensor of shape
                (batch_size, state_dim)

        Returns:
            Joint Q-value Q_tot with shape
            (batch_size, 1)
        """

        batch_size = agent_q_values.size(0)

        # Generate non-negative first-layer mixing weights.
        w1 = torch.abs(
            self.hyper_w1(global_state)
        )

        w1 = w1.view(
            batch_size,
            self.num_agents,
            self.mixing_hidden_dim,
        )

        # First-layer bias.
        b1 = self.hyper_b1(
            global_state
        ).view(
            batch_size,
            1,
            self.mixing_hidden_dim,
        )

        # Agent Q-values -> hidden mixing layer.
        agent_q_values = agent_q_values.view(
            batch_size,
            1,
            self.num_agents,
        )

        hidden = torch.bmm(
            agent_q_values,
            w1,
        )

        hidden = hidden + b1

        hidden = torch.relu(hidden)

        # Generate non-negative second-layer weights.
        w2 = torch.abs(
            self.hyper_w2(global_state)
        ).view(
            batch_size,
            self.mixing_hidden_dim,
            1,
        )

        # State-dependent final bias.
        b2 = self.hyper_b2(
            global_state
        ).view(
            batch_size,
            1,
            1,
        )

        q_total = torch.bmm(
            hidden,
            w2,
        )

        q_total = q_total + b2

        return q_total.squeeze(-1)