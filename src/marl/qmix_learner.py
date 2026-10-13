from copy import deepcopy

import torch
import torch.nn.functional as F

from src.marl.qmix_network import AgentQNetwork, QMIXMixer


class QMIXLearner:
    """
    Minimal QMIX learner for cooperative multi-agent training.

    The learner uses:
    - one shared individual Q-network
    - one QMIX mixing network
    - corresponding target networks
    - common team reward
    """

    def __init__(
        self,
        num_agents: int = 5,
        observation_dim: int = 50,
        action_dim: int = 4,
        state_dim: int = 250,
        agent_hidden_dim: int = 64,
        mixing_hidden_dim: int = 32,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        target_update_interval: int = 10,
    ):
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.target_update_interval = target_update_interval

        self.agent_network = AgentQNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=agent_hidden_dim,
        )

        self.target_agent_network = deepcopy(
            self.agent_network
        )

        self.mixer = QMIXMixer(
            num_agents=num_agents,
            state_dim=state_dim,
            mixing_hidden_dim=mixing_hidden_dim,
        )

        self.target_mixer = deepcopy(
            self.mixer
        )

        self.optimizer = torch.optim.Adam(
            list(self.agent_network.parameters())
            + list(self.mixer.parameters()),
            lr=learning_rate,
        )

        self.update_count = 0

    def select_actions(
        self,
        observations: torch.Tensor,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """
        Select one action for each agent using epsilon-greedy policy.

        Args:
            observations:
                Shape (num_agents, observation_dim)

            epsilon:
                Probability of selecting a random action.

        Returns:
            Tensor of shape (num_agents,)
        """

        if observations.dim() != 2:
            raise ValueError(
                "observations must have shape "
                "(num_agents, observation_dim)"
            )

        if observations.size(0) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} agents, "
                f"received {observations.size(0)}"
            )

        with torch.no_grad():
            q_values = self.agent_network(
                observations
            )

        greedy_actions = q_values.argmax(
            dim=-1
        )

        if epsilon <= 0:
            return greedy_actions

        random_actions = torch.randint(
            low=0,
            high=self.action_dim,
            size=(self.num_agents,),
        )

        random_mask = (
            torch.rand(self.num_agents)
            < epsilon
        )

        return torch.where(
            random_mask,
            random_actions,
            greedy_actions,
        )

    def train_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        global_states: torch.Tensor,
        next_global_states: torch.Tensor,
        terminated: torch.Tensor,
    ):
        """
        Perform one QMIX training update.

        Expected shapes:

        observations:
            (batch, num_agents, observation_dim)

        actions:
            (batch, num_agents)

        rewards:
            (batch,)

        next_observations:
            (batch, num_agents, observation_dim)

        global_states:
            (batch, state_dim)

        next_global_states:
            (batch, state_dim)

        terminated:
            (batch,)
        """

        if observations.dim() != 3:
            raise ValueError(
                "observations must have shape "
                "(batch, num_agents, observation_dim)"
            )

        if actions.shape != (
            observations.size(0),
            self.num_agents,
        ):
            raise ValueError(
                "actions must have shape "
                "(batch, num_agents)"
            )

        # ---------------------------------------------
        # Current individual Q-values
        # ---------------------------------------------

        q_values = self.agent_network(
            observations
        )

        chosen_q_values = torch.gather(
            q_values,
            dim=2,
            index=actions.long().unsqueeze(-1),
        ).squeeze(-1)

        # ---------------------------------------------
        # Current joint Q-value
        # ---------------------------------------------

        q_total = self.mixer(
            chosen_q_values,
            global_states,
        ).squeeze(-1)

        # ---------------------------------------------
        # Target individual Q-values
        # ---------------------------------------------

        with torch.no_grad():

            next_q_values = (
                self.target_agent_network(
                    next_observations
                )
            )

            next_max_q_values = (
                next_q_values.max(
                    dim=2
                ).values
            )

            next_q_total = (
                self.target_mixer(
                    next_max_q_values,
                    next_global_states,
                ).squeeze(-1)
            )

            target = rewards + (
                self.gamma
                * (1.0 - terminated.float())
                * next_q_total
            )

        # ---------------------------------------------
        # TD loss
        # ---------------------------------------------

        loss = F.mse_loss(
            q_total,
            target,
        )

        # ---------------------------------------------
        # Optimization
        # ---------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.agent_network.parameters())
            + list(self.mixer.parameters()),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.update_count += 1

        if (
            self.update_count
            % self.target_update_interval
            == 0
        ):
            self.update_target_networks()

        return {
            "loss": float(loss.item()),
            "q_total_mean": float(
                q_total.mean().item()
            ),
            "target_mean": float(
                target.mean().item()
            ),
        }

    def update_target_networks(self):
        """Synchronize target networks."""

        self.target_agent_network.load_state_dict(
            self.agent_network.state_dict()
        )

        self.target_mixer.load_state_dict(
            self.mixer.state_dict()
        )