from dataclasses import dataclass


@dataclass(frozen=True)
class MARLEnvironmentSpec:
    """
    Specification of the warehouse multi-agent RL interface.

    This defines the contract used by the future QMIX/EPyMARL
    training implementation.
    """

    num_agents: int = 5
    observation_dim: int = 50
    action_dim: int = 4
    max_steps: int = 100

    @property
    def agent_ids(self) -> list[int]:
        """Return the integer IDs used by the Python environment."""
        return list(range(self.num_agents))

    @property
    def action_ids(self) -> list[int]:
        """Return all valid discrete policy actions."""
        return list(range(self.action_dim))