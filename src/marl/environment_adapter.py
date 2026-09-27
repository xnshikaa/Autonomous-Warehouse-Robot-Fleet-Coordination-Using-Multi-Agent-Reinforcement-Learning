from src.ai.environment import WarehouseEnvironment


class MARLEnvironmentAdapter:
    """
    Adapter between the existing warehouse environment and
    a multi-agent RL training interface.

    This keeps the Week 3 WarehouseEnvironment unchanged while
    exposing observations, actions, rewards, and global state
    in a structure suitable for later QMIX/EPyMARL integration.
    """

    def __init__(self, num_agents=5, max_steps=100, layout_id=None):
        self.num_agents = num_agents

        self.environment = WarehouseEnvironment(
            num_robots=num_agents,
            max_steps=max_steps,
            layout_id=layout_id,
        )

    def reset(self):
        """
        Reset the underlying warehouse environment.

        Returns:
            observations: dictionary keyed by agent ID
            global_state: centralized training state
        """

        global_state, observations = self.environment.reset()

        observation_dict = {
            robot_observation.robot_id: robot_observation
            for robot_observation in observations
        }

        return observation_dict, global_state

    def step(self, actions):
        """
        Execute one action for every agent.

        Args:
            actions:
                Dictionary mapping agent IDs to action IDs.

        Returns:
            observations
            rewards
            terminated
            global_state
        """

        observations, rewards, terminated, global_state = (
            self.environment.step(actions)
        )

        observation_dict = {
            robot_observation.robot_id: robot_observation
            for robot_observation in observations
        }

        return (
            observation_dict,
            rewards,
            terminated,
            global_state
        )

    def get_action_space(self):
        """Return the shared discrete action space."""

        return self.environment.get_action_space()

    def get_num_agents(self):
        """Return the number of cooperative agents."""

        return self.num_agents

    def get_active_layout(self):
        return self.environment.get_active_layout()

    def set_layout(self, layout_id):
        self.environment.set_layout(layout_id)
