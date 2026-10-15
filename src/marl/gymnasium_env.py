import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.marl.environment_adapter import MARLEnvironmentAdapter
from src.marl.observation_encoder import ObservationEncoder
from src.marl.spec import MARLEnvironmentSpec


class WarehouseGymEnv(gym.Env):
    """
    Gymnasium wrapper for the existing warehouse MARL environment.

    The existing environment uses:
        {robot_id: action}

    This wrapper provides the standard Gymnasium API.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_agents: int = 5,
        max_steps: int = 100,
    ):
        super().__init__()

        self.spec_config = MARLEnvironmentSpec(
            num_agents=num_agents,
            max_steps=max_steps,
        )

        self.adapter = MARLEnvironmentAdapter(
            num_agents=num_agents,
            max_steps=max_steps,
        )

        self.encoder = ObservationEncoder()

        self.num_agents = num_agents
        self.observation_dim = self.spec_config.observation_dim
        self.action_dim = self.spec_config.action_dim

        # One 50-dimensional observation for one robot.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # 0 = UP
        # 1 = DOWN
        # 2 = LEFT
        # 3 = RIGHT
        self.action_space = spaces.Discrete(self.action_dim)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        observations, global_state = self.adapter.reset()

        observation_array = self._encode_observations()

        info = {
            "global_state": global_state,
        }

        return observation_array, info

    def step(self, actions):
        """
        Execute one action for every robot.

        Gymnasium input:
            [action_0, action_1, ...]

        Existing environment input:
            {0: action_0, 1: action_1, ...}
        """

        if np.isscalar(actions):
            actions = [int(actions)]

        actions = [int(action) for action in actions]

        if len(actions) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} actions, "
                f"received {len(actions)}."
            )

        action_dict = {
            robot_id: actions[robot_id]
            for robot_id in self.spec_config.agent_ids
        }

        (
            observation_dict,
            rewards,
            terminated,
            global_state,
        ) = self.adapter.step(action_dict)

        observation_array = self._encode_observations()

        # QMIX uses a common team reward.
        reward = float(np.mean(list(rewards.values())))

        # The current Week 3 environment reports only one
        # termination flag. Truncation is therefore False here.
        truncated = False

        info = {
            "global_state": global_state,
            "agent_rewards": rewards,
            "agent_observations": observation_dict,
        }

        return (
            observation_array,
            reward,
            terminated,
            truncated,
            info,
        )

    def _encode_observations(self):
        """
        Encode the current warehouse state into one
        50-dimensional observation for each robot.
        """

        environment = self.adapter.environment

        robots = environment.robots
        tasks = environment.tasks

        encoded = []

        for robot_id in self.spec_config.agent_ids:

            robot = robots[robot_id]

            vector = self.encoder.encode(
                robot,
                robots,
                tasks,
            )

            vector = np.asarray(
                vector,
                dtype=np.float32,
            )

            if vector.shape != (self.observation_dim,):
                raise ValueError(
                    f"Expected observation shape "
                    f"({self.observation_dim},), "
                    f"got {vector.shape}."
                )

            encoded.append(vector)

        return np.asarray(
            encoded,
            dtype=np.float32,
        )