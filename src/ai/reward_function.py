from src.infrastructure.config_manager import load_all_configs


class RewardFunction:
    """
    Calculates rewards for the warehouse robot RL environment.

    Reward values are loaded from config/reward.json.
    """

    def __init__(self):
        configs = load_all_configs()
        self.rewards = configs["reward"]

    def calculate(
        self,
        previous_distance,
        current_distance,
        delivered=False,
        collision=False,
        danger=False
    ):
        """
        Calculate the reward for one environment step.

        Args:
            previous_distance: Distance to target before the action.
            current_distance: Distance to target after the action.
            delivered: Whether the robot completed its delivery.
            collision: Whether a collision occurred.
            danger: Whether the robot entered a dangerous area/corridor.

        Returns:
            Float reward value.
        """

        reward = self.rewards["step_penalty"]

        # Reward progress toward the target.
        progress = previous_distance - current_distance
        reward += self.rewards["progress_reward"] * progress

        # Reward successful delivery.
        if delivered:
            reward += self.rewards["delivery_reward"]

        # Penalize collision.
        if collision:
            reward += self.rewards["collision_penalty"]

        # Penalize dangerous movement.
        if danger:
            reward += self.rewards["danger_penalty"]

        return reward