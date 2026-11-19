from collections.abc import Mapping
import math

from src.infrastructure.config_manager import load_all_configs


class RewardFunction:
    """
    Calculates rewards for the warehouse robot RL environment.

    Reward values are loaded from config/reward.json.
    """

    def __init__(self, reward_overrides: Mapping[str, float] | None = None):
        configs = load_all_configs()
        self.rewards = dict(configs["reward"])
        if reward_overrides:
            unknown = set(reward_overrides) - set(self.rewards)
            if unknown:
                raise ValueError(f"Unknown reward settings: {sorted(unknown)}")
            for name, value in reward_overrides.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Reward setting {name!r} must be numeric")
                if not math.isfinite(value):
                    raise ValueError(f"Reward setting {name!r} must be finite")
                self.rewards[name] = float(value)

    def calculate(
        self,
        previous_distance,
        current_distance,
        delivered=False,
        collision=False,
        danger=False,
        blocked=False,
    ):
        return self.calculate_components(
            previous_distance,
            current_distance,
            delivered=delivered,
            collision=collision,
            danger=danger,
            blocked=blocked,
        )["total"]

    def calculate_components(
        self,
        previous_distance,
        current_distance,
        delivered=False,
        collision=False,
        danger=False,
        blocked=False,
    ) -> dict[str, float]:
        """Return each applied reward term and their sum for diagnostics."""
        progress = previous_distance - current_distance
        components = {
            "step_penalty": float(self.rewards["step_penalty"]),
            "progress_reward": float(self.rewards["progress_reward"] * progress),
            "delivery_reward": float(self.rewards["delivery_reward"] if delivered else 0.0),
            "collision_penalty": float(self.rewards["collision_penalty"] if collision else 0.0),
            "danger_penalty": float(self.rewards["danger_penalty"] if danger else 0.0),
            "blocked_action_penalty": float(
                self.rewards["blocked_action_penalty"] if blocked else 0.0
            ),
        }
        components["total"] = float(sum(components.values()))
        return components
