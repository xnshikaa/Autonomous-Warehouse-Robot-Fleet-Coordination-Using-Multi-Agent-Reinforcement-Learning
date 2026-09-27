"""
Configuration and Metrics Schemas for Autonomous Warehouse MARL Infrastructure.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

@dataclass
class ExperimentConfig:
    """
    Standard Hyperparameter Configuration Schema for Warehouse MARL Experiments.
    Designed for future QMIX, IQL, and rule-based algorithm integration.
    """
    project_name: str = "Autonomous-Warehouse-MARL"
    experiment_name: str = "warehouse_marl_coordination"
    algorithm: str = "QMIX"  # QMIX, IQL, HEURISTIC, etc.
    number_of_agents: int = 10  # Fleet size: 5, 10, 20
    environment_name: str = "Warehouse-20x20-v1"
    environment_version: str = "1.0.0"
    
    # Environment observation & action space specs
    observation_dim: int = 50  # 50-D observation vector
    action_space: str = "Discrete(4)"  # Discrete(4) action space
    action_dim: int = 4  # UP=0, DOWN=1, LEFT=2, RIGHT=3
    
    # Hyperparameters
    random_seed: int = 42
    learning_rate: float = 0.0005
    gamma: float = 0.99
    batch_size: int = 32
    episode_count: int = 1000
    max_episode_steps: int = 100
    
    # Exploration parameters (if applicable)
    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = 50000
    
    # Versioning & Tags
    reward_config_version: str = "v1.0"
    experiment_type: str = "training"  # training, evaluation, smoke_test
    
    # Extension dictionary for algorithm-specific parameters (e.g. QMIX mixer hyperparams)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary suitable for MLflow parameter logging."""
        d = asdict(self)
        extra = d.pop("extra_params", {})
        if isinstance(extra, dict):
            for k, v in extra.items():
                d[f"extra_{k}"] = v
        return d


@dataclass
class TrainingMetrics:
    """
    Standard Evaluation & Training Metrics Schema for Warehouse MARL.
    """
    episode_reward: float = 0.0
    task_completion_rate: float = 0.0
    completed_tasks: int = 0
    collision_or_conflict_count: int = 0
    average_delivery_time: float = 0.0
    throughput: float = 0.0
    idle_time: float = 0.0
    episode_length: int = 0
    
    # Extension dictionary for algorithm-specific metrics (e.g. QMIX loss, grad norm)
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for MLflow logging."""
        d = asdict(self)
        extra = d.pop("extra_metrics", {})
        res: Dict[str, float] = {}
        for k, v in d.items():
            res[k] = float(v)
        if isinstance(extra, dict):
            for k, v in extra.items():
                res[f"extra_{k}"] = float(v)
        return res
