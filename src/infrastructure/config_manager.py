from pathlib import Path

from src.infrastructure.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_all_configs():
    """
    Load all project configuration files.

    Returns:
        Dictionary containing environment, robots, reward,
        training, and experiment configurations.
    """

    configs = {
        "environment": load_config(CONFIG_DIR / "environment.json"),
        "robots": load_config(CONFIG_DIR / "robots.json"),
        "reward": load_config(CONFIG_DIR / "reward.json"),
        "training": load_config(CONFIG_DIR / "training.json"),
        "experiment": load_config(CONFIG_DIR / "experiment.json")
    }

    return configs