from src.infrastructure.config_loader import load_config


def test_environment_config():
    config = load_config("config/environment.json")

    assert "warehouse" in config
    assert "shelves" in config
    assert "dispatch" in config
    assert "human_corridors" in config


def test_robots_config():
    config = load_config("config/robots.json")

    assert "num_robots" in config
    assert "fleet_sizes" in config
    assert "actions" in config


def test_reward_config():
    config = load_config("config/reward.json")

    assert "delivery_reward" in config
    assert "collision_penalty" in config


def test_training_config():
    config = load_config("config/training.json")

    assert "learning_rate" in config
    assert "gamma" in config


def test_experiment_config():
    config = load_config("config/experiment.json")

    assert "experiment_name" in config
    assert "algorithm" in config
    assert "random_seed" in config