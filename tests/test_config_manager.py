from src.infrastructure.config_manager import load_all_configs


def test_load_all_configs():
    configs = load_all_configs()

    assert "environment" in configs
    assert "robots" in configs
    assert "reward" in configs
    assert "training" in configs
    assert "experiment" in configs


def test_config_values():
    configs = load_all_configs()

    assert configs["robots"]["num_robots"] == 5
    assert configs["experiment"]["algorithm"] == "QMIX"
    assert configs["experiment"]["random_seed"] == 42