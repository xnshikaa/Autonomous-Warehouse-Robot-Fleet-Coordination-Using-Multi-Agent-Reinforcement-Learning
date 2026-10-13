from src.infrastructure.config_manager import load_all_configs
from src.infrastructure.logger import (
    get_logger,
    log_episode_start,
    log_episode_end,
    log_task_completed,
    log_collision,
    log_safety_override
)


def test_infrastructure_integration():

    # Load all configurations
    configs = load_all_configs()

    experiment = configs["experiment"]

    # Create logger using experiment configuration
    logger = get_logger(
        "integration_test",
        experiment_name=experiment["experiment_name"],
        algorithm=experiment["algorithm"],
        num_robots=experiment["num_robots"],
        random_seed=experiment["random_seed"]
    )

    # Simulate important warehouse events
    log_episode_start(logger, 1)
    log_task_completed(logger, 1, 12.5)
    log_safety_override(logger, 2, "human_corridor")
    log_collision(logger, 3)
    log_episode_end(logger, 1, 25.0)

    assert configs["experiment"]["algorithm"] == "QMIX"
    assert configs["robots"]["num_robots"] == 5