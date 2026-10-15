from pathlib import Path

from src.infrastructure.logger import get_logger, LOG_DIR


def test_logger_creates_log_file():
    logger = get_logger("test")

    logger.info("Test log message")

    log_files = list(Path(LOG_DIR).glob("*.log"))

    assert len(log_files) > 0


def test_logger_has_handlers():
    logger = get_logger("handler_test")

    assert len(logger.handlers) == 2


def test_logger_records_experiment_information():
    logger = get_logger(
        "experiment_test",
        experiment_name="qmix_baseline",
        algorithm="QMIX",
        num_robots=5,
        random_seed=42
    )

    logger.info("Experiment started")

    log_files = list(Path(LOG_DIR).glob("*.log"))

    assert len(log_files) > 0

def test_warehouse_event_logging():
    from src.infrastructure.logger import (
        log_episode_start,
        log_episode_end,
        log_task_completed,
        log_collision,
        log_safety_override,
        log_training_step
    )

    logger = get_logger("warehouse_event_test")

    log_episode_start(logger, 1)
    log_task_completed(logger, 10, 15.5)
    log_collision(logger, 2)
    log_safety_override(logger, 3, "human_corridor")
    log_training_step(logger, 1, 0.25)
    log_episode_end(logger, 1, 20.5)

    assert True