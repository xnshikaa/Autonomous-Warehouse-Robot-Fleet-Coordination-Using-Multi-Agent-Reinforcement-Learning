import logging
from pathlib import Path
from datetime import datetime


# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory where log files will be stored
LOG_DIR = PROJECT_ROOT / "logs"

# Create the logs directory if it doesn't exist
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(
    name="warehouse",
    experiment_name=None,
    algorithm=None,
    num_robots=None,
    random_seed=None
):
    """
    Create and return a configured logger.

    The logger writes messages to:
    1. The terminal
    2. A log file inside the logs/ directory

    Experiment information can optionally be included.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Create a unique log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"experiment_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Record experiment information
    if experiment_name is not None:
        logger.info(f"Experiment: {experiment_name}")

    if algorithm is not None:
        logger.info(f"Algorithm: {algorithm}")

    if num_robots is not None:
        logger.info(f"Robots: {num_robots}")

    if random_seed is not None:
        logger.info(f"Random seed: {random_seed}")

    return logger

def log_episode_start(logger, episode):
    """Log the start of an episode."""
    logger.info(f"Episode started | episode={episode}")


def log_episode_end(logger, episode, reward):
    """Log the end of an episode and its total reward."""
    logger.info(
        f"Episode completed | episode={episode} | reward={reward}"
    )


def log_task_completed(logger, task_id, delivery_time):
    """Log successful task completion."""
    logger.info(
        f"Task completed | task_id={task_id} | "
        f"delivery_time={delivery_time}"
    )


def log_collision(logger, robot_id):
    """Log a robot collision."""
    logger.warning(
        f"Collision detected | robot_id={robot_id}"
    )


def log_safety_override(
    logger,
    robot_id,
    reason,
    *,
    current_cell=None,
    proposed_action=None,
    target_cell=None,
    final_action=None,
    episode=None,
    timestep=None,
):
    """Log a complete safety override event.

    The optional event fields preserve compatibility with existing callers
    while allowing the live environment to log enough context for audit and
    replay.
    """
    logger.warning(
        "Safety override | "
        f"robot_id={robot_id} | reason={reason} | "
        f"current_cell={current_cell} | proposed_action={proposed_action} | "
        f"target_cell={target_cell} | final_action={final_action} | "
        f"episode={episode} | timestep={timestep}"
    )


def log_training_step(logger, episode, loss):
    """Log training information."""
    logger.debug(
        f"Training step | episode={episode} | loss={loss}"
    )
