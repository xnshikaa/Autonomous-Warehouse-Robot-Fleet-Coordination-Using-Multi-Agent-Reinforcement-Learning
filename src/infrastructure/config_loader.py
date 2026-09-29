import json
from pathlib import Path


def load_config(file_path):
    """
    Load a JSON configuration file.

    Args:
        file_path: Path to the JSON configuration file.

    Returns:
        A Python dictionary containing the configuration.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    if path.suffix.lower() != ".json":
        raise ValueError("Configuration file must be a JSON file.")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in configuration file: {file_path}"
        ) from error

    return config