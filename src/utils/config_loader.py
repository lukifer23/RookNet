import os
import yaml
from pydantic import ValidationError

from .config_schema import ConfigModel


def load_config(config_path: str = "configs/config.v2.yaml") -> dict:
    """Load and validate the unified YAML configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        dict: The validated configuration with defaults applied.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the configuration fails schema validation.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f) or {}

    try:
        validated = ConfigModel(**raw_config)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}") from e

    return validated.model_dump()
