from pathlib import Path

import pytest
import yaml

from utils.config_loader import load_config


def test_load_config_applies_defaults(tmp_path: Path) -> None:
    """Loading a minimal config should fill in default values."""
    minimal = {
        "model": {},
        "training": {},
        "data": {},
        "evaluation": {},
        "web_gui": {},
        "system": {},
        "logging": {},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(minimal))

    config = load_config(str(cfg_path))
    assert config["model"]["chess_transformer"]["input_channels"] == 12


def test_load_config_missing_section(tmp_path: Path) -> None:
    """Missing a required top-level section should raise a clear error."""
    partial = {"training": {}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(partial))

    with pytest.raises(ValueError):
        load_config(str(cfg_path))

