import pytest

from utils.config_loader import load_config


def test_load_config_success():
    cfg = load_config("configs/config.v2.yaml")
    assert isinstance(cfg, dict)
    assert "training" in cfg


def test_load_config_missing():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
