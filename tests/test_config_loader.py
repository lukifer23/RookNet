from src.utils.config_loader import load_config

REQUIRED_TOP_LEVEL_KEYS = {"model", "training", "evaluation", "system", "logging"}


def test_config_contains_required_sections():
    config = load_config("configs/config.v2.yaml")
    missing = REQUIRED_TOP_LEVEL_KEYS - config.keys()
    assert not missing, f"Config missing required sections: {missing}" 