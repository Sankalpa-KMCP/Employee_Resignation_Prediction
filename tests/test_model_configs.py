"""Unit tests for model configuration files (configs/models/*.yaml)."""

from pathlib import Path
import yaml
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_MODELS_DIR = PROJECT_ROOT / "configs" / "models"

EXPECTED_MODELS = [
    "catboost",
    "ebm",
    "ft_transformer",
    "lightgbm",
    "tabnet",
    "xgboost",
]

REQUIRED_TOP_LEVEL_KEYS = [
    "model",
    "reproducibility",
    "fixed_params",
    "validation",
    "search_space",
]


@pytest.mark.parametrize("model_id", EXPECTED_MODELS)
def test_model_config_file_exists_and_parses(model_id: str):
    """Verify that every required model config file exists and parses valid YAML."""
    config_file = CONFIGS_MODELS_DIR / f"{model_id}.yaml"
    assert config_file.exists(), f"Missing config file: {config_file}"

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), f"Config in {config_file} must be a dictionary."

    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in config, f"Missing key '{key}' in {config_file.name}"


@pytest.mark.parametrize("model_id", EXPECTED_MODELS)
def test_model_config_metadata(model_id: str):
    """Verify model identity and reproducibility structure in config."""
    config_file = CONFIGS_MODELS_DIR / f"{model_id}.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_meta = config["model"]
    assert "name" in model_meta
    assert model_meta["model_id"] == model_id
    assert "member" in model_meta
    assert "implementation" in model_meta

    repro = config["reproducibility"]
    assert ("random_seed" in repro) or ("seed" in repro) or ("random_state" in repro)

    search_space = config["search_space"]
    assert isinstance(search_space, dict)
    assert len(search_space) > 0, f"Search space in {config_file.name} must not be empty."
