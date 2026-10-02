"""Unit tests for model configuration files and centralized config loader.

Tests both the transitional legacy configuration files currently in production
and the modernized target schema loader, validator, and candidate resolver.
"""

from pathlib import Path
import copy
import pytest
import yaml

from src.models.config import (
    load_model_config,
    resolve_candidate,
    validate_model_config,
)

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

# Models migrated to the modernized target schema (strict schema enforcement)
MIGRATED_TARGET_MODELS = {"catboost", "ft_transformer", "lightgbm", "ebm"}

# TRANSITIONAL: Legacy schema top-level keys for unmigrated configs
LEGACY_REQUIRED_TOP_LEVEL_KEYS = [
    "model",
    "reproducibility",
    "fixed_params",
    "validation",
    "search_space",
]


# =============================================================================
# Production Config Tests (Target schema for migrated models, legacy for unmigrated)
# =============================================================================

@pytest.mark.parametrize("model_id", EXPECTED_MODELS)
def test_model_config_file_exists_and_parses(model_id: str):
    """Verify that every required model config file exists and parses valid YAML."""
    config_file = CONFIGS_MODELS_DIR / f"{model_id}.yaml"
    assert config_file.exists(), f"Missing config file: {config_file}"

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), f"Config in {config_file} must be a dictionary."

    if model_id in MIGRATED_TARGET_MODELS:
        validate_model_config(config, strict_schema=True, model_id=model_id)
    else:
        for key in LEGACY_REQUIRED_TOP_LEVEL_KEYS:
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

    if model_id in MIGRATED_TARGET_MODELS:
        assert "candidates" in config
        assert isinstance(config["candidates"], list)
        assert len(config["candidates"]) > 0
    else:
        search_space = config["search_space"]
        assert isinstance(search_space, dict)
        assert len(search_space) > 0, f"Search space in {config_file.name} must not be empty."


# =============================================================================
# Centralized Loader Tests
# =============================================================================

@pytest.mark.parametrize("model_id", EXPECTED_MODELS)
def test_load_model_config_loader_reads_production_files(model_id: str):
    """Verify that load_model_config loads each production YAML via centralized loader."""
    config = load_model_config(model_id, strict_schema=False)
    assert isinstance(config, dict)
    assert config["model"]["model_id"] == model_id


def test_load_model_config_missing_file_raises():
    """Verify that attempting to load a non-existent config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model_config("non_existent_model_id_12345")


def test_load_model_config_malformed_yaml_raises(tmp_path: Path):
    """Verify that a malformed YAML file raises ValueError."""
    broken_yaml = tmp_path / "broken_model.yaml"
    broken_yaml.write_text("model: {name: 'Broken', invalid: [1, 2, }", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed YAML"):
        load_model_config(broken_yaml)


def test_load_model_config_non_dictionary_root_raises(tmp_path: Path):
    """Verify that a YAML file with a non-dict root raises ValueError."""
    list_yaml = tmp_path / "list_model.yaml"
    list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="did not parse into a top-level dictionary"):
        load_model_config(list_yaml)


# =============================================================================
# Modernized Target Schema & Strict Validation Tests
# =============================================================================

@pytest.fixture
def valid_target_config() -> dict:
    """Fixture providing a clean configuration conforming to the target schema."""
    return {
        "model": {
            "id": "m3_xgboost",
            "model_id": "xgboost",
            "name": "XGBoost Classifier",
            "owner": "member_3",
            "member": "Member 3",
            "algorithm_family": "Extreme Gradient Boosting",
            "implementation": "xgboost.XGBClassifier",
            "config_version": "2.0.0",
        },
        "reproducibility": {
            "seed": 42,
            "n_jobs": -1,
            "device": "cpu",
            "deterministic": True,
        },
        "inner_validation": {
            "fraction": 0.15,
            "stratified": True,
            "metric": "average_precision",
            "patience": 50,
            "max_iterations": 600,
            "refit_outer_train": False,
        },
        "class_imbalance": {
            "strategy": "dynamic_ratio",
            "param_name": "scale_pos_weight",
        },
        "fixed_params": {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "verbosity": 0,
            "tree_method": "hist",
        },
        "candidates": [
            {
                "id": "C01",
                "description": "Baseline hyperparameters",
                "params": {
                    "learning_rate": 0.05,
                    "max_depth": 5,
                    "subsample": 0.8,
                },
            },
            {
                "id": "C02",
                "description": "Regularized low learning-rate",
                "params": {
                    "learning_rate": 0.02,
                    "max_depth": 4,
                    "subsample": 0.7,
                },
            },
        ],
    }


def test_target_schema_valid_fixture_passes(valid_target_config: dict):
    """Verify that a compliant target configuration passes strict validation."""
    validate_model_config(valid_target_config, strict_schema=True, model_id="xgboost")


def test_target_schema_rejects_missing_section(valid_target_config: dict):
    """Verify that omitting a mandatory target section raises ValueError."""
    del valid_target_config["inner_validation"]
    with pytest.raises(ValueError, match="Strict target schema missing required sections"):
        validate_model_config(valid_target_config, strict_schema=True)


def test_target_schema_rejects_invalid_inner_validation_fraction(valid_target_config: dict):
    """Verify that invalid inner-validation fraction raises ValueError."""
    valid_target_config["inner_validation"]["fraction"] = 0.8  # Unreasonable fraction
    with pytest.raises(ValueError, match="'inner_validation.fraction' must be a float between 0.0 and 0.5"):
        validate_model_config(valid_target_config, strict_schema=True)


def test_target_schema_rejects_duplicate_candidate_ids(valid_target_config: dict):
    """Verify that duplicate candidate IDs in candidates sequence are rejected."""
    valid_target_config["candidates"].append(
        {"id": "C01", "description": "Duplicate ID", "params": {"max_depth": 3}}
    )
    with pytest.raises(ValueError, match="Duplicate candidate id 'C01' detected"):
        validate_model_config(valid_target_config, strict_schema=True)


def test_target_schema_rejects_empty_candidate_params(valid_target_config: dict):
    """Verify that candidates with missing or empty parameters are rejected."""
    valid_target_config["candidates"].append(
        {"id": "C03", "description": "Empty params", "params": {}}
    )
    with pytest.raises(ValueError, match="Candidate 'C03' has no model parameters declared"):
        validate_model_config(valid_target_config, strict_schema=True)


def test_target_schema_rejects_unresolved_todos(valid_target_config: dict):
    """Verify that unresolved TODO placeholders are rejected in strict schema mode."""
    valid_target_config["candidates"][0]["params"]["learning_rate"] = "TODO: tune this"
    with pytest.raises(ValueError, match="Strict schema rejects unresolved TODO"):
        validate_model_config(valid_target_config, strict_schema=True)


def test_target_schema_rejects_duplicated_predictor_list(valid_target_config: dict):
    """Verify that duplicating the 17 predictor names under preprocessing is rejected."""
    valid_target_config["preprocessing"] = {
        "features": [f"feature_{i}" for i in range(17)]
    }
    with pytest.raises(ValueError, match="Do not duplicate the 17 predictor names in model configs"):
        validate_model_config(valid_target_config, strict_schema=True)


def test_target_schema_semantic_class_imbalance_strategies(valid_target_config: dict):
    """Verify semantic class imbalance validation accepts valid strategies and rejects invalid ones."""
    for strategy in ("none", "balanced", "dynamic_ratio"):
        valid_target_config["class_imbalance"]["strategy"] = strategy
        validate_model_config(valid_target_config, strict_schema=True)

    valid_target_config["class_imbalance"]["strategy"] = "invalid_strategy_xyz"
    with pytest.raises(ValueError, match="Invalid class imbalance strategy"):
        validate_model_config(valid_target_config, strict_schema=True)


# =============================================================================
# Candidate Resolver Tests
# =============================================================================

def test_resolve_candidate_deterministic_merging(valid_target_config: dict):
    """Verify that candidate parameters deterministically override fixed parameters."""
    resolved_c01 = resolve_candidate(valid_target_config, candidate_id="C01")
    assert resolved_c01["candidate_id"] == "C01"
    assert resolved_c01["params"]["objective"] == "binary:logistic"  # From fixed_params
    assert resolved_c01["params"]["learning_rate"] == 0.05           # From C01
    assert resolved_c01["params"]["max_depth"] == 5                  # From C01

    resolved_c02 = resolve_candidate(valid_target_config, candidate_id="C02")
    assert resolved_c02["candidate_id"] == "C02"
    assert resolved_c02["params"]["learning_rate"] == 0.02           # From C02
    assert resolved_c02["params"]["max_depth"] == 4                  # From C02


def test_resolve_candidate_defaults_to_first_candidate(valid_target_config: dict):
    """Verify that omitting candidate_id defaults deterministically to the first declared candidate."""
    resolved = resolve_candidate(valid_target_config, candidate_id=None)
    assert resolved["candidate_id"] == "C01"
    assert resolved["params"]["learning_rate"] == 0.05


def test_resolve_candidate_unknown_id_raises(valid_target_config: dict):
    """Verify that requesting an unknown candidate ID raises ValueError."""
    with pytest.raises(ValueError, match="Unknown candidate_id 'UNKNOWN_C99'"):
        resolve_candidate(valid_target_config, candidate_id="UNKNOWN_C99")


def test_resolve_candidate_does_not_mutate_source_config(valid_target_config: dict):
    """Verify that candidate resolution preserves immutability of the source config."""
    original_copy = copy.deepcopy(valid_target_config)

    resolved = resolve_candidate(valid_target_config, candidate_id="C01")
    resolved["params"]["learning_rate"] = 999.0
    resolved["params"]["new_injected_key"] = "test"

    assert valid_target_config == original_copy, "Source config was mutated by resolve_candidate!"


# =============================================================================
# FT-Transformer Production Config & Candidate Tests
# =============================================================================

def test_ft_transformer_production_config_strict_validation():
    """Verify that ft_transformer.yaml parses and passes strict schema validation."""
    config = load_model_config("ft_transformer", strict_schema=True)
    assert config["model"]["model_id"] == "ft_transformer"
    assert config["model"]["config_version"] == "2.0.0"
    assert config["inner_validation"]["fraction"] == 0.15
    assert config["inner_validation"]["metric"] == "average_precision"
    assert config["inner_validation"]["patience"] == 15
    assert config["inner_validation"]["max_epochs"] == 100
    assert config["inner_validation"]["refit_outer_train"] is True


def test_ft_transformer_candidate_registry_integrity():
    """Verify FT01-FT08 candidate IDs, order, architectural guards, and weighting."""
    config = load_model_config("ft_transformer", strict_schema=True)
    candidates = config["candidates"]
    assert len(candidates) == 8, f"Expected exactly 8 candidates, got {len(candidates)}"

    expected_ids = [f"FT{i:02d}" for i in range(1, 9)]
    actual_ids = [c["id"] for c in candidates]
    assert actual_ids == expected_ids, f"Candidate IDs mismatch: {actual_ids} != {expected_ids}"
    assert len(set(actual_ids)) == len(actual_ids), "Candidate IDs must be strictly unique"

    valid_weighting = {"none", "dynamic_ratio"}

    for c in candidates:
        cid = c["id"]
        params = c["params"]

        # Architectural bounds
        assert params["n_blocks"] in {2, 3}, f"{cid} has invalid n_blocks: {params['n_blocks']}"
        assert params["d_block"] in {64, 128, 192}, f"{cid} has invalid d_block: {params['d_block']}"
        assert params["attention_n_heads"] in {4, 8}, f"{cid} has invalid attention_n_heads: {params['attention_n_heads']}"

        # Divisibility guard
        assert params["d_block"] % params["attention_n_heads"] == 0, (
            f"{cid}: d_block ({params['d_block']}) must be divisible by attention_n_heads ({params['attention_n_heads']})"
        )

        # Weighting strategy
        assert params["weighting"] in valid_weighting, (
            f"{cid} has invalid weighting: {params['weighting']}"
        )

        # Candidate resolution via existing central API
        resolved = resolve_candidate(config, candidate_id=cid)
        assert resolved["candidate_id"] == cid
        assert resolved["params"]["n_blocks"] == params["n_blocks"]
        assert resolved["params"]["d_block"] == params["d_block"]
        assert resolved["params"]["attention_n_heads"] == params["attention_n_heads"]
        assert resolved["params"]["weighting"] == params["weighting"]
        assert resolved["params"]["batch_size"] == 512
        assert resolved["params"]["weight_decay"] == 1e-5
        assert resolved["params"]["attention_dropout"] == 0.1
        assert resolved["params"]["residual_dropout"] == 0.0
