"""Unit tests for CatBoost configuration and CatBoostAdapter foundation (Step 2A).

Verifies strict schema loading, candidate resolution, inner-validation early stopping,
refit-on-all-outer-rows semantics, native categorical handling, and ModelAdapter compliance.
"""

import copy
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.base import ModelAdapter
from src.models.catboost_model import CatBoostAdapter
from src.models.config import (
    _detect_unresolved_placeholders,
    load_model_config,
    resolve_candidate,
    validate_model_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "configs" / "feature_contract.yaml"


@pytest.fixture
def catboost_raw_config() -> dict:
    """Load authoritative CatBoost configuration in strict schema mode."""
    return load_model_config("catboost", strict_schema=True)


@pytest.fixture
def synthetic_mixed_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provide synthetic dataset with numerical and native categorical columns."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "Age": np.random.randint(20, 60, size=n),
            "Monthly_Salary": np.random.uniform(3000, 15000, size=n),
            "Work_Hours_Per_Week": np.random.uniform(35, 60, size=n),
            "Department": np.random.choice(["IT", "HR", "Sales", "Finance"], size=n),
            "Gender": np.random.choice(["Male", "Female", "Other"], size=n),
            "Job_Title": np.random.choice(["Analyst", "Engineer", "Manager"], size=n),
            "Education_Level": np.random.choice(["Bachelor", "Master", "PhD"], size=n),
        }
    )
    # Binary imbalanced target (~15% positive)
    y = pd.Series(np.random.choice([0, 1], size=n, p=[0.85, 0.15]), name="Resigned")
    return df, y


# =============================================================================
# 1. Strict Schema & Config Integrity Tests
# =============================================================================

def test_catboost_yaml_loads_in_strict_schema_mode(catboost_raw_config: dict):
    """1. Verify that catboost.yaml loads and validates cleanly under strict schema mode."""
    validate_model_config(catboost_raw_config, strict_schema=True, model_id="catboost")
    assert catboost_raw_config["model"]["model_id"] == "catboost"
    assert catboost_raw_config["model"]["implementation"] == "catboost.CatBoostClassifier"
    assert catboost_raw_config["model"]["config_version"] == "2.0.0"
    assert "inner_validation" in catboost_raw_config
    assert "candidates" in catboost_raw_config


def test_catboost_yaml_has_no_todo_or_unresolved_search_ranges(catboost_raw_config: dict):
    """2. Verify that catboost.yaml contains no unresolved TODO tokens or legacy search ranges."""
    todos = _detect_unresolved_placeholders(catboost_raw_config)
    assert len(todos) == 0, f"Found unresolved placeholders in catboost.yaml: {todos}"
    assert "search_space" not in catboost_raw_config, "Legacy 'search_space' must be replaced by 'candidates'."


def test_catboost_candidate_ids_unique_and_resolvable(catboost_raw_config: dict):
    """3. Verify that all candidate IDs in catboost.yaml are unique and deterministically resolvable."""
    candidates = catboost_raw_config["candidates"]
    candidate_ids = [c["id"] for c in candidates]
    assert len(candidate_ids) == len(set(candidate_ids)), "Candidate IDs must be strictly unique."
    assert len(candidate_ids) >= 2, "Candidate registry must define at least two candidate configurations."

    for cid in candidate_ids:
        resolved = resolve_candidate(catboost_raw_config, candidate_id=cid)
        assert resolved["candidate_id"] == cid
        assert "depth" in resolved["params"]
        assert "learning_rate" in resolved["params"]
        assert "loss_function" in resolved["params"]  # From fixed_params


def test_catboost_unweighted_and_balanced_strategies_resolve_correctly(catboost_raw_config: dict):
    """4. Verify that unweighted (null) and Balanced auto_class_weights resolve correctly."""
    # Find at least one unweighted candidate and at least one balanced candidate
    unweighted_found = False
    balanced_found = False

    for cand in catboost_raw_config["candidates"]:
        cid = cand["id"]
        resolved = resolve_candidate(catboost_raw_config, candidate_id=cid)
        acw = resolved["params"].get("auto_class_weights")
        if acw is None:
            unweighted_found = True
        elif acw == "Balanced":
            balanced_found = True

    assert unweighted_found, "Registry must define an unweighted (auto_class_weights=null) candidate."
    assert balanced_found, "Registry must define a Balanced (auto_class_weights='Balanced') candidate."


# =============================================================================
# 2. Adapter Initialization & Methodology Tests
# =============================================================================

def test_catboost_adapter_accepts_resolved_config(catboost_raw_config: dict):
    """5. Verify that CatBoostAdapter accepts an already resolved configuration mapping."""
    resolved = resolve_candidate(catboost_raw_config, candidate_id="C01")
    adapter = CatBoostAdapter(config=resolved)

    assert isinstance(adapter, ModelAdapter)
    assert adapter.name == "CatBoost Classifier"
    assert adapter.model_id == "catboost"
    assert adapter.candidate_id == "C01"
    assert adapter.params["depth"] == resolved["params"]["depth"]
    assert adapter.params["learning_rate"] == resolved["params"]["learning_rate"]
    assert adapter.inner_val_fraction == 0.15
    assert adapter.early_stopping_rounds == 50
    assert adapter.refit_outer_train is True


def test_catboost_inner_split_contains_only_rows_supplied_to_fit(
    catboost_raw_config: dict, synthetic_mixed_data
):
    """6. Verify that inner validation split derives strictly from rows supplied to fit()."""
    X, y = synthetic_mixed_data
    resolved = resolve_candidate(catboost_raw_config, candidate_id="C01")
    adapter = CatBoostAdapter(
        config=resolved,
        params={**resolved["params"], "iterations": 20},
        early_stopping_rounds=5,
    )
    adapter.fit(X, y)

    meta = adapter.get_metadata()
    inner_eval = meta["inner_evaluation"]
    assert "inner_train_size" in inner_eval
    assert "inner_val_size" in inner_eval
    assert inner_eval["inner_train_size"] + inner_eval["inner_val_size"] == len(X)
    assert inner_eval["inner_val_size"] == int(len(X) * adapter.inner_val_fraction)


def test_catboost_outer_validation_never_used_internally():
    """7. Verify that adapter.fit() contract never accepts or references outer validation data."""
    sig = inspect.signature(CatBoostAdapter.fit)
    params = list(sig.parameters.keys())
    assert "eval_set" not in params, "CatBoostAdapter.fit must not expose an eval_set argument."
    assert "X_val" not in params, "CatBoostAdapter.fit must not expose an X_val argument."
    assert "y_val" not in params, "CatBoostAdapter.fit must not expose an y_val argument."


def test_catboost_selected_tree_count_captured(catboost_raw_config: dict, synthetic_mixed_data):
    """8. Verify that inner early-stopping captures the selected tree count."""
    X, y = synthetic_mixed_data
    resolved = resolve_candidate(catboost_raw_config, candidate_id="C01")
    adapter = CatBoostAdapter(
        config=resolved,
        params={**resolved["params"], "iterations": 50},
        early_stopping_rounds=5,
    )
    adapter.fit(X, y)

    assert adapter.selected_tree_count_ is not None
    assert isinstance(adapter.selected_tree_count_, int)
    assert adapter.selected_tree_count_ > 0
    assert adapter.selected_tree_count_ <= 50

    meta = adapter.get_metadata()
    assert meta["selected_tree_count"] == adapter.selected_tree_count_


def test_catboost_final_refit_uses_all_outer_training_rows_and_fixed_tree_count(
    catboost_raw_config: dict, synthetic_mixed_data
):
    """9. Verify that final refit trains on 100% of outer rows using the selected fixed tree count."""
    X, y = synthetic_mixed_data
    resolved = resolve_candidate(catboost_raw_config, candidate_id="C01")
    adapter = CatBoostAdapter(
        config=resolved,
        params={**resolved["params"], "iterations": 50},
        early_stopping_rounds=5,
        refit_outer_train=True,
    )
    adapter.fit(X, y)

    assert adapter._fitted_on_all_outer_rows is True
    assert adapter._model.tree_count_ == adapter.selected_tree_count_

    meta = adapter.get_metadata()
    assert meta["fitted_on_all_outer_rows"] is True


def test_catboost_native_categorical_columns_preserved(
    catboost_raw_config: dict, synthetic_mixed_data
):
    """10. Verify that categorical columns are preserved natively and passed as cat_features."""
    X, y = synthetic_mixed_data
    resolved = resolve_candidate(catboost_raw_config, candidate_id="C01")
    adapter = CatBoostAdapter(
        config=resolved,
        params={**resolved["params"], "iterations": 10},
        categorical_features=["Department", "Gender", "Job_Title", "Education_Level"],
        early_stopping_rounds=None,
        inner_val_fraction=0.0,
    )
    adapter.fit(X, y)

    expected_indices = [
        i for i, col in enumerate(X.columns)
        if col in ["Department", "Gender", "Job_Title", "Education_Level"]
    ]
    actual_indices = adapter._model.get_cat_feature_indices()
    assert sorted(actual_indices) == sorted(expected_indices)


def test_catboost_predict_proba_produces_valid_probabilities(
    catboost_raw_config: dict, synthetic_mixed_data
):
    """11. Verify that predict_proba produces valid [0, 1] calibrated/positive probabilities."""
    X, y = synthetic_mixed_data
    resolved = resolve_candidate(catboost_raw_config, candidate_id="C01")
    adapter = CatBoostAdapter(
        config=resolved,
        params={**resolved["params"], "iterations": 15},
        early_stopping_rounds=None,
        inner_val_fraction=0.0,
    )
    adapter.fit(X, y)

    probs = adapter.predict_proba(X)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(X),)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert not np.any(np.isnan(probs))


def test_catboost_direct_params_unit_test_mode_works(synthetic_mixed_data):
    """12. Verify that direct-params construction for unit testing continues to work."""
    X, y = synthetic_mixed_data
    direct_params = {
        "iterations": 10,
        "depth": 3,
        "learning_rate": 0.1,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    adapter = CatBoostAdapter(params=direct_params)
    assert adapter.candidate_id == "direct"
    assert adapter.params["depth"] == 3

    adapter.fit(X, y)
    probs = adapter.predict_proba(X)
    assert len(probs) == len(X)
    assert adapter.get_metadata()["fitted"] is True


# =============================================================================
# 3. Selection Evidence Preservation Tests (Step 2B.6)
# =============================================================================

def test_catboost_selection_evidence_api_exists():
    """13. Verify get_selection_evidence exists and is callable."""
    assert hasattr(CatBoostAdapter, "get_selection_evidence")
    assert callable(getattr(CatBoostAdapter, "get_selection_evidence"))


def test_catboost_selection_evidence_fails_before_fit():
    """14. Verify get_selection_evidence raises RuntimeError before fit()."""
    adapter = CatBoostAdapter()
    with pytest.raises(RuntimeError, match="not fitted"):
        adapter.get_selection_evidence()


def test_catboost_selection_evidence_multi_candidate_tuning(
    catboost_raw_config: dict, synthetic_mixed_data
):
    """15-20. Verify comprehensive selection evidence capture across candidate pool."""
    X, y = synthetic_mixed_data

    # Create a fast 2-candidate configuration for testing
    fast_config = copy.deepcopy(catboost_raw_config)
    fast_config["candidates"] = [
        {
            "id": "C01",
            "description": "Fast C01",
            "params": {"depth": 4, "learning_rate": 0.1, "iterations": 10},
        },
        {
            "id": "C02",
            "description": "Fast C02",
            "params": {"depth": 6, "learning_rate": 0.05, "iterations": 10},
        },
    ]

    adapter = CatBoostAdapter(
        config=fast_config,
        candidate_id="auto",
        early_stopping_rounds=3,
        inner_val_fraction=0.20,
        refit_outer_train=True,
    )

    adapter.fit(X, y)

    evidence = adapter.get_selection_evidence()

    # Winning candidate
    assert evidence["winning_candidate_id"] in ["C01", "C02"]
    assert evidence["winning_candidate"] == evidence["winning_candidate_id"]
    assert isinstance(evidence["winning_inner_ap"], float)
    assert 0.0 <= evidence["winning_inner_ap"] <= 1.0

    # All configured candidates present
    assert "all_candidates" in evidence
    all_cands = evidence["all_candidates"]
    assert "C01" in all_cands
    assert "C02" in all_cands

    # Candidate AP values retained
    for cid in ["C01", "C02"]:
        assert "inner_ap" in all_cands[cid]
        assert "ap_score" in all_cands[cid]
        assert isinstance(all_cands[cid]["inner_ap"], float)
        assert 0.0 <= all_cands[cid]["inner_ap"] <= 1.0

    # Selected tree counts retained
    assert isinstance(evidence["selected_tree_count"], int)
    assert evidence["selected_tree_count"] > 0
    for cid in ["C01", "C02"]:
        assert "selected_trees" in all_cands[cid]
        assert "selected_tree_count" in all_cands[cid]
        assert isinstance(all_cands[cid]["selected_trees"], int)
        assert all_cands[cid]["selected_trees"] > 0

    # Non-negative finite runtimes
    assert isinstance(evidence["candidate_selection_time_seconds"], float)
    assert np.isfinite(evidence["candidate_selection_time_seconds"])
    assert evidence["candidate_selection_time_seconds"] >= 0.0

    assert isinstance(evidence["refit_time_seconds"], float)
    assert np.isfinite(evidence["refit_time_seconds"])
    assert evidence["refit_time_seconds"] >= 0.0

    # Defensive deep copy
    evidence["mutated_key"] = "test_mutation"
    evidence["all_candidates"]["C01"]["mutated"] = True
    second_evidence = adapter.get_selection_evidence()
    assert "mutated_key" not in second_evidence
    assert "mutated" not in second_evidence["all_candidates"]["C01"]

    # Outer and inner row metadata
    assert evidence["inner_train_rows"] + evidence["inner_val_rows"] == len(X)
    assert evidence["inner_val_fraction"] == 0.20
    assert evidence["fitted_on_all_outer_rows"] is True

