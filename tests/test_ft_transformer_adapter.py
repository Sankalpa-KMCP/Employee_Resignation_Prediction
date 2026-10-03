"""Unit tests for FT-Transformer configuration and FTTransformerAdapter (Step 2).

Verifies strict v2 schema loading, candidate registry resolution (FT01–FT08),
automatic candidate selection, epoch semantics, weighting semantics, preprocessing isolation,
refit on 100% of outer rows, defensive evidence reporting, and prediction validity.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.base import ModelAdapter
from src.models.config import load_model_config, resolve_candidate, validate_model_config
from src.models.ft_transformer import FTTransformerAdapter, _evaluate_candidate, _fit_preprocessors, _transform_features


@pytest.fixture
def ft_raw_config() -> dict:
    """Load authoritative FT-Transformer configuration in strict schema mode."""
    return load_model_config("ft_transformer", strict_schema=True)


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provide small synthetic tabular dataset for testing adapter mechanics."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "Age": np.random.randint(20, 60, size=n),
            "Monthly_Salary": np.random.uniform(3000, 15000, size=n),
            "Work_Hours_Per_Week": np.random.uniform(35, 60, size=n),
            "Department": np.random.choice(["IT", "HR", "Sales", "Finance"], size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
            "Job_Title": np.random.choice(["Analyst", "Engineer", "Manager"], size=n),
            "Education_Level": np.random.choice(["Bachelor", "Master", "PhD"], size=n),
        }
    )
    # Ensure at least 15 positive samples and 85 negative samples
    y_vals = np.array([1] * 15 + [0] * (n - 15))
    np.random.shuffle(y_vals)
    y = pd.Series(y_vals, name="Resigned")
    return df, y


# =============================================================================
# 1. Config Loading & Candidate Resolution Tests
# =============================================================================

def test_ft_transformer_adapter_accepts_strict_v2_config(ft_raw_config: dict):
    """1. Verify adapter accepts strict v2 configuration and initializes attributes."""
    validate_model_config(ft_raw_config, strict_schema=True, model_id="ft_transformer")
    adapter = FTTransformerAdapter(config=ft_raw_config, candidate_id="auto")

    assert isinstance(adapter, ModelAdapter)
    assert adapter.name == "FT-Transformer"
    assert adapter.model_id == "ft_transformer"
    assert adapter.candidate_id == "auto"
    assert adapter.inner_val_fraction == 0.15
    assert adapter.patience == 15
    assert adapter.max_epochs == 100
    assert adapter.refit_outer_train is True
    assert adapter.random_seed == 42
    assert len(adapter._candidate_pool) == 8


def test_ft01_to_ft08_candidates_resolve_correctly(ft_raw_config: dict):
    """2. Verify that candidates FT01 through FT08 resolve correctly with exact specs."""
    expected_ids = [f"FT{i:02d}" for i in range(1, 9)]
    candidates = ft_raw_config["candidates"]
    actual_ids = [c["id"] for c in candidates]
    assert actual_ids == expected_ids

    for cid in expected_ids:
        resolved = resolve_candidate(ft_raw_config, candidate_id=cid)
        assert resolved["candidate_id"] == cid
        params = resolved["params"]
        assert "n_blocks" in params
        assert "d_block" in params
        assert "attention_n_heads" in params
        assert "ffn_d_hidden" in params
        assert "learning_rate" in params
        assert "weighting" in params

    # Verify unweighted vs weighted pairs
    for odd_id, even_id in [("FT01", "FT02"), ("FT03", "FT04"), ("FT05", "FT06"), ("FT07", "FT08")]:
        res_odd = resolve_candidate(ft_raw_config, candidate_id=odd_id)
        res_even = resolve_candidate(ft_raw_config, candidate_id=even_id)
        assert res_odd["params"]["weighting"] == "none", f"{odd_id} must be unweighted"
        assert res_even["params"]["weighting"] == "dynamic_ratio", f"{even_id} must be dynamic_ratio"


# =============================================================================
# 2. Automatic Candidate Selection & Deterministic Winner Selection
# =============================================================================

def test_automatic_selection_evaluates_all_configured_candidates(
    ft_raw_config: dict, synthetic_data
):
    """3. Verify automatic selection evaluates all 8 candidates independently."""
    X, y = synthetic_data
    adapter = FTTransformerAdapter(config=ft_raw_config, candidate_id="auto")

    evaluated_candidates = []

    def mock_evaluate(candidate_dict, **kwargs):
        cid = candidate_dict["id"]
        evaluated_candidates.append(cid)
        return {
            "candidate_id": cid,
            "inner_ap": 0.20 + (0.01 if cid == "FT03" else 0.0),
            "ap_score": 0.20 + (0.01 if cid == "FT03" else 0.0),
            "selected_epoch_count": 5,
            "selected_epochs": 5,
            "weighting": candidate_dict["params"].get("weighting", "none"),
            "pos_weight": None,
            "inner_pos_weight": None,
            "actual_pos_weight": None,
            "fit_time_seconds": 0.05,
            "runtime_seconds": 0.05,
            "params": candidate_dict["params"],
        }

    with patch("src.models.ft_transformer._evaluate_candidate", side_effect=lambda **kw: mock_evaluate(kw["candidate_dict"])):
        with patch.object(adapter, "_refit_fresh_outer_model"):
            adapter.fit(X, y)

    assert evaluated_candidates == [f"FT{i:02d}" for i in range(1, 9)]
    evidence = adapter.get_selection_evidence()
    assert len(evidence["all_candidates"]) == 8


def test_highest_inner_ap_candidate_selected_deterministically(
    ft_raw_config: dict, synthetic_data
):
    """4. Verify candidate with highest inner AP is selected deterministically."""
    X, y = synthetic_data
    adapter = FTTransformerAdapter(config=ft_raw_config, candidate_id="auto")

    # Force FT06 to have highest AP
    ap_scores = {
        "FT01": 0.12, "FT02": 0.14, "FT03": 0.13, "FT04": 0.15,
        "FT05": 0.11, "FT06": 0.35, "FT07": 0.18, "FT08": 0.17,
    }

    with patch(
        "src.models.ft_transformer._evaluate_candidate",
        side_effect=lambda **kw: {
            "candidate_id": kw["candidate_dict"]["id"],
            "inner_ap": ap_scores[kw["candidate_dict"]["id"]],
            "ap_score": ap_scores[kw["candidate_dict"]["id"]],
            "selected_epoch_count": 8,
            "selected_epochs": 8,
            "weighting": kw["candidate_dict"]["params"].get("weighting", "none"),
            "pos_weight": 5.0 if kw["candidate_dict"]["params"].get("weighting") == "dynamic_ratio" else None,
            "inner_pos_weight": 5.0 if kw["candidate_dict"]["params"].get("weighting") == "dynamic_ratio" else None,
            "actual_pos_weight": 5.0 if kw["candidate_dict"]["params"].get("weighting") == "dynamic_ratio" else None,
            "fit_time_seconds": 0.02,
            "runtime_seconds": 0.02,
            "params": kw["candidate_dict"]["params"],
        },
    ):
        with patch.object(adapter, "_refit_fresh_outer_model"):
            adapter.fit(X, y)

    assert adapter.candidate_id == "FT06"
    assert adapter.selected_epoch_count == 8
    evidence = adapter.get_selection_evidence()
    assert evidence["winning_candidate_id"] == "FT06"
    assert evidence["winning_inner_ap"] == 0.35
    assert evidence["selected_weighting"] == "dynamic_ratio"


# =============================================================================
# 3. Weighting Semantics (Candidate Driven, NOT Top-Level Strategy)
# =============================================================================

def test_unweighted_candidate_creates_bce_loss_without_pos_weight(
    ft_raw_config: dict, synthetic_data
):
    """5. Verify unweighted candidate (weighting=='none') uses unweighted BCE without pos_weight."""
    X, y = synthetic_data
    resolved = resolve_candidate(ft_raw_config, candidate_id="FT01")
    assert resolved["params"]["weighting"] == "none"

    adapter = FTTransformerAdapter(config=resolved, candidate_id="FT01", max_epochs=1, patience=1)

    import torch.nn as nn
    with patch("torch.nn.BCEWithLogitsLoss", wraps=nn.BCEWithLogitsLoss) as mock_bce:
        adapter.fit(X, y)

        mock_bce.assert_called()
        for call in mock_bce.call_args_list:
            call_kwargs = call.kwargs
            assert "pos_weight" not in call_kwargs or call_kwargs["pos_weight"] is None
        assert adapter._pos_weight is None


def test_dynamic_ratio_candidate_derives_pos_weight_from_training_labels_only(
    ft_raw_config: dict, synthetic_data
):
    """6. Verify dynamic_ratio candidate derives pos_weight strictly from training rows."""
    X, y = synthetic_data
    resolved = resolve_candidate(ft_raw_config, candidate_id="FT02")
    assert resolved["params"]["weighting"] == "dynamic_ratio"

    adapter = FTTransformerAdapter(config=resolved, candidate_id="FT02", max_epochs=1, patience=1)
    adapter.fit(X, y)

    # In synthetic data, y has exactly 85 zeros and 15 ones
    expected_ratio = float(np.sum(y == 0) / np.sum(y == 1))
    assert adapter._pos_weight == pytest.approx(expected_ratio, rel=1e-5)
    # Never hardcoded to 8.99
    assert adapter._pos_weight != 8.99


def test_weighting_driven_by_candidate_params_not_top_level_strategy(
    ft_raw_config: dict, synthetic_data
):
    """17. Verify weighting is governed strictly by candidate params['weighting']."""
    assert ft_raw_config["class_imbalance"]["strategy"] == "dynamic_ratio"

    # Candidate FT01 is 'none' despite top-level 'dynamic_ratio'
    res_ft01 = resolve_candidate(ft_raw_config, candidate_id="FT01")
    assert res_ft01["params"]["weighting"] == "none"

    adapter_ft01 = FTTransformerAdapter(config=res_ft01, candidate_id="FT01")
    with patch.object(adapter_ft01, "_refit_fresh_outer_model"):
        with patch("src.models.ft_transformer._evaluate_candidate", return_value={
            "candidate_id": "FT01", "inner_ap": 0.2, "ap_score": 0.2, "selected_epoch_count": 2,
            "selected_epochs": 2, "weighting": "none", "pos_weight": None, "inner_pos_weight": None,
            "actual_pos_weight": None, "fit_time_seconds": 0.01, "runtime_seconds": 0.01, "params": res_ft01["params"],
        }):
            adapter_ft01.fit(synthetic_data[0], synthetic_data[1])

    evidence_ft01 = adapter_ft01.get_selection_evidence()
    assert evidence_ft01["selected_weighting"] == "none"
    assert evidence_ft01["selected_pos_weight"] is None


# =============================================================================
# 4. Outer Refit Semantics & Preprocessing Isolation
# =============================================================================

def test_full_refit_uses_all_outer_training_rows(
    ft_raw_config: dict, synthetic_data
):
    """7 & 8. Verify full outer refit trains on 100% of outer rows, including inner-val rows."""
    X, y = synthetic_data
    total_rows = len(X)
    adapter = FTTransformerAdapter(config=ft_raw_config, candidate_id="auto")

    captured_refit_lengths = []

    def mock_refit(X_outer, y_outer, **kwargs):
        captured_refit_lengths.append((len(X_outer), len(y_outer)))
        adapter._model = MagicMock()
        adapter._fitted_on_all_outer_rows = True
        adapter._epochs_trained = kwargs.get("selected_epochs", 1)

    with patch("src.models.ft_transformer._evaluate_candidate", return_value={
        "candidate_id": "FT01", "inner_ap": 0.3, "ap_score": 0.3, "selected_epoch_count": 4,
        "selected_epochs": 4, "weighting": "none", "pos_weight": None, "inner_pos_weight": None,
        "actual_pos_weight": None, "fit_time_seconds": 0.01, "runtime_seconds": 0.01, "params": {},
    }):
        with patch.object(adapter, "_refit_fresh_outer_model", side_effect=mock_refit):
            adapter.fit(X, y)

    assert len(captured_refit_lengths) == 1
    assert captured_refit_lengths[0] == (total_rows, total_rows)
    assert adapter._fitted_on_all_outer_rows is True


def test_final_preprocessing_is_refitted_on_all_outer_rows(
    ft_raw_config: dict, synthetic_data
):
    """9. Verify final scaler and category maps are refitted on all outer-training rows."""
    X, y = synthetic_data
    adapter = FTTransformerAdapter(config=ft_raw_config, candidate_id="auto", max_epochs=1, patience=1)

    with patch("src.models.ft_transformer._evaluate_candidate", return_value={
        "candidate_id": "FT01", "inner_ap": 0.3, "ap_score": 0.3, "selected_epoch_count": 1,
        "selected_epochs": 1, "weighting": "none", "pos_weight": None, "inner_pos_weight": None,
        "actual_pos_weight": None, "fit_time_seconds": 0.01, "runtime_seconds": 0.01, "params": {},
    }):
        adapter.fit(X, y)

    assert adapter._scaler is not None
    # StandardScaler.n_samples_seen_ reflects all 100 rows
    assert adapter._scaler.n_samples_seen_ == len(X)
    assert len(adapter._feature_names) == len(X.columns)


# =============================================================================
# 5. Epoch Semantics & Off-by-One Guard
# =============================================================================

def test_selected_epoch_count_is_positive(ft_raw_config: dict, synthetic_data):
    """10. Verify selected_epoch_count is always a positive integer >= 1."""
    X, y = synthetic_data
    adapter = FTTransformerAdapter(config=ft_raw_config, candidate_id="auto")

    with patch("src.models.ft_transformer._evaluate_candidate", return_value={
        "candidate_id": "FT01", "inner_ap": 0.3, "ap_score": 0.3, "selected_epoch_count": 1,
        "selected_epochs": 1, "weighting": "none", "pos_weight": None, "inner_pos_weight": None,
        "actual_pos_weight": None, "fit_time_seconds": 0.01, "runtime_seconds": 0.01, "params": {},
    }):
        with patch.object(adapter, "_refit_fresh_outer_model"):
            adapter.fit(X, y)

    assert adapter.selected_epoch_count >= 1
    assert isinstance(adapter.selected_epoch_count, int)


def test_explicit_off_by_one_epoch_guard(ft_raw_config: dict, synthetic_data):
    """11. Explicit off-by-one test: best index 0 / epoch 1 -> refit executes exactly 1 epoch."""
    X, y = synthetic_data
    adapter = FTTransformerAdapter(config=ft_raw_config, candidate_id="auto")

    refit_epochs_executed = []

    def mock_refit(**kwargs):
        refit_epochs_executed.append(kwargs["selected_epochs"])
        adapter._model = MagicMock()
        adapter._fitted_on_all_outer_rows = True
        adapter._epochs_trained = kwargs["selected_epochs"]

    with patch("src.models.ft_transformer._evaluate_candidate", return_value={
        "candidate_id": "FT01", "inner_ap": 0.3, "ap_score": 0.3, "selected_epoch_count": 1,
        "selected_epochs": 1, "weighting": "none", "pos_weight": None, "inner_pos_weight": None,
        "actual_pos_weight": None, "fit_time_seconds": 0.01, "runtime_seconds": 0.01, "params": {},
    }):
        with patch.object(adapter, "_refit_fresh_outer_model", side_effect=lambda **kw: mock_refit(**kw)):
            adapter.fit(X, y)

    assert refit_epochs_executed == [1]
    assert adapter.selected_epoch_count == 1
    assert adapter.selected_epoch_count_ == 1


# =============================================================================
# 6. Refit Characteristics (No Early Stopping, No Validation Set)
# =============================================================================

def test_refit_does_not_use_early_stopping_or_validation_set(
    ft_raw_config: dict, synthetic_data
):
    """12 & 13. Verify outer refit does not take/use a validation set or early stopping."""
    import inspect
    from src.models.ft_transformer import FTTransformerAdapter

    sig = inspect.signature(FTTransformerAdapter._refit_fresh_outer_model)
    param_names = list(sig.parameters.keys())

    assert "X_val" not in param_names
    assert "y_val" not in param_names
    assert "eval_set" not in param_names
    assert "patience" not in param_names


# =============================================================================
# 7. Candidate-Selection Evidence & Diagnostics
# =============================================================================

def test_candidate_selection_evidence_contains_all_ft01_to_ft08(
    ft_raw_config: dict, synthetic_data
):
    """14. Verify candidate-selection evidence contains all FT01–FT08 with required keys."""
    X, y = synthetic_data
    adapter = FTTransformerAdapter(config=ft_raw_config, candidate_id="auto")

    with patch("src.models.ft_transformer._evaluate_candidate", side_effect=lambda **kw: {
        "candidate_id": kw["candidate_dict"]["id"],
        "inner_ap": 0.25,
        "ap_score": 0.25,
        "selected_epoch_count": 3,
        "selected_epochs": 3,
        "weighting": kw["candidate_dict"]["params"].get("weighting", "none"),
        "pos_weight": 5.0 if kw["candidate_dict"]["params"].get("weighting") == "dynamic_ratio" else None,
        "inner_pos_weight": 5.0 if kw["candidate_dict"]["params"].get("weighting") == "dynamic_ratio" else None,
        "actual_pos_weight": 5.0 if kw["candidate_dict"]["params"].get("weighting") == "dynamic_ratio" else None,
        "fit_time_seconds": 0.01,
        "runtime_seconds": 0.01,
        "params": kw["candidate_dict"]["params"],
    }):
        with patch.object(adapter, "_refit_fresh_outer_model"):
            adapter.fit(X, y)

    evidence = adapter.get_selection_evidence()
    assert "winning_candidate_id" in evidence
    assert "winning_inner_ap" in evidence
    assert "selected_epoch_count" in evidence
    assert "inner_train_rows" in evidence
    assert "inner_validation_rows" in evidence
    assert "inner_validation_fraction" in evidence
    assert "random_seed" in evidence
    assert "selected_weighting" in evidence
    assert "candidate_selection_time_seconds" in evidence
    assert "refit_time_seconds" in evidence
    assert evidence["fitted_on_all_outer_rows"] is True
    assert evidence["metric"] == "average_precision"

    all_cands = evidence["all_candidates"]
    expected_ids = {f"FT{i:02d}" for i in range(1, 9)}
    assert set(all_cands.keys()) == expected_ids

    for cid, c_info in all_cands.items():
        assert c_info["candidate_id"] == cid
        assert "inner_ap" in c_info
        assert "selected_epoch_count" in c_info
        assert "weighting" in c_info
        assert "actual_pos_weight" in c_info
        assert "fit_time_seconds" in c_info
        assert "params" in c_info


def test_evidence_is_defensive_copy(ft_raw_config: dict, synthetic_data):
    """15. Verify that modifying the dictionary from get_selection_evidence does not corrupt internal state."""
    X, y = synthetic_data
    adapter = FTTransformerAdapter(config=ft_raw_config, candidate_id="auto")

    with patch("src.models.ft_transformer._evaluate_candidate", return_value={
        "candidate_id": "FT01", "inner_ap": 0.3, "ap_score": 0.3, "selected_epoch_count": 2,
        "selected_epochs": 2, "weighting": "none", "pos_weight": None, "inner_pos_weight": None,
        "actual_pos_weight": None, "fit_time_seconds": 0.01, "runtime_seconds": 0.01, "params": {},
    }):
        with patch.object(adapter, "_refit_fresh_outer_model"):
            adapter.fit(X, y)

    ev1 = adapter.get_selection_evidence()
    ev1["winning_candidate_id"] = "CORRUPTED_ID"
    ev1["all_candidates"]["FT01"]["inner_ap"] = -999.0

    ev2 = adapter.get_selection_evidence()
    assert ev2["winning_candidate_id"] != "CORRUPTED_ID"
    if "FT01" in ev2["all_candidates"]:
        assert ev2["all_candidates"]["FT01"]["inner_ap"] != -999.0


def test_get_selection_evidence_fails_before_fit():
    """16. Verify calling get_selection_evidence before fit raises RuntimeError."""
    adapter = FTTransformerAdapter()
    with pytest.raises(RuntimeError, match="not fitted"):
        adapter.get_selection_evidence()


def test_direct_fit_metadata_has_no_selection_scheduler(synthetic_data):
    X, y = synthetic_data
    adapter = FTTransformerAdapter(params={"max_epochs": 1}, inner_val_fraction=0)
    with patch.object(adapter, "_refit_fresh_outer_model"):
        adapter.fit(X, y)
    assert adapter.get_metadata()["selection_learning_rate_scheduler"] is None


def test_fixed_ft02_metadata_matches_fresh_inner_training_fit(ft_raw_config, synthetic_data):
    """Epoch selection must not be reported as restoring the selection model."""
    X, y = synthetic_data
    adapter = FTTransformerAdapter(
        config=ft_raw_config, candidate_id="FT02", refit_outer_train=False,
    )
    evaluation = {"inner_ap": 0.25, "selected_epoch_count": 3}
    with patch("src.models.ft_transformer._evaluate_candidate", return_value=evaluation) as select:
        with patch.object(adapter, "_refit_fresh_outer_model") as fresh_fit:
            adapter.fit(X, y)

    assert select.call_count == 1
    assert select.call_args.kwargs["candidate_dict"]["id"] == "FT02"
    assert fresh_fit.call_count == 1
    assert fresh_fit.call_args.kwargs["selected_epochs"] == 3
    inner_rows = select.call_args.kwargs["X_in_tr"]
    pd.testing.assert_frame_equal(fresh_fit.call_args.kwargs["X_outer"], inner_rows)
    assert len(inner_rows) < len(X)
    metadata = adapter.get_metadata()
    assert metadata["final_fit_partition"] == "inner_training"
    assert metadata["fitted_on_all_outer_rows"] is False
    assert metadata["best_weights_restored"] is False
    assert metadata["selection_checkpoint_used_for_final_predictions"] is False
    assert metadata["selection_learning_rate_scheduler"] == "ReduceLROnPlateau"
    assert metadata["final_fit_learning_rate_scheduler"] is None


# =============================================================================
# 8. Inference Validity & Direct Compatibility
# =============================================================================

def test_predictions_remain_finite_probabilities_in_unit_interval(
    ft_raw_config: dict, synthetic_data
):
    """18. Verify predict_proba returns finite 1D probabilities strictly in [0, 1]."""
    X, y = synthetic_data
    # Fast 1-epoch single candidate fit
    resolved = resolve_candidate(ft_raw_config, candidate_id="FT03")
    adapter = FTTransformerAdapter(
        config=resolved,
        candidate_id="FT03",
        max_epochs=1,
        patience=1,
        refit_outer_train=True,
    )
    adapter.fit(X, y)

    probs = adapter.predict_proba(X)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == len(X)
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_existing_direct_and_single_config_compatibility(synthetic_data):
    """19. Verify direct param initialization and zero-argument instantiation remain functional."""
    X, y = synthetic_data

    # Zero-argument instantiation
    adapter_zero = FTTransformerAdapter()
    assert adapter_zero.name == "FT-Transformer"
    assert adapter_zero.model_id == "ft_transformer"
    assert adapter_zero._fitted is False

    meta = adapter_zero.get_metadata()
    assert meta["fitted"] is False

    # Direct param mode
    direct_params = {
        "n_blocks": 2,
        "d_block": 32,
        "attention_n_heads": 2,
        "ffn_d_hidden": 64,
        "learning_rate": 0.001,
        "max_epochs": 1,
        "patience": 1,
        "seed": 42,
        "weighting": "none",
    }
    adapter_direct = FTTransformerAdapter(params=direct_params, refit_outer_train=False)
    adapter_direct.fit(X, y)
    assert adapter_direct._fitted is True
    probs = adapter_direct.predict_proba(X)
    assert len(probs) == len(X)


def test_ft_transformer_adapter_save_and_load_round_trip(synthetic_data, tmp_path):
    """20. Verify serialization save and load round-trip restores model and preprocessing state."""
    X, y = synthetic_data
    direct_params = {
        "n_blocks": 2,
        "d_block": 32,
        "attention_n_heads": 2,
        "ffn_d_hidden": 64,
        "learning_rate": 0.001,
        "max_epochs": 1,
        "patience": 1,
        "seed": 42,
        "weighting": "none",
    }
    adapter = FTTransformerAdapter(params=direct_params, refit_outer_train=False)
    adapter.fit(X, y)
    probs_orig = adapter.predict_proba(X)

    save_dir = tmp_path / "ft_model_test"
    meta = adapter.save(save_dir)
    assert "weights_file" in meta
    assert "preprocessor_file" in meta
    assert (save_dir / "ft_transformer_weights.pt").exists()
    assert (save_dir / "ft_transformer_preprocessor.pkl").exists()

    loaded_adapter = FTTransformerAdapter.load(save_dir)
    assert loaded_adapter._fitted is True
    probs_loaded = loaded_adapter.predict_proba(X)
    np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-5, atol=1e-5)
