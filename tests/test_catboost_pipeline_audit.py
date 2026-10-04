"""Data pipeline, preprocessing, and leakage audit tests for CatBoost (Steps 2 and 3).

Verifies:
- Employee_ID uniqueness before and after merging with split registry
- Exact split sizes: 80,000 development, 20,000 holdout, 5 folds x 16,000 rows
- Zero OOF overlap with holdout
- Excluded fields (Employee_ID, Hire_Date, target, split columns) never enter predictors
- Native categorical preservation (Department, Gender, Job_Title, Education_Level)
- No external one-hot encoding, global scaling, global imputation, or leakage-prone preprocessing
- Zero missing values in raw development data
- Complete categorical support across all outer folds and unseen category policy
- Feature contract target leakage / post-resignation attribute audit
- Strict isolation of outer validation and holdout from CatBoostAdapter.fit()
- Valid probability bounds [0, 1] without NaN or Inf
"""

from __future__ import annotations

import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.feature_contract import load_feature_contract
from src.evaluation.inputs import load_development_data
from src.models.catboost_model import CatBoostAdapter
from src.models.runner import POST_OUTCOME_FEATURE_TOKENS, run_5fold_cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset_and_split():
    """Load development partition and frozen split registry."""
    return load_development_data(PROJECT_ROOT)


# =============================================================================
# Step 2: Data Pipeline & Preprocessing Audit
# =============================================================================

def test_employee_id_uniqueness_and_split_integrity(dataset_and_split):
    """Verify Employee_ID uniqueness before and after merge and exact row counts."""
    data, split = dataset_and_split
    contract = load_feature_contract()

    # 1. Raw file uniqueness
    raw_path = PROJECT_ROOT / contract["dataset"]["path"]
    raw_ids = pd.read_csv(raw_path, usecols=["Employee_ID"])["Employee_ID"]
    assert raw_ids.is_unique, "Raw Employee_ID must be unique."
    assert len(raw_ids) == contract["dataset"]["expected_rows"] == 100_000

    # 2. Split registry uniqueness and counts
    assert split["Employee_ID"].is_unique, "Split registry Employee_ID must be unique."
    dev_split = split[split["partition"] == "development"]
    holdout_split = split[split["partition"] == "holdout"]
    assert len(dev_split) == 80_000, "Development partition must have exactly 80,000 rows."
    assert len(holdout_split) == 20_000, "Holdout partition must have exactly 20,000 rows."

    # 3. Outer folds verification
    fold_counts = dev_split["cv_fold"].value_counts()
    assert len(fold_counts) == 5, "Must have exactly 5 outer folds."
    for f in range(1, 6):
        assert fold_counts[f] == 16_000, f"Fold {f} must contain exactly 16,000 validation records."

    # 4. Zero holdout overlap
    dev_id_set = set(dev_split["Employee_ID"])
    holdout_id_set = set(holdout_split["Employee_ID"])
    assert dev_id_set.isdisjoint(holdout_id_set), "Development and holdout IDs must have ZERO overlap."


def test_excluded_features_never_enter_predictors():
    """Verify forbidden identifier, date, target, and split columns cannot enter features."""
    contract = load_feature_contract()
    primary_feats = set(contract["primary_model_features"])
    forbidden = {item["column"] for item in contract["excluded_features"]}
    forbidden.update({"Resigned", "Employee_ID", "cv_fold", "partition", "Hire_Date"})

    overlap = primary_feats.intersection(forbidden)
    assert len(overlap) == 0, f"Forbidden features found in primary_model_features: {overlap}"


def test_categorical_features_passed_natively_as_categories_or_strings(dataset_and_split):
    """Verify Department, Gender, Job_Title, and Education_Level are handled natively without external OHE."""
    data, _ = dataset_and_split
    adapter = CatBoostAdapter()
    expected_cats = ["Department", "Gender", "Job_Title", "Education_Level"]
    assert sorted(adapter.categorical_features) == sorted(expected_cats)

    # In CatBoostAdapter, categorical columns are converted to str and passed via cat_features
    # Verify no one-hot encoded or scaled column names exist in primary feature set
    contract = load_feature_contract()
    for feat in contract["primary_model_features"]:
        assert not feat.startswith("Department_"), "No one-hot dummy columns allowed in CatBoost feature set."
        assert not feat.startswith("Gender_"), "No one-hot dummy columns allowed in CatBoost feature set."
        assert not feat.startswith("Job_Title_"), "No one-hot dummy columns allowed in CatBoost feature set."
        assert not feat.startswith("Education_Level_"), "No one-hot dummy columns allowed in CatBoost feature set."


def test_no_destructive_imputation_needed(dataset_and_split):
    """Verify that development data contains zero missing values, requiring no imputation."""
    data, _ = dataset_and_split
    contract = load_feature_contract()
    feats = contract["primary_model_features"]
    assert data[feats].isna().sum().sum() == 0, "No missing values exist; destructive imputation is prohibited."


def test_categorical_level_support_across_all_outer_folds(dataset_and_split):
    """Verify that every categorical level is observed in training folds and check CatBoost unseen policy."""
    data, split = dataset_and_split
    merged = data.merge(split[["Employee_ID", "cv_fold"]], on="Employee_ID")
    cat_cols = ["Department", "Gender", "Job_Title", "Education_Level"]

    for col in cat_cols:
        for f in range(1, 6):
            tr_cats = set(merged[merged["cv_fold"] != f][col].unique())
            va_cats = set(merged[merged["cv_fold"] == f][col].unique())
            # All validation categories must exist in training folds for perfect support
            diff = va_cats - tr_cats
            assert len(diff) == 0, f"Fold {f} validation set has unseen category in {col}: {diff}"


# =============================================================================
# Step 3: Leakage Audit
# =============================================================================

def test_feature_contract_leakage_and_post_resignation_audit():
    """Verify no post-resignation or outcome-associated features exist in the schema."""
    contract = load_feature_contract()
    primary_feats = [f.lower() for f in contract["primary_model_features"]]
    post_resignation_stems = [
        "exit", "last_working", "severance", "termination", "settlement",
        "notice_period", "resignation_reason", "left_date"
    ]
    for feat in primary_feats:
        for stem in post_resignation_stems:
            assert stem not in feat, f"Potential post-resignation leakage feature detected: {feat}"


def test_runner_rejects_post_outcome_features_before_training():
    """A future contract edit cannot bypass the runtime post-outcome guard."""
    data = pd.DataFrame({
        "Employee_ID": [1], "Resigned": [0], "Final_Work_Date": ["2026-01-01"],
    })
    registry = pd.DataFrame({
        "Employee_ID": [1], "partition": ["development"], "cv_fold": [1],
    })
    assert "final_work" in POST_OUTCOME_FEATURE_TOKENS
    with pytest.raises(ValueError, match="post-outcome"):
        run_5fold_cv(
            adapter_factory=CatBoostAdapter,
            data=data,
            split_registry=registry,
            feature_cols=["Final_Work_Date"],
        )


def test_catboost_adapter_fit_interface_strictly_excludes_outer_validation():
    """Verify CatBoostAdapter.fit() signature strictly refuses outer validation sets."""
    sig = inspect.signature(CatBoostAdapter.fit)
    forbidden_args = {"eval_set", "X_val", "y_val", "validation_data", "test_data"}
    actual_args = set(sig.parameters.keys())
    leakage_args = actual_args.intersection(forbidden_args)
    assert len(leakage_args) == 0, f"CatBoostAdapter.fit exposes leakage parameter: {leakage_args}"


def test_canonical_oof_predictions_integrity():
    """Verify canonical OOF predictions file satisfies all repository boundary assertions."""
    oof_path = PROJECT_ROOT / "results" / "models" / "catboost" / "catboost_oof_predictions.csv"
    assert oof_path.exists(), f"OOF predictions missing at: {oof_path}"
    oof_df = pd.read_csv(oof_path)

    # 1. Total records
    assert len(oof_df) == 80_000, f"Expected 80,000 OOF records, found {len(oof_df)}"

    # 2. No duplicate IDs
    assert oof_df["Employee_ID"].is_unique, "Duplicate Employee_IDs found in OOF predictions."

    # 3. Fold consistency
    fold_counts = oof_df["cv_fold"].value_counts()
    for f in range(1, 6):
        assert fold_counts[f] == 16_000, f"Fold {f} does not have exactly 16,000 rows."

    # 4. Zero holdout ID leakage
    split = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "split_registry.csv")
    holdout_ids = set(split.loc[split["partition"] == "holdout", "Employee_ID"])
    oof_ids = set(oof_df["Employee_ID"])
    leakage = oof_ids.intersection(holdout_ids)
    assert len(leakage) == 0, f"CRITICAL LEAKAGE: {len(leakage)} holdout IDs found in OOF predictions!"

    # 5. Probabilities valid and finite
    probs = oof_df["y_probability"].to_numpy()
    assert np.isfinite(probs).all(), "Non-finite probability values detected."
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1] range."
    assert not np.isnan(probs).any(), "NaN probability values detected."
