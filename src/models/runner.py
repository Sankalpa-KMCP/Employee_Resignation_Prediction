"""Shared five-fold Out-Of-Fold (OOF) cross-validation runner."""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    calculate_binary_metrics,
    probabilities_to_predictions,
)
from src.evaluation.oof_validation import validate_oof_predictions
from src.models.base import ModelAdapter, OOFResult


def run_5fold_cv(
    adapter_factory: Callable[[], ModelAdapter],
    data: pd.DataFrame,
    split_registry: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "Resigned",
    id_col: str = "Employee_ID",
    fold_col: str = "cv_fold",
    partition_col: str = "partition",
    development_label: str = "development",
    holdout_label: str = "holdout",
    expected_folds: Sequence[int] = (1, 2, 3, 4, 5),
    expected_model_name: str | None = None,
) -> OOFResult:
    """Execute leakage-safe 5-fold cross-validation across the development partition.

    This function coordinates out-of-fold evaluation across the 5 frozen
    cross-validation folds defined in the split registry. It enforces strict
    isolation boundaries, creates a fresh independent model adapter instance for
    every fold, validates probability outputs, and produces canonical OOF artifacts.

    Parameters
    ----------
    adapter_factory : Callable[[], ModelAdapter]
        A zero-argument callable returning a newly instantiated, unfitted
        `ModelAdapter` object. A fresh instance is constructed for each fold
        to prevent cross-fold state or preprocessor leakage.
    data : pd.DataFrame
        Source dataset containing development records, `id_col`, `target_col`,
        and all predictors specified in `feature_cols`.
    split_registry : pd.DataFrame
        Standardized split registry defining `id_col`, `partition_col`, and `fold_col`.
    feature_cols : Sequence[str]
        Sequence of approved predictor column names.
    target_col : str, default="Resigned"
        Name of the binary target column.
    id_col : str, default="Employee_ID"
        Name of the unique record identifier column.
    fold_col : str, default="cv_fold"
        Name of the cross-validation fold column.
    partition_col : str, default="partition"
        Name of the split partition column ("development" vs "holdout").
    development_label : str, default="development"
        Value in `partition_col` identifying development partition records.
    holdout_label : str, default="holdout"
        Value in `partition_col` identifying formal holdout records.
    expected_folds : Sequence[int], default=(1, 2, 3, 4, 5)
        Expected outer cross-validation fold numbers.
    expected_model_name : str | None, default=None
        Optional expected model name string to enforce across all fold adapters.

    Returns
    -------
    OOFResult
        Structured container holding assembled OOF predictions, fold-level metrics,
        summary statistics, total execution times, and adapter metadata.

    Raises
    ------
    ValueError
        If inputs violate schema requirements, if formal holdout records are present
        in `data`, if probability outputs are invalid, or if OOF validation fails.
    TypeError
        If `adapter_factory` produces an object that does not conform to `ModelAdapter`.

    Notes
    -----
    **Inner Validation Boundary:** The runner fits each fold adapter using strictly
    the 4 outer-training folds (`train_df[feature_cols]`, `train_df[target_col]`).
    The outer validation fold is NEVER passed to `.fit()` as an evaluation or
    early-stopping set. Adapters requiring validation during training (e.g. for early
    stopping) must partition their own training fold internally.
    """
    # ---------------------------------------------------------
    # 1. Input and Schema Validation
    # ---------------------------------------------------------
    required_registry_cols = {id_col, partition_col, fold_col}
    missing_registry_cols = required_registry_cols - set(split_registry.columns)
    if missing_registry_cols:
        raise ValueError(
            f"Split registry is missing required columns: {sorted(missing_registry_cols)}"
        )

    required_data_cols = {id_col, target_col}.union(set(feature_cols))
    missing_data_cols = required_data_cols - set(data.columns)
    if missing_data_cols:
        raise ValueError(
            f"Data dataframe is missing required columns: {sorted(missing_data_cols)}"
        )

    if data[id_col].isna().any():
        raise ValueError(f"Data contains missing {id_col} values.")

    if not data[id_col].is_unique:
        raise ValueError(f"Data contains duplicate {id_col} values.")

    # ---------------------------------------------------------
    # 2. Strict Formal Holdout Isolation Guard
    # ---------------------------------------------------------
    holdout_rows = split_registry.loc[
        split_registry[partition_col] == holdout_label
    ]
    holdout_ids = set(holdout_rows[id_col])

    data_ids = set(data[id_col])
    leaked_holdout_ids = data_ids.intersection(holdout_ids)
    if leaked_holdout_ids:
        raise ValueError(
            f"CRITICAL LEAKAGE ERROR: Formal holdout records detected in data "
            f"passed to OOF runner ({len(leaked_holdout_ids)} holdout IDs found). "
            "Formal holdout rows must NEVER enter model fitting or OOF evaluation."
        )

    # ---------------------------------------------------------
    # 3. Development Partition Alignment
    # ---------------------------------------------------------
    dev_registry = split_registry.loc[
        split_registry[partition_col] == development_label
    ].copy()
    dev_ids = set(dev_registry[id_col])

    missing_dev_ids = dev_ids - data_ids
    if missing_dev_ids:
        raise ValueError(
            f"Data is missing {len(missing_dev_ids)} development records defined in the registry."
        )

    # Filter data to development partition and attach cv_fold
    dev_data = data.loc[data[id_col].isin(dev_ids)].copy()
    if len(dev_data) != len(dev_registry):
        raise ValueError(
            f"Expected exactly {len(dev_registry)} development records, found {len(dev_data)}."
        )

    # Merge fold assignments from registry (drop existing split columns to avoid merge collisions)
    for col in (fold_col, partition_col):
        if col in dev_data.columns:
            dev_data = dev_data.drop(columns=[col])

    fold_mapping = dev_registry[[id_col, fold_col]].copy()
    fold_mapping[fold_col] = fold_mapping[fold_col].astype(int)
    dev_data = dev_data.merge(fold_mapping, on=id_col, validate="one_to_one")

    observed_folds = sorted(dev_data[fold_col].unique())
    expected_folds_sorted = sorted(expected_folds)
    if observed_folds != expected_folds_sorted:
        raise ValueError(
            f"Observed folds {observed_folds} do not match expected folds {expected_folds_sorted}."
        )

    # ---------------------------------------------------------
    # 4. Out-of-Fold Cross-Validation Iteration
    # ---------------------------------------------------------
    fold_prediction_dfs: list[pd.DataFrame] = []
    fold_metric_records: list[dict[str, Any]] = []
    total_fit_time = 0.0
    total_predict_time = 0.0
    captured_model_name: str | None = None
    captured_model_id: str | None = None
    last_metadata: dict[str, Any] = {}

    for fold_num in expected_folds_sorted:
        # Create a fresh, independent adapter instance for this fold
        adapter = adapter_factory()

        if not isinstance(adapter, ModelAdapter):
            # Also check duck-typing in case Protocol runtime check is skipped
            if not (
                hasattr(adapter, "name")
                and hasattr(adapter, "model_id")
                and callable(getattr(adapter, "fit", None))
                and callable(getattr(adapter, "predict_proba", None))
            ):
                raise TypeError(
                    f"Object created by adapter_factory for fold {fold_num} "
                    f"({type(adapter).__name__}) does not implement ModelAdapter Protocol."
                )

        if expected_model_name is not None and adapter.name != expected_model_name:
            raise ValueError(
                f"Adapter name mismatch on fold {fold_num}: "
                f"expected '{expected_model_name}', found '{adapter.name}'."
            )

        if captured_model_name is None:
            captured_model_name = adapter.name
            captured_model_id = adapter.model_id
        else:
            if adapter.name != captured_model_name:
                raise ValueError(
                    f"Inconsistent adapter name on fold {fold_num}: "
                    f"'{adapter.name}' vs previous '{captured_model_name}'."
                )

        # Partition outer training vs outer validation data
        train_mask = dev_data[fold_col] != fold_num
        val_mask = dev_data[fold_col] == fold_num

        train_df = dev_data.loc[train_mask]
        val_df = dev_data.loc[val_mask]

        train_id_set = set(train_df[id_col])
        val_id_set = set(val_df[id_col])
        if not train_id_set.isdisjoint(val_id_set):
            raise ValueError(
                f"Outer fold {fold_num} training and validation sets overlap: "
                f"{len(train_id_set.intersection(val_id_set))} overlapping IDs."
            )

        X_train = train_df[list(feature_cols)]
        y_train = train_df[target_col]
        X_val = val_df[list(feature_cols)]
        y_val = val_df[target_col]

        # Fit adapter strictly on outer training data
        t0_fit = time.perf_counter()
        adapter.fit(X=X_train, y=y_train)
        fit_duration = time.perf_counter() - t0_fit
        total_fit_time += fit_duration

        # Predict probabilities strictly on outer validation data
        t0_pred = time.perf_counter()
        y_prob_raw = adapter.predict_proba(X=X_val)
        pred_duration = time.perf_counter() - t0_pred
        total_predict_time += pred_duration

        # Validate probability predictions
        y_prob = np.asarray(y_prob_raw, dtype=float)

        if y_prob.ndim != 1:
            raise ValueError(
                f"Fold {fold_num} adapter returned probabilities with ndim={y_prob.ndim}, "
                "expected 1D array of shape (N,)."
            )

        if len(y_prob) != len(val_df):
            raise ValueError(
                f"Fold {fold_num} adapter returned {len(y_prob)} probabilities, "
                f"expected {len(val_df)} matching validation fold size."
            )

        if not np.isfinite(y_prob).all():
            raise ValueError(
                f"Fold {fold_num} adapter returned non-finite probability values."
            )

        if (y_prob < 0.0).any() or (y_prob > 1.0).any():
            raise ValueError(
                f"Fold {fold_num} adapter returned probabilities outside [0.0, 1.0] "
                f"(min={y_prob.min():.4f}, max={y_prob.max():.4f})."
            )

        # Convert probabilities to binary predictions at 0.50 threshold
        y_pred_0_5 = probabilities_to_predictions(y_prob, threshold=0.50)
        y_true_int = y_val.astype(int).to_numpy()

        # Calculate fold evaluation metrics
        fold_metrics = calculate_binary_metrics(
            y_true=y_true_int,
            y_probability=y_prob,
            y_pred=y_pred_0_5,
            threshold=0.50,
        )
        fold_metrics["fold"] = fold_num
        fold_metrics["fit_time_seconds"] = float(fit_duration)
        fold_metrics["predict_time_seconds"] = float(pred_duration)
        fold_metrics["pr_auc"] = fold_metrics["average_precision"]
        fold_metric_records.append(fold_metrics)

        # Record fold prediction dataframe
        fold_df = pd.DataFrame(
            {
                "Employee_ID": val_df[id_col].to_numpy(),
                "cv_fold": fold_num,
                "y_true": y_true_int,
                "model_name": adapter.name,
                "y_probability": y_prob,
                "y_pred_0_5": y_pred_0_5,
            }
        )
        fold_prediction_dfs.append(fold_df)

        if hasattr(adapter, "get_metadata") and callable(adapter.get_metadata):
            last_metadata = adapter.get_metadata()

    # ---------------------------------------------------------
    # 5. Assemble and Validate Full OOF Predictions
    # ---------------------------------------------------------
    oof_predictions = (
        pd.concat(fold_prediction_dfs, ignore_index=True)
        .sort_values("Employee_ID")
        .reset_index(drop=True)
    )

    # Perform rigorous repository OOF validation
    validate_oof_predictions(
        predictions=oof_predictions,
        split_registry=split_registry,
        source_data=dev_data,
        expected_model_name=captured_model_name,
    )

    # ---------------------------------------------------------
    # 6. Calculate Summary & Pooled Metrics
    # ---------------------------------------------------------
    fold_metrics_df = pd.DataFrame(fold_metric_records).sort_values("fold").reset_index(drop=True)

    pooled_metrics = calculate_binary_metrics(
        y_true=oof_predictions["y_true"].to_numpy(),
        y_probability=oof_predictions["y_probability"].to_numpy(),
        y_pred=oof_predictions["y_pred_0_5"].to_numpy(),
        threshold=0.50,
    )

    summary_metrics: dict[str, Any] = {
        "model_name": captured_model_name or "Unknown",
        "model_id": captured_model_id or "unknown",
        "total_development_records": len(oof_predictions),
        "total_folds": len(expected_folds_sorted),
        "pooled_average_precision": pooled_metrics["average_precision"],
        "pooled_pr_auc": pooled_metrics["average_precision"],
        "pooled_roc_auc": pooled_metrics["roc_auc"],
        "pooled_f1": pooled_metrics["f1"],
        "pooled_brier_score": pooled_metrics["brier_score"],
        "cv_mean_average_precision": float(fold_metrics_df["average_precision"].mean()),
        "cv_std_average_precision": float(fold_metrics_df["average_precision"].std()),
        "cv_mean_pr_auc": float(fold_metrics_df["average_precision"].mean()),
        "cv_std_pr_auc": float(fold_metrics_df["average_precision"].std()),
        "cv_mean_roc_auc": float(fold_metrics_df["roc_auc"].mean()),
        "cv_std_roc_auc": float(fold_metrics_df["roc_auc"].std()),
        "cv_mean_f1": float(fold_metrics_df["f1"].mean()),
        "cv_std_f1": float(fold_metrics_df["f1"].std()),
        "total_fit_time_seconds": float(total_fit_time),
        "total_predict_time_seconds": float(total_predict_time),
    }

    return OOFResult(
        model_name=captured_model_name or "Unknown",
        model_id=captured_model_id or "unknown",
        predictions=oof_predictions,
        fold_metrics=fold_metrics_df,
        summary_metrics=summary_metrics,
        fit_time_seconds_total=total_fit_time,
        predict_time_seconds_total=total_predict_time,
        metadata=last_metadata,
    )
