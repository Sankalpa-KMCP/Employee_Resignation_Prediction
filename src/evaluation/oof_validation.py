"""Validation utilities for out-of-fold prediction files."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Employee_ID",
    "cv_fold",
    "y_true",
    "model_name",
    "y_probability",
}


def validate_oof_predictions(
    predictions: pd.DataFrame,
    split_registry: pd.DataFrame,
    source_data: pd.DataFrame | None = None,
    expected_model_name: str | None = None,
) -> None:
    """Validate an OOF prediction dataframe against project contracts.

    Raises
    ------
    ValueError
        If any OOF contract rule is violated.
    """

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    missing_columns = (
        REQUIRED_COLUMNS - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required OOF columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # Registry validation
    # ---------------------------------------------------------

    required_registry_columns = {
        "Employee_ID",
        "partition",
        "cv_fold",
    }

    missing_registry_columns = (
        required_registry_columns
        - set(split_registry.columns)
    )

    if missing_registry_columns:
        raise ValueError(
            "Split registry is missing required columns: "
            f"{sorted(missing_registry_columns)}"
        )

    development_registry = split_registry.loc[
        split_registry["partition"] == "development"
    ].copy()

    holdout_registry = split_registry.loc[
        split_registry["partition"] == "holdout"
    ].copy()

    development_ids = set(
        development_registry["Employee_ID"]
    )

    holdout_ids = set(
        holdout_registry["Employee_ID"]
    )

    # ---------------------------------------------------------
    # Row count
    # ---------------------------------------------------------

    expected_rows = len(development_registry)

    if len(predictions) != expected_rows:
        raise ValueError(
            "OOF predictions must contain exactly one row "
            "for every development record. "
            f"Expected {expected_rows:,}, "
            f"found {len(predictions):,}."
        )

    # ---------------------------------------------------------
    # Employee_ID validation
    # ---------------------------------------------------------

    if predictions["Employee_ID"].isna().any():
        raise ValueError(
            "OOF predictions contain missing Employee_ID values."
        )

    if not predictions["Employee_ID"].is_unique:
        raise ValueError(
            "Each development Employee_ID must appear exactly "
            "once in an OOF prediction file."
        )

    prediction_ids = set(
        predictions["Employee_ID"]
    )

    if prediction_ids != development_ids:
        missing_ids = development_ids - prediction_ids
        unexpected_ids = prediction_ids - development_ids

        raise ValueError(
            "OOF Employee_ID set does not exactly match the "
            "development partition. "
            f"Missing={len(missing_ids)}, "
            f"unexpected={len(unexpected_ids)}."
        )

    if prediction_ids.intersection(holdout_ids):
        raise ValueError(
            "Formal holdout Employee_ID values were found in "
            "the OOF prediction file."
        )

    # ---------------------------------------------------------
    # CV fold validation
    # ---------------------------------------------------------

    if predictions["cv_fold"].isna().any():
        raise ValueError(
            "OOF predictions contain missing cv_fold values."
        )

    numeric_folds = pd.to_numeric(predictions["cv_fold"], errors="coerce")
    if not numeric_folds.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("OOF cv_fold values must be integers 1 through 5.")

    observed_folds = set(
        predictions["cv_fold"]
        .astype(int)
        .unique()
    )

    if observed_folds != {1, 2, 3, 4, 5}:
        raise ValueError(
            "OOF predictions must contain exactly CV folds "
            "1 through 5."
        )

    registry_folds = development_registry[
        ["Employee_ID", "cv_fold"]
    ].copy()

    registry_folds["cv_fold"] = (
        registry_folds["cv_fold"].astype(int)
    )

    prediction_folds = predictions[
        ["Employee_ID", "cv_fold"]
    ].copy()

    prediction_folds["cv_fold"] = (
        prediction_folds["cv_fold"].astype(int)
    )

    fold_check = prediction_folds.merge(
        registry_folds,
        on="Employee_ID",
        suffixes=("_prediction", "_registry"),
        validate="one_to_one",
    )

    fold_mismatches = (
        fold_check["cv_fold_prediction"]
        != fold_check["cv_fold_registry"]
    )

    if fold_mismatches.any():
        raise ValueError(
            "One or more OOF cv_fold assignments do not match "
            "the frozen split registry."
        )

    # ---------------------------------------------------------
    # Target validation
    # ---------------------------------------------------------

    if predictions["y_true"].isna().any():
        raise ValueError(
            "OOF predictions contain missing y_true values."
        )

    observed_targets = set(
        predictions["y_true"].unique()
    )

    if not observed_targets.issubset({0, 1, False, True}):
        raise ValueError(
            "y_true must contain only binary values 0 and 1."
        )

    # ---------------------------------------------------------
    # Probability validation
    # ---------------------------------------------------------

    probabilities = pd.to_numeric(
        predictions["y_probability"],
        errors="coerce",
    )

    if probabilities.isna().any():
        raise ValueError(
            "OOF predictions contain missing or non-numeric "
            "probabilities."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "OOF predictions contain non-finite probabilities."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "All OOF probabilities must be between 0 and 1."
        )

    # ---------------------------------------------------------
    # Model-name validation
    # ---------------------------------------------------------

    if predictions["model_name"].isna().any():
        raise ValueError(
            "OOF predictions contain missing model_name values."
        )

    model_names = predictions[
        "model_name"
    ].astype(str)

    if (model_names.str.strip() == "").any():
        raise ValueError(
            "model_name cannot be empty."
        )

    unique_model_names = set(model_names)

    if len(unique_model_names) != 1:
        raise ValueError(
            "A single-model OOF prediction file must contain "
            "exactly one model_name."
        )

    if expected_model_name is not None:
        observed_model_name = next(
            iter(unique_model_names)
        )

        if observed_model_name != expected_model_name:
            raise ValueError(
                "Unexpected model_name. "
                f"Expected '{expected_model_name}', "
                f"found '{observed_model_name}'."
            )

    # ---------------------------------------------------------
    # Optional threshold prediction validation
    # ---------------------------------------------------------

    if "y_pred_0_5" in predictions.columns:

        y_pred = predictions[
            "y_pred_0_5"
        ]

        if y_pred.isna().any():
            raise ValueError(
                "y_pred_0_5 contains missing values."
            )

        if not set(
            y_pred.unique()
        ).issubset({0, 1, False, True}):
            raise ValueError(
                "y_pred_0_5 must contain only binary values."
            )

        expected_predictions = (
            probabilities >= 0.50
        ).astype(int)

        if not np.array_equal(
            y_pred.astype(int).to_numpy(),
            expected_predictions.to_numpy(),
        ):
            raise ValueError(
                "y_pred_0_5 does not match a 0.50 probability "
                "threshold."
            )

    # ---------------------------------------------------------
    # Original target parity
    # ---------------------------------------------------------

    if source_data is not None:

        required_source_columns = {
            "Employee_ID",
            "Resigned",
        }

        missing_source_columns = (
            required_source_columns
            - set(source_data.columns)
        )

        if missing_source_columns:
            raise ValueError(
                "Source data is missing required columns: "
                f"{sorted(missing_source_columns)}"
            )

        source_targets = source_data[
            ["Employee_ID", "Resigned"]
        ].copy()

        source_targets["expected_y_true"] = (
            source_targets["Resigned"]
            .astype(int)
        )

        target_check = predictions[
            ["Employee_ID", "y_true"]
        ].merge(
            source_targets[
                [
                    "Employee_ID",
                    "expected_y_true",
                ]
            ],
            on="Employee_ID",
            validate="one_to_one",
        )

        target_mismatch = (
            target_check["y_true"].astype(int)
            != target_check["expected_y_true"]
        )

        if target_mismatch.any():
            raise ValueError(
                "One or more y_true values do not match the "
                "original dataset."
            )
