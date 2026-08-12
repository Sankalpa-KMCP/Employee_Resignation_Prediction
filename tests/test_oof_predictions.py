from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.oof_validation import (
    validate_oof_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split_registry.csv"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Extended_Employee_Performance_and_Productivity_Data.csv"
)

BASELINE_FOLDER = (
    PROJECT_ROOT
    / "results"
    / "baselines"
)

DUMMY_PRIOR_PATH = (
    BASELINE_FOLDER
    / "dummy_prior_oof_predictions.csv"
)

DUMMY_MOST_FREQUENT_PATH = (
    BASELINE_FOLDER
    / "dummy_most_frequent_oof_predictions.csv"
)


@pytest.fixture(scope="module")
def split_registry():
    return pd.read_csv(REGISTRY_PATH)


@pytest.fixture(scope="module")
def source_data():
    return pd.read_csv(
        DATA_PATH,
        usecols=[
            "Employee_ID",
            "Resigned",
        ],
    )


@pytest.fixture(scope="module")
def dummy_prior():
    return pd.read_csv(
        DUMMY_PRIOR_PATH
    )


@pytest.fixture(scope="module")
def dummy_most_frequent():
    return pd.read_csv(
        DUMMY_MOST_FREQUENT_PATH
    )


def test_dummy_prior_oof_is_valid(
    dummy_prior,
    split_registry,
    source_data,
):
    validate_oof_predictions(
        dummy_prior,
        split_registry,
        source_data=source_data,
        expected_model_name="Dummy Prior",
    )


def test_dummy_most_frequent_oof_is_valid(
    dummy_most_frequent,
    split_registry,
    source_data,
):
    validate_oof_predictions(
        dummy_most_frequent,
        split_registry,
        source_data=source_data,
        expected_model_name="Dummy Most Frequent",
    )


def test_oof_contains_80000_rows(
    dummy_prior,
):
    assert len(dummy_prior) == 80_000


def test_oof_employee_ids_are_unique(
    dummy_prior,
):
    assert dummy_prior[
        "Employee_ID"
    ].is_unique


def test_oof_contains_five_folds(
    dummy_prior,
):
    assert set(
        dummy_prior["cv_fold"].unique()
    ) == {
        1,
        2,
        3,
        4,
        5,
    }


def test_duplicate_employee_is_rejected(
    dummy_prior,
    split_registry,
):
    broken = dummy_prior.copy()

    broken.loc[
        broken.index[1],
        "Employee_ID",
    ] = broken.loc[
        broken.index[0],
        "Employee_ID",
    ]

    with pytest.raises(
        ValueError,
        match="exactly once",
    ):
        validate_oof_predictions(
            broken,
            split_registry,
        )


def test_wrong_fold_is_rejected(
    dummy_prior,
    split_registry,
):
    broken = dummy_prior.copy()

    original_fold = int(
        broken.loc[0, "cv_fold"]
    )

    broken.loc[
        0,
        "cv_fold",
    ] = (
        1
        if original_fold != 1
        else 2
    )

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        validate_oof_predictions(
            broken,
            split_registry,
        )


def test_probability_above_one_is_rejected(
    dummy_prior,
    split_registry,
):
    broken = dummy_prior.copy()

    broken.loc[
        0,
        "y_probability",
    ] = 1.10

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        validate_oof_predictions(
            broken,
            split_registry,
        )


def test_probability_below_zero_is_rejected(
    dummy_prior,
    split_registry,
):
    broken = dummy_prior.copy()

    broken.loc[
        0,
        "y_probability",
    ] = -0.10

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        validate_oof_predictions(
            broken,
            split_registry,
        )


def test_missing_probability_is_rejected(
    dummy_prior,
    split_registry,
):
    broken = dummy_prior.copy()

    broken.loc[
        0,
        "y_probability",
    ] = None

    with pytest.raises(
        ValueError,
        match="missing or non-numeric",
    ):
        validate_oof_predictions(
            broken,
            split_registry,
        )


def test_wrong_model_name_is_rejected(
    dummy_prior,
    split_registry,
):
    with pytest.raises(
        ValueError,
        match="Unexpected model_name",
    ):
        validate_oof_predictions(
            dummy_prior,
            split_registry,
            expected_model_name="CatBoost",
        )


def test_multiple_model_names_are_rejected(
    dummy_prior,
    split_registry,
):
    broken = dummy_prior.copy()

    broken.loc[
        0,
        "model_name",
    ] = "Different Model"

    with pytest.raises(
        ValueError,
        match="exactly one model_name",
    ):
        validate_oof_predictions(
            broken,
            split_registry,
        )


def test_wrong_target_is_rejected(
    dummy_prior,
    split_registry,
    source_data,
):
    broken = dummy_prior.copy()

    broken.loc[
        0,
        "y_true",
    ] = 1 - int(
        broken.loc[0, "y_true"]
    )

    with pytest.raises(
        ValueError,
        match="do not match the original dataset",
    ):
        validate_oof_predictions(
            broken,
            split_registry,
            source_data=source_data,
        )


def test_wrong_threshold_prediction_is_rejected(
    dummy_prior,
    split_registry,
):
    broken = dummy_prior.copy()

    broken.loc[
        0,
        "y_pred_0_5",
    ] = 1 - int(
        broken.loc[0, "y_pred_0_5"]
    )

    with pytest.raises(
        ValueError,
        match="0.50 probability threshold",
    ):
        validate_oof_predictions(
            broken,
            split_registry,
        )


def test_missing_required_column_is_rejected(
    dummy_prior,
    split_registry,
):
    broken = dummy_prior.drop(
        columns=["y_probability"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required OOF columns",
    ):
        validate_oof_predictions(
            broken,
            split_registry,
        )