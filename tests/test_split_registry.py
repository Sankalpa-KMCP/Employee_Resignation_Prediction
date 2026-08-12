from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split_registry.csv"
)

EXPECTED_TOTAL_ROWS = 100_000
EXPECTED_DEVELOPMENT_ROWS = 80_000
EXPECTED_HOLDOUT_ROWS = 20_000
EXPECTED_FOLDS = {1, 2, 3, 4, 5}
EXPECTED_ROWS_PER_FOLD = 16_000


def load_registry() -> pd.DataFrame:
    """Load the official frozen split registry."""
    assert REGISTRY_PATH.exists(), (
        f"Split registry not found: {REGISTRY_PATH}"
    )

    return pd.read_csv(REGISTRY_PATH)


def test_registry_has_expected_columns():
    registry = load_registry()

    assert list(registry.columns) == [
        "Employee_ID",
        "partition",
        "cv_fold",
    ]


def test_registry_has_expected_total_rows():
    registry = load_registry()

    assert len(registry) == EXPECTED_TOTAL_ROWS


def test_employee_ids_are_unique():
    registry = load_registry()

    assert registry["Employee_ID"].is_unique
    assert registry["Employee_ID"].notna().all()


def test_only_valid_partitions_exist():
    registry = load_registry()

    assert set(registry["partition"].unique()) == {
        "development",
        "holdout",
    }


def test_partition_sizes_are_correct():
    registry = load_registry()

    counts = registry["partition"].value_counts()

    assert counts["development"] == EXPECTED_DEVELOPMENT_ROWS
    assert counts["holdout"] == EXPECTED_HOLDOUT_ROWS


def test_development_rows_have_cv_folds():
    registry = load_registry()

    development = registry[
        registry["partition"] == "development"
    ]

    assert development["cv_fold"].notna().all()


def test_holdout_rows_have_no_cv_fold():
    registry = load_registry()

    holdout = registry[
        registry["partition"] == "holdout"
    ]

    assert holdout["cv_fold"].isna().all()


def test_only_expected_cv_folds_exist():
    registry = load_registry()

    development = registry[
        registry["partition"] == "development"
    ]

    observed_folds = set(
        development["cv_fold"].astype(int).unique()
    )

    assert observed_folds == EXPECTED_FOLDS


def test_each_cv_fold_has_expected_size():
    registry = load_registry()

    development = registry[
        registry["partition"] == "development"
    ]

    fold_counts = (
        development["cv_fold"]
        .astype(int)
        .value_counts()
        .sort_index()
    )

    for fold in EXPECTED_FOLDS:
        assert fold_counts[fold] == EXPECTED_ROWS_PER_FOLD


def test_every_development_employee_has_exactly_one_fold():
    registry = load_registry()

    development = registry[
        registry["partition"] == "development"
    ]

    assert len(development) == development["Employee_ID"].nunique()

    assert (
        development["cv_fold"]
        .astype(int)
        .between(1, 5)
        .all()
    )


def test_development_and_holdout_do_not_overlap():
    registry = load_registry()

    development_ids = set(
        registry.loc[
            registry["partition"] == "development",
            "Employee_ID",
        ]
    )

    holdout_ids = set(
        registry.loc[
            registry["partition"] == "holdout",
            "Employee_ID",
        ]
    )

    assert development_ids.isdisjoint(holdout_ids)