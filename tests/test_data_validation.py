from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Extended_Employee_Performance_and_Productivity_Data.csv"
)

EXPECTED_SHA256 = (
    "2b1e07caf58e7678a780d428c1a8893a"
    "95d9c4a556134fd8e7368db5ed596cdb"
)

EXPECTED_ROWS = 100_000
EXPECTED_COLUMNS = 20

EXPECTED_SCHEMA = [
    "Employee_ID",
    "Department",
    "Gender",
    "Age",
    "Job_Title",
    "Hire_Date",
    "Years_At_Company",
    "Education_Level",
    "Performance_Score",
    "Monthly_Salary",
    "Work_Hours_Per_Week",
    "Projects_Handled",
    "Overtime_Hours",
    "Sick_Days",
    "Remote_Work_Frequency",
    "Team_Size",
    "Training_Hours",
    "Promotions",
    "Employee_Satisfaction_Score",
    "Resigned",
]

EXPECTED_CATEGORIES = {
    "Department": {
        "Customer Support",
        "Engineering",
        "Finance",
        "HR",
        "IT",
        "Legal",
        "Marketing",
        "Operations",
        "Sales",
    },
    "Gender": {
        "Female",
        "Male",
        "Other",
    },
    "Job_Title": {
        "Analyst",
        "Consultant",
        "Developer",
        "Engineer",
        "Manager",
        "Specialist",
        "Technician",
    },
    "Education_Level": {
        "High School",
        "Bachelor",
        "Master",
        "PhD",
    },
}

NUMERICAL_COLUMNS = [
    "Age",
    "Years_At_Company",
    "Performance_Score",
    "Monthly_Salary",
    "Work_Hours_Per_Week",
    "Projects_Handled",
    "Overtime_Hours",
    "Sick_Days",
    "Remote_Work_Frequency",
    "Team_Size",
    "Training_Hours",
    "Promotions",
    "Employee_Satisfaction_Score",
]


@pytest.fixture(scope="module")
def data():
    assert DATA_PATH.exists(), (
        f"Raw dataset not found: {DATA_PATH}"
    )

    return pd.read_csv(DATA_PATH)


def test_dataset_checksum_matches_frozen_source():
    actual_hash = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    assert actual_hash == EXPECTED_SHA256


def test_dataset_has_expected_shape(data):
    assert data.shape == (
        EXPECTED_ROWS,
        EXPECTED_COLUMNS,
    )


def test_dataset_schema_is_exact(data):
    assert data.columns.tolist() == EXPECTED_SCHEMA


def test_dataset_has_no_missing_values(data):
    assert data.isna().sum().sum() == 0


def test_dataset_has_no_duplicate_rows(data):
    assert data.duplicated().sum() == 0


def test_employee_ids_are_present_and_unique(data):
    assert data["Employee_ID"].notna().all()
    assert data["Employee_ID"].is_unique


def test_target_is_binary(data):
    observed = set(
        data["Resigned"].unique()
    )

    assert observed == {
        False,
        True,
    }


def test_target_distribution_matches_audit(data):
    counts = (
        data["Resigned"]
        .value_counts()
        .to_dict()
    )

    assert counts[False] == 89_990
    assert counts[True] == 10_010


def test_expected_categorical_values(data):
    for column, expected_values in (
        EXPECTED_CATEGORIES.items()
    ):
        observed_values = set(
            data[column].unique()
        )

        assert observed_values == expected_values, (
            f"Unexpected categories in {column}: "
            f"{observed_values - expected_values}"
        )


def test_numerical_columns_are_finite(data):
    numerical_data = data[
        NUMERICAL_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    assert numerical_data.notna().all().all()

    assert np.isfinite(
        numerical_data.to_numpy()
    ).all()


def test_hire_dates_are_valid(data):
    parsed_dates = pd.to_datetime(
        data["Hire_Date"],
        errors="coerce",
    )

    assert parsed_dates.notna().all()


def test_hire_date_time_component_is_consistent(data):
    parsed_dates = pd.to_datetime(
        data["Hire_Date"],
        errors="raise",
    )

    time_components = parsed_dates.dt.time

    assert time_components.nunique() == 1


def test_remote_work_frequency_uses_expected_values(data):
    observed = set(
        data[
            "Remote_Work_Frequency"
        ].unique()
    )

    assert observed == {
        0,
        25,
        50,
        75,
        100,
    }


def test_salary_is_deterministic_from_title_and_performance(data):
    salary_counts = (
        data.groupby(
            [
                "Job_Title",
                "Performance_Score",
            ]
        )["Monthly_Salary"]
        .nunique()
    )

    assert salary_counts.max() == 1


def test_primary_feature_count_is_17():
    categorical_features = [
        "Department",
        "Gender",
        "Job_Title",
        "Education_Level",
    ]

    numerical_features = NUMERICAL_COLUMNS

    assert (
        len(categorical_features)
        + len(numerical_features)
        == 17
    )