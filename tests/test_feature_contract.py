from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_CONTRACT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "feature_contract.yaml"
)


EXPECTED_TARGET = "Resigned"
EXPECTED_IDENTIFIER = "Employee_ID"

EXPECTED_CATEGORICAL_FEATURES = {
    "Department",
    "Gender",
    "Job_Title",
    "Education_Level",
}

EXPECTED_NUMERICAL_FEATURES = {
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
}

EXPECTED_PRIMARY_FEATURES = (
    EXPECTED_CATEGORICAL_FEATURES
    | EXPECTED_NUMERICAL_FEATURES
)

EXPECTED_EXCLUDED_FEATURES = {
    "Employee_ID",
    "Hire_Date",
    "Resigned",
}

EXPECTED_SENSITIVE_ATTRIBUTES = {
    "Gender",
    "Age",
}


def load_feature_contract() -> dict:
    """Load the frozen shared feature contract."""
    assert FEATURE_CONTRACT_PATH.exists(), (
        f"Feature contract not found: {FEATURE_CONTRACT_PATH}"
    )

    with FEATURE_CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def test_contract_is_frozen():
    contract = load_feature_contract()

    assert contract["contract"]["status"] == "frozen"


def test_target_column_is_correct():
    contract = load_feature_contract()

    assert contract["target"]["column"] == EXPECTED_TARGET
    assert contract["target"]["use_as_predictor"] is False


def test_identifier_column_is_correct():
    contract = load_feature_contract()

    assert contract["identifier"]["column"] == EXPECTED_IDENTIFIER
    assert contract["identifier"]["use_as_predictor"] is False


def test_categorical_features_are_correct():
    contract = load_feature_contract()

    observed = {
        feature["name"]
        for feature in contract["categorical_features"]
    }

    assert observed == EXPECTED_CATEGORICAL_FEATURES


def test_numerical_features_are_correct():
    contract = load_feature_contract()

    observed = {
        feature["name"]
        for feature in contract["numerical_features"]
    }

    assert observed == EXPECTED_NUMERICAL_FEATURES


def test_primary_feature_set_is_correct():
    contract = load_feature_contract()

    observed = set(
        contract["primary_model_features"]
    )

    assert observed == EXPECTED_PRIMARY_FEATURES


def test_primary_feature_count_is_correct():
    contract = load_feature_contract()

    observed = contract["primary_model_features"]

    assert len(observed) == 17
    assert len(set(observed)) == 17

    counts = contract["feature_counts"]

    assert counts["categorical"] == 4
    assert counts["numerical_or_ordinal"] == 13
    assert counts["total_primary_predictors"] == 17


def test_excluded_features_are_correct():
    contract = load_feature_contract()

    observed = {
        feature["column"]
        for feature in contract["excluded_features"]
    }

    assert observed == EXPECTED_EXCLUDED_FEATURES


def test_excluded_features_are_not_predictors():
    contract = load_feature_contract()

    predictors = set(
        contract["primary_model_features"]
    )

    excluded = {
        feature["column"]
        for feature in contract["excluded_features"]
    }

    assert predictors.isdisjoint(excluded)


def test_identifier_is_not_a_predictor():
    contract = load_feature_contract()

    predictors = set(
        contract["primary_model_features"]
    )

    assert EXPECTED_IDENTIFIER not in predictors


def test_target_is_not_a_predictor():
    contract = load_feature_contract()

    predictors = set(
        contract["primary_model_features"]
    )

    assert EXPECTED_TARGET not in predictors


def test_raw_hire_date_is_not_a_predictor():
    contract = load_feature_contract()

    predictors = set(
        contract["primary_model_features"]
    )

    assert "Hire_Date" not in predictors


def test_categorical_and_numerical_features_do_not_overlap():
    contract = load_feature_contract()

    categorical = {
        feature["name"]
        for feature in contract["categorical_features"]
    }

    numerical = {
        feature["name"]
        for feature in contract["numerical_features"]
    }

    assert categorical.isdisjoint(numerical)


def test_categorical_and_numerical_features_cover_primary_set():
    contract = load_feature_contract()

    categorical = {
        feature["name"]
        for feature in contract["categorical_features"]
    }

    numerical = {
        feature["name"]
        for feature in contract["numerical_features"]
    }

    primary = set(
        contract["primary_model_features"]
    )

    assert categorical | numerical == primary


def test_sensitive_attributes_are_correct():
    contract = load_feature_contract()

    observed = set(
        contract["sensitive_attributes"]
    )

    assert observed == EXPECTED_SENSITIVE_ATTRIBUTES


def test_sensitive_attributes_are_primary_predictors():
    contract = load_feature_contract()

    sensitive = set(
        contract["sensitive_attributes"]
    )

    predictors = set(
        contract["primary_model_features"]
    )

    assert sensitive.issubset(predictors)


def test_gender_is_marked_sensitive():
    contract = load_feature_contract()

    feature_map = {
        feature["name"]: feature
        for feature in contract["categorical_features"]
    }

    assert feature_map["Gender"]["sensitive_attribute"] is True


def test_age_is_marked_sensitive():
    contract = load_feature_contract()

    feature_map = {
        feature["name"]: feature
        for feature in contract["numerical_features"]
    }

    assert feature_map["Age"]["sensitive_attribute"] is True


def test_only_gender_and_age_are_marked_sensitive():
    contract = load_feature_contract()

    marked_sensitive = {
        feature["name"]
        for feature in (
            contract["categorical_features"]
            + contract["numerical_features"]
        )
        if feature["sensitive_attribute"] is True
    }

    assert marked_sensitive == EXPECTED_SENSITIVE_ATTRIBUTES


def test_universal_preprocessed_dataset_is_forbidden():
    contract = load_feature_contract()

    policy = contract["preprocessing_policy"]

    assert (
        policy["universal_preprocessed_dataset_allowed"]
        is False
    )


def test_split_registry_contract_matches_project_design():
    contract = load_feature_contract()

    split = contract["split_registry"]

    assert split["development_rows"] == 80_000
    assert split["formal_holdout_rows"] == 20_000
    assert split["cv_folds"] == 5


def test_prediction_output_contract_requires_identity():
    contract = load_feature_contract()

    output = contract["prediction_output_contract"]

    assert set(
        output["required_identity_columns"]
    ) == {
        "Employee_ID",
        "cv_fold",
    }

    assert output["required_target_column"] == [
        "y_true"
    ]

    assert set(
        output["required_prediction_fields"]
    ) == {
        "model_name",
        "y_probability",
    }