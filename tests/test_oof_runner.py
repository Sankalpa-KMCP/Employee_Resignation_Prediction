"""Unit tests for the shared five-fold OOF cross-validation runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.base import ModelAdapter
from src.models.runner import run_5fold_cv


class InstrumentedMockAdapter:
    """Instrumented mock adapter recording fitted and predicted IDs."""

    def __init__(self, name: str = "Instrumented Mock", model_id: str = "inst_mock"):
        self.name = name
        self.model_id = model_id
        self.fitted_train_indices: list[Any] = []
        self.predicted_indices: list[Any] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> InstrumentedMockAdapter:
        self._is_fitted = True
        self.fitted_train_indices = list(X.index)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Adapter is not fitted.")
        self.predicted_indices = list(X.index)
        # Deterministic valid probabilities
        return np.full(len(X), 0.25, dtype=float)

    def save(self, directory: Path) -> dict[str, Any]:
        return {"status": "saved"}

    @classmethod
    def load(cls, directory: Path) -> InstrumentedMockAdapter:
        inst = cls()
        inst._is_fitted = True
        return inst

    def get_metadata(self) -> dict[str, Any]:
        return {"name": self.name, "model_id": self.model_id}


@pytest.fixture
def synthetic_split_data():
    """Create a synthetic 1,000-record dataset with 800 development and 200 holdout rows."""
    n_dev = 800
    n_holdout = 200
    total = n_dev + n_holdout

    employee_ids = [f"EMP_{i:04d}" for i in range(1, total + 1)]

    # 5 balanced folds of 160 rows each for development
    folds = ([1, 2, 3, 4, 5] * (n_dev // 5))
    partitions = ["development"] * n_dev + ["holdout"] * n_holdout
    cv_folds = folds + [np.nan] * n_holdout

    registry = pd.DataFrame(
        {
            "Employee_ID": employee_ids,
            "partition": partitions,
            "cv_fold": cv_folds,
        }
    )

    # Feature data for development only
    np.random.seed(42)
    dev_ids = employee_ids[:n_dev]
    features = pd.DataFrame(
        {
            "Employee_ID": dev_ids,
            "Age": np.random.randint(22, 60, size=n_dev),
            "Monthly_Salary": np.random.uniform(3000, 15000, size=n_dev),
            "Department": np.random.choice(["Sales", "Engineering", "HR"], size=n_dev),
            "Resigned": np.random.choice([0, 1], size=n_dev, p=[0.9, 0.1]),
        }
    )

    feature_cols = ["Age", "Monthly_Salary", "Department"]
    return features, registry, feature_cols


def test_runner_fresh_adapter_isolation(synthetic_split_data):
    features, registry, feature_cols = synthetic_split_data
    instantiated_adapters = []

    def factory():
        adapter = InstrumentedMockAdapter()
        instantiated_adapters.append(adapter)
        return adapter

    result = run_5fold_cv(
        adapter_factory=factory,
        data=features,
        split_registry=registry,
        feature_cols=feature_cols,
    )

    # Assert factory was called exactly 5 times
    assert len(instantiated_adapters) == 5

    # Assert each fold received a distinct instance
    for i in range(len(instantiated_adapters)):
        for j in range(i + 1, len(instantiated_adapters)):
            assert instantiated_adapters[i] is not instantiated_adapters[j]


def test_runner_preserves_metadata_from_every_fold(synthetic_split_data):
    features, registry, feature_cols = synthetic_split_data
    adapters = []

    class FoldAdapter(InstrumentedMockAdapter):
        def get_metadata(self):
            return {"instance": adapters.index(self) + 1, "diagnostic": {"epochs": len(adapters)}}

    def factory():
        adapter = FoldAdapter()
        adapters.append(adapter)
        return adapter

    result = run_5fold_cv(factory, features, registry, feature_cols)
    assert result.metadata["instance"] == 5
    evidence = result.metadata["fold_metadata"]
    assert [item["fold"] for item in evidence] == [1, 2, 3, 4, 5]
    assert [item["adapter_metadata"]["instance"] for item in evidence] == [1, 2, 3, 4, 5]
    assert [item["adapter_metadata"]["diagnostic"]["epochs"] for item in evidence] == [1, 2, 3, 4, 5]


def test_runner_row_parity_and_canonical_schema(synthetic_split_data):
    features, registry, feature_cols = synthetic_split_data

    result = run_5fold_cv(
        adapter_factory=InstrumentedMockAdapter,
        data=features,
        split_registry=registry,
        feature_cols=feature_cols,
    )

    # Check row count
    expected_dev_count = 800
    assert len(result.predictions) == expected_dev_count

    # Check canonical columns
    expected_cols = [
        "Employee_ID",
        "cv_fold",
        "y_true",
        "model_name",
        "y_probability",
        "y_pred_0_5",
    ]
    assert list(result.predictions.columns) == expected_cols

    # Check Employee_ID uniqueness and alignment
    assert result.predictions["Employee_ID"].is_unique
    assert set(result.predictions["Employee_ID"]) == set(features["Employee_ID"])

    # Check threshold prediction match
    expected_preds = (result.predictions["y_probability"] >= 0.50).astype(int)
    np.testing.assert_array_equal(result.predictions["y_pred_0_5"].to_numpy(), expected_preds.to_numpy())

    # Check fold metrics table
    assert len(result.fold_metrics) == 5
    assert set(result.fold_metrics["fold"]) == {1, 2, 3, 4, 5}
    assert "average_precision" in result.fold_metrics.columns
    assert "fit_time_seconds" in result.fold_metrics.columns
    assert "predict_time_seconds" in result.fold_metrics.columns


def test_runner_holdout_isolation_guard(synthetic_split_data):
    features, registry, feature_cols = synthetic_split_data

    # Deliberately contaminate data with a formal holdout record
    holdout_record = pd.DataFrame(
        {
            "Employee_ID": ["EMP_0850"],  # Known holdout ID
            "Age": [35],
            "Monthly_Salary": [5000.0],
            "Department": ["Sales"],
            "Resigned": [0],
        }
    )
    contaminated_features = pd.concat([features, holdout_record], ignore_index=True)

    with pytest.raises(ValueError, match="CRITICAL LEAKAGE ERROR"):
        run_5fold_cv(
            adapter_factory=InstrumentedMockAdapter,
            data=contaminated_features,
            split_registry=registry,
            feature_cols=feature_cols,
        )


def test_runner_outer_validation_isolation(synthetic_split_data):
    features, registry, feature_cols = synthetic_split_data
    created_adapters: list[InstrumentedMockAdapter] = []

    def factory():
        adapter = InstrumentedMockAdapter()
        created_adapters.append(adapter)
        return adapter

    result = run_5fold_cv(
        adapter_factory=factory,
        data=features,
        split_registry=registry,
        feature_cols=feature_cols,
    )

    # For each fold adapter, verify outer-val rows were never passed to fit()
    dev_reg = registry[registry["partition"] == "development"]
    for fold_num, adapter in enumerate(created_adapters, start=1):
        val_emp_ids = set(dev_reg.loc[dev_reg["cv_fold"] == fold_num, "Employee_ID"])
        train_emp_ids = set(dev_reg.loc[dev_reg["cv_fold"] != fold_num, "Employee_ID"])

        # Train indices should match train_emp_ids exactly
        fitted_ids = set(features.loc[adapter.fitted_train_indices, "Employee_ID"])
        assert fitted_ids == train_emp_ids
        assert fitted_ids.isdisjoint(val_emp_ids)


def test_runner_catches_invalid_probability_shapes(synthetic_split_data):
    features, registry, feature_cols = synthetic_split_data

    class BadShapeAdapter(InstrumentedMockAdapter):
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            # Returns 2D array instead of 1D
            return np.full((len(X), 2), 0.5)

    with pytest.raises(ValueError, match="expected 1D array"):
        run_5fold_cv(
            adapter_factory=BadShapeAdapter,
            data=features,
            split_registry=registry,
            feature_cols=feature_cols,
        )


def test_runner_catches_out_of_bounds_probabilities(synthetic_split_data):
    features, registry, feature_cols = synthetic_split_data

    class OutOfBoundsAdapter(InstrumentedMockAdapter):
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            probs = np.full(len(X), 0.5)
            probs[0] = 1.25  # Exceeds 1.0
            return probs

    with pytest.raises(ValueError, match="probabilities outside"):
        run_5fold_cv(
            adapter_factory=OutOfBoundsAdapter,
            data=features,
            split_registry=registry,
            feature_cols=feature_cols,
        )


def test_runner_catches_nan_probabilities(synthetic_split_data):
    features, registry, feature_cols = synthetic_split_data

    class NanProbAdapter(InstrumentedMockAdapter):
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            probs = np.full(len(X), 0.5)
            probs[0] = np.nan
            return probs

    with pytest.raises(ValueError, match="non-finite probability"):
        run_5fold_cv(
            adapter_factory=NanProbAdapter,
            data=features,
            split_registry=registry,
            feature_cols=feature_cols,
        )


def test_runner_catches_invalid_factory_object(synthetic_split_data):
    features, registry, feature_cols = synthetic_split_data

    def bad_factory():
        return object()  # Not a ModelAdapter

    with pytest.raises(TypeError, match="does not implement ModelAdapter"):
        run_5fold_cv(
            adapter_factory=bad_factory,
            data=features,
            split_registry=registry,
            feature_cols=feature_cols,
        )


@pytest.mark.parametrize("column", ["Resigned", "Employee_ID", "Hire_Date", "cv_fold", "partition"])
def test_runner_rejects_forbidden_predictors_before_creating_adapter(synthetic_split_data, column):
    features, registry, feature_cols = synthetic_split_data
    created = []

    def factory():
        created.append(True)
        return InstrumentedMockAdapter()

    with pytest.raises(ValueError, match="Forbidden predictor"):
        run_5fold_cv(factory, features, registry, [*feature_cols, column])
    assert created == []  # Even missing forbidden columns must fail before fitting.


@pytest.mark.parametrize("feature_cols", [[], ["Age", "Age"]])
def test_runner_requires_unique_nonempty_predictors(synthetic_split_data, feature_cols):
    features, registry, _ = synthetic_split_data
    with pytest.raises(ValueError, match="non-empty and unique"):
        run_5fold_cv(InstrumentedMockAdapter, features, registry, feature_cols)


@pytest.mark.parametrize("invalid", [0.8, -1.0, 2.0, np.nan, np.inf, "invalid", "0"])
def test_runner_rejects_original_invalid_labels_before_fitting(synthetic_split_data, invalid):
    features, registry, feature_cols = synthetic_split_data
    features = features.copy()
    features["Resigned"] = features.Resigned.astype(object if isinstance(invalid, str) else float)
    features.loc[0, "Resigned"] = invalid
    created = []

    def factory():
        created.append(True)
        return InstrumentedMockAdapter()

    with pytest.raises(ValueError, match="Resigned"):
        run_5fold_cv(factory, features, registry, feature_cols)
    assert created == []


@pytest.mark.parametrize("dtype", [float, bool])
def test_runner_accepts_exact_binary_float_and_boolean_labels(synthetic_split_data, dtype):
    features, registry, feature_cols = synthetic_split_data
    features = features.copy()
    features["Resigned"] = features.Resigned.astype(dtype)
    result = run_5fold_cv(InstrumentedMockAdapter, features, registry, feature_cols)
    assert result.predictions.y_true.tolist() == features.Resigned.astype(int).tolist()
