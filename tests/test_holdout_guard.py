"""Unit tests for the formal holdout isolation guard and leakage prevention."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.models.base import ModelAdapter
from src.models.runner import run_5fold_cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_REGISTRY_PATH = PROJECT_ROOT / "data" / "processed" / "split_registry.csv"


class DummyTestAdapter:
    """Minimal adapter for testing runner guard behavior."""

    def __init__(self, name="Dummy Guard Test", model_id="dummy_guard"):
        self.name = name
        self.model_id = model_id
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.1, dtype=float)

    def save(self, directory: Path):
        return {}

    @classmethod
    def load(cls, directory: Path):
        return cls()

    def get_metadata(self):
        return {"name": self.name, "model_id": self.model_id}


def test_oof_runner_rejects_data_containing_formal_holdout_records():
    """Verify run_5fold_cv raises ValueError if formal holdout records are in the data dataframe."""
    # Synthetic registry with 80 dev and 20 holdout records
    dev_ids = [f"EMP_{i:04d}" for i in range(1, 81)]
    holdout_ids = [f"EMP_{i:04d}" for i in range(81, 101)]

    registry_df = pd.DataFrame(
        {
            "Employee_ID": dev_ids + holdout_ids,
            "partition": ["development"] * 80 + ["holdout"] * 20,
            "cv_fold": [1, 2, 3, 4, 5] * 16 + [np.nan] * 20,
        }
    )

    # Data containing both dev and 1 holdout row
    all_ids = dev_ids + [holdout_ids[0]]
    data_df = pd.DataFrame(
        {
            "Employee_ID": all_ids,
            "feature_1": np.random.randn(len(all_ids)),
            "Resigned": np.random.randint(0, 2, len(all_ids)),
        }
    )

    with pytest.raises(ValueError, match="CRITICAL LEAKAGE ERROR"):
        run_5fold_cv(
            adapter_factory=DummyTestAdapter,
            data=data_df,
            split_registry=registry_df,
            feature_cols=["feature_1"],
            target_col="Resigned",
            id_col="Employee_ID",
        )


def test_oof_runner_executes_when_data_is_strictly_development_partition():
    """Verify run_5fold_cv succeeds when data contains exclusively development records."""
    dev_ids = [f"EMP_{i:04d}" for i in range(1, 81)]
    holdout_ids = [f"EMP_{i:04d}" for i in range(81, 101)]

    registry_df = pd.DataFrame(
        {
            "Employee_ID": dev_ids + holdout_ids,
            "partition": ["development"] * 80 + ["holdout"] * 20,
            "cv_fold": [1, 2, 3, 4, 5] * 16 + [np.nan] * 20,
        }
    )

    dev_data_df = pd.DataFrame(
        {
            "Employee_ID": dev_ids,
            "feature_1": np.random.randn(80),
            "Resigned": [0, 1] * 40,
        }
    )

    result = run_5fold_cv(
        adapter_factory=DummyTestAdapter,
        data=dev_data_df,
        split_registry=registry_df,
        feature_cols=["feature_1"],
        target_col="Resigned",
        id_col="Employee_ID",
    )

    assert result.model_name == "Dummy Guard Test"
    assert len(result.predictions) == 80
    assert set(result.predictions["Employee_ID"]) == set(dev_ids)
