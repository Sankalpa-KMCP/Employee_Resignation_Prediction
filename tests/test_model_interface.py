"""Unit tests for the ModelAdapter Protocol and base structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.base import ModelAdapter, OOFResult


class MockValidAdapter:
    """Mock model adapter conforming strictly to ModelAdapter Protocol."""

    def __init__(self, name: str = "Mock Model", model_id: str = "mock_model"):
        self.name = name
        self.model_id = model_id
        self._fitted = False
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> MockValidAdapter:
        self._fitted = True
        self._feature_names = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")
        n_samples = len(X)
        # Return deterministic probabilities in [0, 1]
        return np.linspace(0.1, 0.9, n_samples)

    def save(self, directory: Path) -> dict[str, Any]:
        return {"saved": True, "name": self.name}

    @classmethod
    def load(cls, directory: Path) -> MockValidAdapter:
        adapter = cls()
        adapter._fitted = True
        return adapter

    def get_metadata(self) -> dict[str, Any]:
        return {"name": self.name, "model_id": self.model_id, "fitted": self._fitted}


class MockIncompleteAdapter:
    """Mock object missing predict_proba and save."""

    def __init__(self):
        self.name = "Incomplete"
        self.model_id = "incomplete"

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        return self


def test_valid_adapter_satisfies_protocol():
    adapter = MockValidAdapter()
    assert isinstance(adapter, ModelAdapter)


def test_incomplete_adapter_fails_protocol_check():
    adapter = MockIncompleteAdapter()
    assert not isinstance(adapter, ModelAdapter)


def test_adapter_fit_and_predict_proba():
    adapter = MockValidAdapter()
    X = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0], "feat_2": [10.0, 20.0, 30.0]})
    y = pd.Series([0, 1, 0])

    fitted_adapter = adapter.fit(X, y)
    assert fitted_adapter is adapter

    probs = adapter.predict_proba(X)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 1
    assert len(probs) == len(X)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert np.isfinite(probs).all()


def test_adapter_metadata_and_serialization():
    adapter = MockValidAdapter(name="Test EBM", model_id="ebm_test")
    metadata = adapter.get_metadata()
    assert metadata["name"] == "Test EBM"
    assert metadata["model_id"] == "ebm_test"

    save_info = adapter.save(Path("/tmp/dummy"))
    assert save_info["saved"] is True

    loaded = MockValidAdapter.load(Path("/tmp/dummy"))
    assert loaded._fitted is True


def test_oof_result_dataclass_structure():
    oof_df = pd.DataFrame({"Employee_ID": [1, 2], "y_true": [0, 1]})
    fold_df = pd.DataFrame({"fold": [1], "average_precision": [0.85]})
    result = OOFResult(
        model_name="Mock",
        model_id="mock",
        predictions=oof_df,
        fold_metrics=fold_df,
        summary_metrics={"pooled_ap": 0.85},
        fit_time_seconds_total=1.2,
        predict_time_seconds_total=0.3,
    )
    assert result.model_name == "Mock"
    assert len(result.predictions) == 2
    assert result.fit_time_seconds_total == 1.2
