"""LightGBM Model Adapter for Employee Resignation Prediction (Member 4).

Conforms to the ModelAdapter protocol with fold-local category encoding
and LightGBM gradient boosted trees.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelAdapter


class LightGBMAdapter:
    """LightGBM model adapter with fold-local categorical handling."""

    def __init__(
        self,
        name: str = "LightGBM",
        model_id: str = "lightgbm",
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.params = params or {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "random_state": 42,
            "verbosity": -1,
            "force_col_wise": True,
            "n_jobs": -1,
        }
        self.categorical_features = categorical_features or [
            "Department",
            "Gender",
            "Job_Title",
            "Education_Level",
        ]
        self._model: Any = None
        self._category_maps: dict[str, dict[str, int]] = {}
        self._feature_names: list[str] = []
        self._fitted = False

    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Convert categorical features to integer category codes fold-locally."""
        X_proc = X.copy()
        for col in self.categorical_features:
            if col in X_proc.columns:
                if fit:
                    unique_vals = sorted(X_proc[col].dropna().unique())
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    self._category_maps[col] = mapping
                mapping = self._category_maps.get(col, {})
                X_proc[col] = X_proc[col].map(mapping).fillna(-1).astype("int32")
                X_proc[col] = X_proc[col].astype("category")
        return X_proc

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> LightGBMAdapter:
        """Fit fold-local encoders and LightGBM model on training fold data."""
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is required for LightGBMAdapter. Install with pip install lightgbm."
            ) from exc

        self._feature_names = list(X.columns)
        X_proc = self._preprocess(X, fit=True)
        y_arr = np.asarray(y, dtype=int)

        self._model = LGBMClassifier(**self.params)
        self._model.fit(X_proc, y_arr)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive class probabilities for input features."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Adapter is not fitted. Call fit() first.")

        X_proc = self._preprocess(X, fit=False)
        probs = self._model.predict_proba(X_proc)[:, 1]
        return np.asarray(probs, dtype=float)

    def save(self, directory: Path) -> dict[str, Any]:
        """Serialize model artifacts and preprocessor state to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        artifact_file = directory / "lightgbm_adapter.pkl"
        with open(artifact_file, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "category_maps": self._category_maps,
                    "params": self.params,
                    "categorical_features": self.categorical_features,
                    "feature_names": self._feature_names,
                },
                f,
            )

        metadata = self.get_metadata()
        metadata["artifact_file"] = str(artifact_file.name)
        return metadata

    @classmethod
    def load(cls, directory: Path) -> LightGBMAdapter:
        """Load serialized LightGBM adapter from disk."""
        directory = Path(directory)
        artifact_file = directory / "lightgbm_adapter.pkl"

        adapter = cls()
        if artifact_file.exists():
            with open(artifact_file, "rb") as f:
                state = pickle.load(f)
            adapter._model = state["model"]
            adapter._category_maps = state["category_maps"]
            adapter.params = state["params"]
            adapter.categorical_features = state["categorical_features"]
            adapter._feature_names = state["feature_names"]
            adapter._fitted = True
        return adapter

    def get_metadata(self) -> dict[str, Any]:
        """Return model metadata, hyperparameter settings, and feature names."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "fitted": self._fitted,
            "params": self.params,
            "categorical_features": self.categorical_features,
            "feature_names": self._feature_names,
        }
