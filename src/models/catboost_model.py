"""CatBoost Model Adapter for Employee Resignation Prediction (Member 2).

Conforms to the ModelAdapter protocol with fold-local preprocessing
and native categorical feature support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelAdapter


class CatBoostAdapter:
    """CatBoost model adapter with native categorical handling."""

    def __init__(
        self,
        name: str = "CatBoost",
        model_id: str = "catboost",
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.params = params or {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        }
        self.categorical_features = categorical_features or [
            "Department",
            "Gender",
            "Job_Title",
            "Education_Level",
        ]
        self._model: Any = None
        self._feature_names: list[str] = []
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> CatBoostAdapter:
        """Fit CatBoost model on training fold data with fold-local categorical handling."""
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "catboost is required for CatBoostAdapter. Install with pip install catboost."
            ) from exc

        self._feature_names = list(X.columns)
        cat_indices = [
            i for i, col in enumerate(self._feature_names) if col in self.categorical_features
        ]

        # Ensure categoricals are treated as string/category objects
        X_proc = X.copy()
        for col in self.categorical_features:
            if col in X_proc.columns:
                X_proc[col] = X_proc[col].astype(str)

        y_arr = np.asarray(y, dtype=int)

        self._model = CatBoostClassifier(**self.params, cat_features=cat_indices)
        self._model.fit(X_proc, y_arr, verbose=False)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive class probabilities for input features."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Adapter is not fitted. Call fit() first.")

        X_proc = X.copy()
        for col in self.categorical_features:
            if col in X_proc.columns:
                X_proc[col] = X_proc[col].astype(str)

        probs = self._model.predict_proba(X_proc)[:, 1]
        return np.asarray(probs, dtype=float)

    def save(self, directory: Path) -> dict[str, Any]:
        """Serialize model artifacts and metadata to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_file = directory / "model.cbm"
        if self._model is not None:
            self._model.save_model(str(model_file))

        metadata = self.get_metadata()
        metadata["model_file"] = str(model_file.name)
        return metadata

    @classmethod
    def load(cls, directory: Path) -> CatBoostAdapter:
        """Load serialized CatBoost adapter from disk."""
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("catboost is required to load CatBoostAdapter.") from exc

        directory = Path(directory)
        model_file = directory / "model.cbm"

        adapter = cls()
        if model_file.exists():
            model = CatBoostClassifier()
            model.load_model(str(model_file))
            adapter._model = model
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
