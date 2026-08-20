"""XGBoost Model Adapter for Employee Resignation Prediction (Member 3).

Conforms to the ModelAdapter protocol with fold-local one-hot encoding
and histogram-based gradient boosting.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelAdapter


class XGBoostAdapter:
    """XGBoost model adapter with fold-local preprocessing."""

    def __init__(
        self,
        name: str = "XGBoost",
        model_id: str = "xgboost",
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.params = params or {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }
        self.categorical_features = categorical_features or [
            "Department",
            "Gender",
            "Job_Title",
            "Education_Level",
        ]
        self._model: Any = None
        self._encoder: Any = None
        self._feature_names: list[str] = []
        self._fitted_columns: list[str] = []
        self._fitted = False

    def _preprocess(self, X: pd.DataFrame, fit_encoder: bool = False) -> pd.DataFrame:
        """Apply fold-local one-hot encoding to categorical features."""
        from sklearn.preprocessing import OneHotEncoder

        X_num = X.drop(columns=[c for c in self.categorical_features if c in X.columns])
        cat_cols = [c for c in self.categorical_features if c in X.columns]

        if not cat_cols:
            return X.copy()

        if fit_encoder:
            self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cat_encoded = self._encoder.fit_transform(X[cat_cols])
            cat_feature_names = list(self._encoder.get_feature_names_out(cat_cols))
        else:
            if self._encoder is None:
                raise RuntimeError("Encoder is not fitted.")
            cat_encoded = self._encoder.transform(X[cat_cols])
            cat_feature_names = list(self._encoder.get_feature_names_out(cat_cols))

        cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=X.index)
        combined = pd.concat([X_num, cat_df], axis=1)
        return combined

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> XGBoostAdapter:
        """Fit fold-local encoder and XGBoost classifier on training fold data."""
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoostAdapter. Install with pip install xgboost."
            ) from exc

        self._feature_names = list(X.columns)
        X_proc = self._preprocess(X, fit_encoder=True)
        self._fitted_columns = list(X_proc.columns)

        y_arr = np.asarray(y, dtype=int)

        self._model = XGBClassifier(**self.params)
        self._model.fit(X_proc, y_arr)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive class probabilities for input features."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Adapter is not fitted. Call fit() first.")

        X_proc = self._preprocess(X, fit_encoder=False)
        probs = self._model.predict_proba(X_proc)[:, 1]
        return np.asarray(probs, dtype=float)

    def save(self, directory: Path) -> dict[str, Any]:
        """Serialize model artifacts and preprocessors to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        artifact_file = directory / "xgboost_adapter.pkl"
        with open(artifact_file, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "encoder": self._encoder,
                    "params": self.params,
                    "categorical_features": self.categorical_features,
                    "fitted_columns": self._fitted_columns,
                    "feature_names": self._feature_names,
                },
                f,
            )

        metadata = self.get_metadata()
        metadata["artifact_file"] = str(artifact_file.name)
        return metadata

    @classmethod
    def load(cls, directory: Path) -> XGBoostAdapter:
        """Load serialized XGBoost adapter from disk."""
        directory = Path(directory)
        artifact_file = directory / "xgboost_adapter.pkl"

        adapter = cls()
        if artifact_file.exists():
            with open(artifact_file, "rb") as f:
                state = pickle.load(f)
            adapter._model = state["model"]
            adapter._encoder = state["encoder"]
            adapter.params = state["params"]
            adapter.categorical_features = state["categorical_features"]
            adapter._fitted_columns = state["fitted_columns"]
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
            "num_fitted_features": len(self._fitted_columns),
        }
