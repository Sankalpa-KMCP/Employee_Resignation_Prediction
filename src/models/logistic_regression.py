"""Logistic Regression Adapter for Employee Resignation Prediction.

Serves as the reference linear benchmark and PCA demonstration model.
Supports fold-local standardization, one-hot encoding, and optional PCA
decomposition (e.g. >=95% variance retention).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.base import ModelAdapter


class LogisticRegressionAdapter:
    """Logistic Regression adapter with fold-local preprocessing and optional PCA."""

    def __init__(
        self,
        name: str = "Logistic Regression",
        model_id: str = "logistic_regression",
        use_pca: bool = False,
        pca_variance_ratio: float = 0.95,
        pca_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.use_pca = use_pca
        self.pca_variance_ratio = pca_variance_ratio
        self.pca_features = pca_features
        self.categorical_features = categorical_features or [
            "Department",
            "Gender",
            "Job_Title",
            "Education_Level",
        ]
        self.params = params or {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
        }
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._encoder: OneHotEncoder | None = None
        self._model: LogisticRegression | None = None
        self._feature_names: list[str] = []
        self._num_features: list[str] = []
        self._fitted = False

    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Apply fold-local standardization, optional PCA on numerics, and one-hot encoding."""
        X_proc = X.copy()
        cat_cols = [c for c in self.categorical_features if c in X_proc.columns]
        num_cols = [c for c in self._feature_names if c not in cat_cols]

        if fit:
            self._num_features = num_cols
            if num_cols:
                self._scaler = StandardScaler()
                X_num_scaled = self._scaler.fit_transform(X_proc[num_cols])
                if self.use_pca:
                    self._pca = PCA(n_components=self.pca_variance_ratio, random_state=42)
                    X_num_proc = self._pca.fit_transform(X_num_scaled)
                else:
                    X_num_proc = X_num_scaled
            else:
                X_num_proc = np.empty((len(X), 0))

            if cat_cols:
                self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                X_cat_proc = self._encoder.fit_transform(X_proc[cat_cols])
            else:
                X_cat_proc = np.empty((len(X), 0))
        else:
            if num_cols and self._scaler is not None:
                X_num_scaled = self._scaler.transform(X_proc[num_cols])
                if self.use_pca and self._pca is not None:
                    X_num_proc = self._pca.transform(X_num_scaled)
                else:
                    X_num_proc = X_num_scaled
            else:
                X_num_proc = np.empty((len(X), 0))

            if cat_cols and self._encoder is not None:
                X_cat_proc = self._encoder.transform(X_proc[cat_cols])
            else:
                X_cat_proc = np.empty((len(X), 0))

        if X_num_proc.shape[1] > 0 and X_cat_proc.shape[1] > 0:
            return np.hstack([X_num_proc, X_cat_proc])
        elif X_num_proc.shape[1] > 0:
            return X_num_proc
        else:
            return X_cat_proc

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> LogisticRegressionAdapter:
        """Fit preprocessing pipeline and Logistic Regression on training fold data."""
        self._feature_names = list(X.columns)
        X_proc = self._preprocess(X, fit=True)
        y_arr = np.asarray(y, dtype=int)

        self._model = LogisticRegression(**self.params)
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
        """Serialize model artifacts and preprocessors to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        artifact_file = directory / f"{self.model_id}_adapter.pkl"
        with open(artifact_file, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "scaler": self._scaler,
                    "pca": self._pca,
                    "encoder": self._encoder,
                    "params": self.params,
                    "use_pca": self.use_pca,
                    "pca_variance_ratio": self.pca_variance_ratio,
                    "categorical_features": self.categorical_features,
                    "feature_names": self._feature_names,
                },
                f,
            )

        metadata = self.get_metadata()
        metadata["artifact_file"] = str(artifact_file.name)
        return metadata

    @classmethod
    def load(cls, directory: Path, model_id: str = "logistic_regression") -> LogisticRegressionAdapter:
        """Load serialized Logistic Regression adapter from disk."""
        directory = Path(directory)
        artifact_file = directory / f"{model_id}_adapter.pkl"

        adapter = cls(model_id=model_id)
        if artifact_file.exists():
            with open(artifact_file, "rb") as f:
                state = pickle.load(f)
            adapter._model = state["model"]
            adapter._scaler = state["scaler"]
            adapter._pca = state["pca"]
            adapter._encoder = state["encoder"]
            adapter.params = state["params"]
            adapter.use_pca = state["use_pca"]
            adapter.pca_variance_ratio = state["pca_variance_ratio"]
            adapter.categorical_features = state["categorical_features"]
            adapter._feature_names = state["feature_names"]
            adapter._fitted = True
        return adapter

    def get_metadata(self) -> dict[str, Any]:
        """Return model metadata, PCA settings, and hyperparameters."""
        meta: dict[str, Any] = {
            "name": self.name,
            "model_id": self.model_id,
            "use_pca": self.use_pca,
            "pca_variance_ratio": self.pca_variance_ratio,
            "fitted": self._fitted,
            "params": self.params,
            "categorical_features": self.categorical_features,
            "feature_names": self._feature_names,
        }
        if self._pca is not None:
            meta["pca_n_components_"] = int(self._pca.n_components_)
            meta["pca_explained_variance_ratio_sum"] = float(
                np.sum(self._pca.explained_variance_ratio_)
            )
        return meta
