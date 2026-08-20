"""TabNet Model Adapter for Employee Resignation Prediction (Member 6).

Conforms to the ModelAdapter protocol with fold-local categorical embeddings
and numerical feature standardization.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelAdapter


class TabNetAdapter:
    """TabNet model adapter with fold-local preprocessing."""

    def __init__(
        self,
        name: str = "TabNet",
        model_id: str = "tabnet",
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.params = params or {
            "n_d": 16,
            "n_a": 16,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 0.001,
            "seed": 42,
            "verbose": 0,
        }
        self.categorical_features = categorical_features or [
            "Department",
            "Gender",
            "Job_Title",
            "Education_Level",
        ]
        self._model: Any = None
        self._scaler: Any = None
        self._category_maps: dict[str, dict[str, int]] = {}
        self._feature_names: list[str] = []
        self._cat_idxs: list[int] = []
        self._cat_dims: list[int] = []
        self._fitted = False

    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Preprocess inputs: scale numerics and encode categoricals to integer indices."""
        from sklearn.preprocessing import StandardScaler

        X_proc = X.copy()
        num_cols = [c for c in self._feature_names if c not in self.categorical_features]

        if fit:
            self._scaler = StandardScaler()
            if num_cols:
                self._scaler.fit(X_proc[num_cols])

            self._cat_idxs = []
            self._cat_dims = []
            self._category_maps = {}

            for idx, col in enumerate(self._feature_names):
                if col in self.categorical_features:
                    unique_vals = sorted(X_proc[col].dropna().unique())
                    mapping = {val: i for i, val in enumerate(unique_vals)}
                    self._category_maps[col] = mapping
                    self._cat_idxs.append(idx)
                    self._cat_dims.append(len(mapping))

        # Transform numerics
        if num_cols and self._scaler is not None:
            X_proc[num_cols] = self._scaler.transform(X_proc[num_cols])

        # Transform categoricals
        for col in self.categorical_features:
            if col in X_proc.columns:
                mapping = self._category_maps.get(col, {})
                X_proc[col] = X_proc[col].map(mapping).fillna(0).astype("int64")

        return X_proc[self._feature_names].values.astype(np.float32)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        max_epochs: int = 50,
        batch_size: int = 1024,
    ) -> TabNetAdapter:
        """Fit fold-local preprocessors and TabNet classifier on training fold data."""
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError as exc:
            raise ImportError(
                "pytorch-tabnet is required for TabNetAdapter. Install with pip install pytorch-tabnet."
            ) from exc

        self._feature_names = list(X.columns)
        X_mat = self._preprocess(X, fit=True)
        y_arr = np.asarray(y, dtype=int)

        self._model = TabNetClassifier(
            cat_idxs=self._cat_idxs,
            cat_dims=self._cat_dims,
            **self.params,
        )
        self._model.fit(
            X_train=X_mat,
            y_train=y_arr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            virtual_batch_size=128,
            drop_last=False,
        )
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive class probabilities for input features."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Adapter is not fitted. Call fit() first.")

        X_mat = self._preprocess(X, fit=False)
        probs = self._model.predict_proba(X_mat)[:, 1]
        return np.asarray(probs, dtype=float)

    def save(self, directory: Path) -> dict[str, Any]:
        """Serialize TabNet model and preprocessors to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        artifact_file = directory / "tabnet_adapter.pkl"
        with open(artifact_file, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "scaler": self._scaler,
                    "category_maps": self._category_maps,
                    "cat_idxs": self._cat_idxs,
                    "cat_dims": self._cat_dims,
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
    def load(cls, directory: Path) -> TabNetAdapter:
        """Load serialized TabNet adapter from disk."""
        directory = Path(directory)
        artifact_file = directory / "tabnet_adapter.pkl"

        adapter = cls()
        if artifact_file.exists():
            with open(artifact_file, "rb") as f:
                state = pickle.load(f)
            adapter._model = state["model"]
            adapter._scaler = state["scaler"]
            adapter._category_maps = state["category_maps"]
            adapter._cat_idxs = state["cat_idxs"]
            adapter._cat_dims = state["cat_dims"]
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
