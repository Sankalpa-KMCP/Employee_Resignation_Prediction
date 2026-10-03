"""Explainable Boosting Machine (EBM) Model Adapter (Member 5).

Conforms to the ModelAdapter protocol with Generalized Additive Models
plus pairwise interactions (GA2M).

This single-fit adapter is separate from the canonical nested workflow in ebm_experiments.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelAdapter


class EBMAdapter:
    """Explainable Boosting Machine (EBM) model adapter."""

    def __init__(
        self,
        name: str = "EBM",
        model_id: str = "ebm",
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        """Configure a single EBM adapter and its nominal feature names without fitting.

        The formal six-variety workflow is implemented separately in ebm_experiments.
        """
        self.name = name
        self.model_id = model_id
        self.params = params or {
            "max_bins": 256,
            "max_interaction_bins": 32,
            "interactions": 10,
            "learning_rate": 0.04,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.categorical_features = categorical_features or [
            "Department",
            "Gender",
            "Job_Title",
            "Education_Level",
        ]
        self._model: Any = None
        self._feature_names: list[str] = []
        self._feature_types: list[str] = []
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> EBMAdapter:
        """Fit EBM classifier on training fold data."""
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as exc:
            raise ImportError(
                "interpret is required for EBMAdapter. Install with pip install interpret."
            ) from exc

        self._feature_names = list(X.columns)
        # Nominal feature types let EBM model categories without imposing numeric distances.
        self._feature_types = [
            "nominal" if col in self.categorical_features else "continuous"
            for col in self._feature_names
        ]

        X_proc = X.copy()
        for col in self.categorical_features:
            if col in X_proc.columns:
                X_proc[col] = X_proc[col].astype(str)

        y_arr = np.asarray(y, dtype=int)

        self._model = ExplainableBoostingClassifier(
            feature_names=self._feature_names,
            feature_types=self._feature_types,
            **self.params,
        )
        self._model.fit(X_proc, y_arr)
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
        """Serialize EBM model artifacts and metadata to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        artifact_file = directory / "ebm_adapter.pkl"
        with open(artifact_file, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "params": self.params,
                    "categorical_features": self.categorical_features,
                    "feature_names": self._feature_names,
                    "feature_types": self._feature_types,
                },
                f,
            )

        metadata = self.get_metadata()
        metadata["artifact_file"] = str(artifact_file.name)
        return metadata

    @classmethod
    def load(cls, directory: Path) -> EBMAdapter:
        """Load serialized EBM adapter from disk."""
        directory = Path(directory)
        artifact_file = directory / "ebm_adapter.pkl"

        adapter = cls()
        if artifact_file.exists():
            with open(artifact_file, "rb") as f:
                state = pickle.load(f)
            adapter._model = state["model"]
            adapter.params = state["params"]
            adapter.categorical_features = state["categorical_features"]
            adapter._feature_names = state["feature_names"]
            adapter._feature_types = state["feature_types"]
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
