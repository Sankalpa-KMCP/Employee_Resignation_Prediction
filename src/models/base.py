"""Shared Model Adapter interface and cross-validation result structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class ModelAdapter(Protocol):
    """Unified Protocol for Employee Resignation Prediction model adapters.

    Every model adapter (wrapping GBDT, GAM, or Neural architectures) must
    implement this interface to participate in the shared 5-fold cross-validation
    pipeline.

    Responsibilities
    ----------------
    1. Own fold-local preprocessing (encoders, scalers, tokenizers).
    2. Fit parameters strictly on outer training fold data.
    3. Predict positive-class probabilities bounded in [0.0, 1.0].
    4. Provide serialization and reload capabilities.
    5. Expose model metadata and hyperparameter records.
    """

    name: str
    model_id: str

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> ModelAdapter:
        """Fit fold-local preprocessing and model parameters on training fold data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe containing the approved contract predictors.
        y : pd.Series or np.ndarray
            Binary target labels (0 or 1 / False or True).

        Returns
        -------
        ModelAdapter
            Fitted adapter instance (self).
        """
        ...

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Predict positive class probabilities for input features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe containing the approved contract predictors.

        Returns
        -------
        np.ndarray
            1D float array of predicted probabilities of shape `(N,)`
            with values strictly bounded in [0.0, 1.0].
        """
        ...

    def save(
        self,
        directory: Path,
    ) -> dict[str, Any]:
        """Serialize model parameters, fold-local preprocessors, and metadata.

        Parameters
        ----------
        directory : Path
            Directory to save model artifacts.

        Returns
        -------
        dict[str, Any]
            Metadata dictionary summarizing saved artifacts.
        """
        ...

    @classmethod
    def load(
        cls,
        directory: Path,
    ) -> ModelAdapter:
        """Instantiate and load a trained adapter from disk.

        Parameters
        ----------
        directory : Path
            Directory containing saved model artifacts.

        Returns
        -------
        ModelAdapter
            Restored model adapter instance.
        """
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata including hyperparameters, feature lists, and timing.

        Returns
        -------
        dict[str, Any]
            Model metadata dictionary.
        """
        ...


@dataclass(frozen=True)
class OOFResult:
    """Container for out-of-fold cross-validation evaluation results."""

    model_name: str
    model_id: str
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary_metrics: dict[str, Any]
    fit_time_seconds_total: float
    predict_time_seconds_total: float
    metadata: dict[str, Any] = field(default_factory=dict)
