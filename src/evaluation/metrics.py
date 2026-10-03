"""Shared binary-classification evaluation metrics.

All model workflows should use these functions so that model comparison
is performed with identical metric definitions.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Ranking metrics do not use this threshold; precision, recall and F1 do.
DEFAULT_THRESHOLD = 0.50


def validate_binary_targets(y_true: Iterable[int], name: str = "y_true") -> None:
    """Reject invalid original labels before any integer conversion or model fit."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if len(y_true) == 0:
        raise ValueError(f"{name} cannot be empty.")

    if y_true.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain numeric binary labels 0 and 1.")

    if not np.isfinite(y_true).all():
        raise ValueError(f"{name} contains non-finite values.")

    unique_values = set(np.unique(y_true))

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"{name} must contain only binary labels 0 and 1."
        )


def _validate_probabilities(
    y_probability: np.ndarray,
) -> None:
    """Validate predicted probabilities."""
    if y_probability.ndim != 1:
        raise ValueError(
            "y_probability must be one-dimensional."
        )

    if len(y_probability) == 0:
        raise ValueError(
            "y_probability cannot be empty."
        )

    if not np.isfinite(y_probability).all():
        raise ValueError(
            "y_probability contains non-finite values."
        )

    if (
        (y_probability < 0).any()
        or (y_probability > 1).any()
    ):
        raise ValueError(
            "All predicted probabilities must be between 0 and 1."
        )


def probabilities_to_predictions(
    y_probability: Iterable[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Convert probabilities into binary predictions.

    A probability greater than or equal to the threshold is classified
    as the positive class.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = np.asarray(
        y_probability,
        dtype=float,
    )

    _validate_probabilities(probabilities)

    return (
        probabilities >= threshold
    ).astype(int)


def calculate_binary_metrics(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    y_pred: Iterable[int] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | int]:
    """Calculate the project's standard binary-classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.

    y_probability:
        Predicted probability for the positive class.

    y_pred:
        Optional binary class predictions. If omitted, predictions are
        generated from y_probability using the supplied threshold.

    threshold:
        Probability threshold used when y_pred is not supplied.

    Returns
    -------
    dict
        Standard model-evaluation metrics.

    Notes
    -----
    Average Precision (AP) is the project's primary ranking metric.
    It is intentionally named ``average_precision`` rather than
    ``pr_auc`` to avoid conflating AP with trapezoidal integration of
    the precision-recall curve.

    Interpret AP against the positive prevalence of the evaluated rows: high
    accuracy alone can hide poor minority-class performance. AP and ROC-AUC
    measure ranking; precision, recall and F1 depend on the supplied predictions
    or threshold. Brier score measures probability error rather than ranking.
    """
    # Validate the original values before converting to integers. Otherwise
    # values such as 0.8 would be silently truncated to valid-looking labels.
    true_values = np.asarray(y_true)

    probabilities = np.asarray(
        y_probability,
        dtype=float,
    )

    validate_binary_targets(true_values)
    true_values = true_values.astype(int)
    _validate_probabilities(probabilities)

    if len(true_values) != len(probabilities):
        raise ValueError(
            "y_true and y_probability must have the same length."
        )

    if y_pred is None:
        predictions = probabilities_to_predictions(
            probabilities,
            threshold=threshold,
        )
    else:
        predictions = np.asarray(y_pred)

        validate_binary_targets(predictions, name="y_pred")
        predictions = predictions.astype(int)

        if len(predictions) != len(true_values):
            raise ValueError(
                "y_true and y_pred must have the same length."
            )

    # ROC-AUC is undefined when only one target class is present.
    if len(np.unique(true_values)) < 2:
        raise ValueError(
            "Both target classes must be present to calculate "
            "ROC-AUC."
        )

    tn, fp, fn, tp = confusion_matrix(
        true_values,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "average_precision": float(
            average_precision_score(
                true_values,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                true_values,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                true_values,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true_values,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                true_values,
                predictions,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_values,
                predictions,
            )
        ),
        "accuracy": float(
            accuracy_score(
                true_values,
                predictions,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                true_values,
                probabilities,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_count": int(
            predictions.sum()
        ),
        "predicted_positive_rate": float(
            predictions.mean()
        ),
    }
