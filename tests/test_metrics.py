import numpy as np
import pytest

from src.evaluation.metrics import (
    calculate_binary_metrics,
    probabilities_to_predictions,
)


EXPECTED_METRIC_KEYS = {
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "accuracy",
    "brier_score",
    "tn",
    "fp",
    "fn",
    "tp",
    "predicted_positive_count",
    "predicted_positive_rate",
}


def test_probability_thresholding():
    probabilities = [
        0.10,
        0.49,
        0.50,
        0.90,
    ]

    predictions = probabilities_to_predictions(
        probabilities
    )

    expected = np.array(
        [0, 0, 1, 1]
    )

    np.testing.assert_array_equal(
        predictions,
        expected,
    )


def test_custom_probability_threshold():
    probabilities = [
        0.20,
        0.40,
        0.60,
        0.80,
    ]

    predictions = probabilities_to_predictions(
        probabilities,
        threshold=0.70,
    )

    expected = np.array(
        [0, 0, 0, 1]
    )

    np.testing.assert_array_equal(
        predictions,
        expected,
    )


def test_metric_output_contains_expected_keys():
    y_true = [0, 0, 1, 1]
    y_probability = [
        0.10,
        0.40,
        0.60,
        0.90,
    ]

    metrics = calculate_binary_metrics(
        y_true,
        y_probability,
    )

    assert set(metrics) == EXPECTED_METRIC_KEYS


def test_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_probability = [
        0.01,
        0.10,
        0.90,
        0.99,
    ]

    metrics = calculate_binary_metrics(
        y_true,
        y_probability,
    )

    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)

    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tp"] == 2


def test_confusion_matrix_counts():
    y_true = [
        0,
        0,
        1,
        1,
    ]

    y_probability = [
        0.10,
        0.80,
        0.20,
        0.90,
    ]

    metrics = calculate_binary_metrics(
        y_true,
        y_probability,
    )

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1


def test_predicted_positive_count_and_rate():
    y_true = [
        0,
        0,
        1,
        1,
    ]

    y_probability = [
        0.20,
        0.60,
        0.70,
        0.30,
    ]

    metrics = calculate_binary_metrics(
        y_true,
        y_probability,
    )

    assert metrics["predicted_positive_count"] == 2

    assert metrics[
        "predicted_positive_rate"
    ] == pytest.approx(0.50)


def test_explicit_predictions_are_respected():
    y_true = [0, 0, 1, 1]

    y_probability = [
        0.10,
        0.20,
        0.80,
        0.90,
    ]

    custom_predictions = [
        0,
        1,
        1,
        1,
    ]

    metrics = calculate_binary_metrics(
        y_true,
        y_probability,
        y_pred=custom_predictions,
    )

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 0
    assert metrics["tp"] == 2


def test_invalid_probability_above_one_is_rejected():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        probabilities_to_predictions(
            [0.20, 1.20]
        )


def test_invalid_probability_below_zero_is_rejected():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        probabilities_to_predictions(
            [-0.10, 0.50]
        )


def test_nan_probability_is_rejected():
    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        probabilities_to_predictions(
            [0.20, np.nan]
        )


def test_invalid_threshold_is_rejected():
    with pytest.raises(
        ValueError,
        match="threshold",
    ):
        probabilities_to_predictions(
            [0.20, 0.80],
            threshold=1.50,
        )


def test_non_binary_target_is_rejected():
    y_true = [
        0,
        1,
        2,
    ]

    y_probability = [
        0.10,
        0.50,
        0.90,
    ]

    with pytest.raises(
        ValueError,
        match="binary",
    ):
        calculate_binary_metrics(
            y_true,
            y_probability,
        )


def test_length_mismatch_is_rejected():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        calculate_binary_metrics(
            [0, 1, 1],
            [0.20, 0.80],
        )


def test_single_target_class_is_rejected():
    with pytest.raises(
        ValueError,
        match="Both target classes",
    ):
        calculate_binary_metrics(
            [0, 0, 0],
            [0.10, 0.20, 0.30],
        )


def test_brier_score_is_zero_for_perfect_probabilities():
    metrics = calculate_binary_metrics(
        [0, 0, 1, 1],
        [0.0, 0.0, 1.0, 1.0],
    )

    assert metrics["brier_score"] == pytest.approx(
        0.0
    )


def test_metric_values_are_finite():
    metrics = calculate_binary_metrics(
        [0, 0, 1, 1],
        [0.10, 0.40, 0.60, 0.90],
    )

    for key in [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "accuracy",
        "brier_score",
        "predicted_positive_rate",
    ]:
        assert np.isfinite(metrics[key])