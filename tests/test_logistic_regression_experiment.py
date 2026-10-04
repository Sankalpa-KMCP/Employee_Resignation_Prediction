"""Leakage and evidence checks for the supplementary probability model."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.evaluation.inputs import load_development_data
from src.evaluation.oof_validation import validate_oof_predictions
from src.models.logistic_regression_experiment import (
    DIRECTORY, MODEL_NAME, build_pipeline, fold_value_proxy,
)

ROOT = Path(__file__).resolve().parents[1]


def test_training_only_scaling_and_unknown_category():
    train = pd.DataFrame({"number": [0., 2., 4., 6.], "category": ["a", "b", "a", "b"]})
    model = build_pipeline(list(train.columns), ["category"], 0.1, None).fit(train, [0, 1, 0, 1])
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"]
    assert scaler.mean_[0] == 3
    assert np.isfinite(model.predict_proba(pd.DataFrame({"number": [1000.], "category": ["new"]}))).all()
    assert scaler.mean_[0] == 3


def test_proxy_uses_training_bounds_and_handles_constants():
    train = pd.DataFrame({"Performance_Score": [1., 5.], "Monthly_Salary": [100., 500.],
                          "Years_At_Company": [2., 2.]})
    valid = pd.DataFrame({"Performance_Score": [3., 50.], "Monthly_Salary": [300., -100.],
                          "Years_At_Company": [2., 2.]})
    assert np.allclose(fold_value_proxy(train, valid), [1 / 3, 1 / 3])


def test_saved_selection_and_oof_evidence():
    out = ROOT / DIRECTORY
    source, registry = load_development_data(ROOT, usecols=[])
    predictions = pd.read_csv(out / "oof_predictions.csv")
    validate_oof_predictions(predictions, registry, source_data=source, expected_model_name=MODEL_NAME)
    evidence = json.loads((out / "selection_evidence.json").read_text())
    assert len(evidence) == 5
    for fold in evidence:
        assert fold["winner"] == min(fold["candidates"], key=lambda c: c["inner_log_loss"])["candidate_id"]
        assert fold["inner_train_rows"] + fold["inner_validation_rows"] == fold["outer_train_rows"]
        assert fold["outer_train_rows"] + fold["outer_validation_rows"] == len(source)
    summary = json.loads((out / "summary_metrics.json").read_text())
    assert np.isclose(log_loss(predictions.y_true, predictions.y_probability), summary["log_loss"])
    assert summary["formal_holdout_accessed"] is False
    assert np.allclose(predictions.combined_score, predictions.y_probability * predictions.retention_value_proxy)
