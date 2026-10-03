"""Regression checks for subgroup rates, ranking denominators and holdout gates."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyClassifier

from src.evaluation.fairness import subgroup_classification_metrics, top_k_representation
from src.evaluation.calibration import reliability_table
from src.evaluation.holdout import holdout_readiness, positive_probability, run_final_holdout
from src.prioritization.ranking import evaluate_ranking, select_top_k


def test_reliability_boundaries_empty_bins_and_invalid_targets():
    table = reliability_table([0, 1, 1], [0, .5, 1], n_bins=4)
    assert table["count"].tolist() == [1, 0, 1, 1]
    assert np.isnan(table.loc[1, "observed_rate"])
    with pytest.raises(ValueError, match="binary"):
        reliability_table([0, .8], [.1, .2])
    with pytest.raises(ValueError, match="integer"):
        reliability_table([0, 1], [.1, .2], n_bins=2.5)


def test_undefined_subgroup_rates_stay_missing():
    frame = pd.DataFrame({"group": ["negative", "negative", "positive"],
                          "y_true": [0, 0, 1], "y_probability": [.1, .2, .1]})
    metrics = subgroup_classification_metrics(frame, "group", min_count=1).set_index("group")
    assert np.isnan(metrics.loc["negative", "recall"])
    assert np.isnan(metrics.loc["negative", "precision"])
    assert np.isnan(metrics.loc["positive", "false_positive_rate"])
    assert metrics.loc["positive", "recall"] == 0
    assert metrics.loc["negative", "negative_count"] == 2
    assert metrics.average_precision.isna().all()


def test_missing_subgroup_preserved_and_score_ties_use_id():
    frame = pd.DataFrame({"Employee_ID": [3, 1, 2, 4], "group": [None, "A", None, "A"],
                          "y_true": [0, 1, 1, 0], "y_probability": [.5] * 4})
    assert select_top_k(frame, "y_probability", .5).Employee_ID.tolist() == [1, 2]
    assert select_top_k(frame.iloc[::-1], "y_probability", .5).Employee_ID.tolist() == [1, 2]
    table = top_k_representation(frame, "group", .5)
    assert table.selected_count.sum() == 2 and table.population_count.sum() == 4
    assert table.representation_ratio.eq(1).all()


def test_unobserved_categorical_groups_are_omitted():
    frame = pd.DataFrame({"group": pd.Categorical(["A", "A"], categories=["A", "unused"]),
                          "y_true": [0, 1], "y_probability": [.1, .8]})
    table = top_k_representation(frame, "group", .5)
    assert table["group"].tolist() == ["A"]
    assert table.selected_count.sum() == 1


def test_ranking_sorts_numeric_scores_instead_of_string_order():
    frame = pd.DataFrame({"Employee_ID": [1, 2], "score": ["2", "10"]})
    assert select_top_k(frame, "score", .5).Employee_ID.tolist() == [2]


def test_weighted_capture_uses_all_resigned_proxy_value():
    frame = pd.DataFrame({"y_true": [1, 1, 0, 0], "score": [.9, .1, .8, .2],
                          "value": [.2, .8, .6, .4]})
    result = evaluate_ranking(frame, "score", .5, value_col="value")
    assert result["captured_positive_count"] == 1
    assert result["recall_at_capacity"] == .5
    assert result["value_weighted_recall_at_capacity"] == pytest.approx(.2)
    assert result["selected_proxy_value_share"] == pytest.approx(.4)
    frame["y_true"] = 0
    result = evaluate_ranking(frame, "score", .5, value_col="value")
    assert np.isnan(result["recall_at_capacity"])
    assert np.isnan(result["lift_at_capacity"])
    assert np.isnan(result["value_weighted_recall_at_capacity"])


@pytest.mark.parametrize("score", [np.nan, np.inf])
def test_invalid_ranking_scores_rejected(score):
    with pytest.raises(ValueError, match="finite"):
        evaluate_ranking(pd.DataFrame({"y_true": [0, 1], "score": [score, .2]}), "score", .5)


def test_disabled_and_unfrozen_holdout_never_read_employee_data(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Readiness or disabled evaluation accessed employee data/artifacts")
    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    checks, manifest = holdout_readiness(tmp_path)
    assert not checks.passed.all() and manifest is None
    with pytest.raises(RuntimeError, match="disabled"):
        run_final_holdout(tmp_path)
    with pytest.raises(RuntimeError, match="blocked"):
        run_final_holdout(tmp_path, authorized=True)
    assert not (tmp_path / "results/final_holdout").exists()


@pytest.fixture
def frozen_synthetic_project(tmp_path):
    """A fully task-owned fixture; no real holdout or fitted employee model is used."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "models/frozen").mkdir(parents=True)
    raw = pd.DataFrame({"Employee_ID": range(25), "f": np.arange(25),
                        "Resigned": ["development-not-parsed"] * 5 + [0, 1] * 10})
    split = pd.DataFrame({"Employee_ID": range(25), "partition": ["development"] * 5 + ["holdout"] * 20,
                          "cv_fold": [1, 2, 3, 4, 5] + [None] * 20})
    raw.to_csv(tmp_path / "raw.csv", index=False)
    split.to_csv(tmp_path / "split.csv", index=False)
    def digest(path, normalized=False):
        data = path.read_bytes()
        return hashlib.sha256(data.replace(b"\r\n", b"\n") if normalized else data).hexdigest()
    contract = {"dataset": {"path": "raw.csv", "sha256": digest(tmp_path / "raw.csv", True)},
                "split_registry": {"path": "split.csv", "sha256": digest(tmp_path / "split.csv", True),
                                   "development_rows": 5, "formal_holdout_rows": 20},
                "primary_model_features": ["f"]}
    (tmp_path / "configs/feature_contract.yaml").write_text(yaml.safe_dump(contract))
    (tmp_path / "configs/analysis_registry.yaml").write_text("models: {}")
    decisions = {
        "comparison/comparison_summary.json": {"final_model_frozen": True, "final_model": "SyntheticDummy"},
        "calibration/calibration_decision.json": {"probability_source_frozen": True, "probability_source": "raw"},
        "fairness/fairness_summary.json": {"diagnostics_complete": True},
        "prioritization/prioritization_decision.json": {"policy_frozen": True, "holdout_scope": "prediction_only"},
    }
    for relative, document in decisions.items():
        path = tmp_path / "results" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
    pipeline = DummyClassifier(strategy="prior").fit(pd.DataFrame({"f": range(5)}), [0, 1, 0, 0, 0])
    joblib.dump(pipeline, tmp_path / "models/frozen/pipeline.joblib")
    manifest = {"freeze_id": "synthetic-test", "pipeline_frozen": True, "model": "SyntheticDummy",
                "training_partition": "development", "training_rows": 5, "feature_columns": ["f"],
                "pipeline_path": "models/frozen/pipeline.joblib", "threshold": .5,
                "threshold_source": "synthetic development policy", "probability_source": "raw",
                "includes_frozen_probability_source": True, "downstream_policy_frozen": True,
                "formal_holdout_used_for_selection": False, "historical_audit_exposure_disclosed": True,
                "holdout_scope": "prediction_only"}
    artifacts = ["configs/feature_contract.yaml", "configs/analysis_registry.yaml", "models/frozen/pipeline.joblib",
                 *["results/" + relative for relative in decisions]]
    manifest["artifacts"] = {relative: digest(tmp_path / relative) for relative in artifacts}
    (tmp_path / "models/frozen/freeze.json").write_text(json.dumps(manifest))
    return tmp_path


def test_synthetic_holdout_evaluates_once_and_keeps_no_employee_export(frozen_synthetic_project):
    root = frozen_synthetic_project
    (root / "results/final_holdout").mkdir()
    (root / "results/final_holdout/.gitkeep").touch()
    checks, _ = holdout_readiness(root)
    assert checks.passed.all(), checks.loc[~checks.passed].to_dict("records")
    metrics, reliability, intervals = run_final_holdout(root, authorized=True)
    assert metrics["rows"] == 20 and metrics["roc_auc"] == .5
    assert reliability["count"].sum() == 20 and len(intervals) == 7
    assert (root / "results/final_holdout/assessment_receipt.json").exists()
    assert not (root / "results/final_holdout/holdout_predictions.csv").exists()
    with pytest.raises(RuntimeError, match="previous assessment"):
        run_final_holdout(root, authorized=True)


def test_tampered_frozen_artifact_blocks_before_reading_raw(frozen_synthetic_project, monkeypatch):
    root = frozen_synthetic_project
    (root / "models/frozen/pipeline.joblib").write_bytes(b"changed")
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pytest.fail("Raw data accessed before freeze verification"))
    with pytest.raises(RuntimeError, match="artifact"):
        run_final_holdout(root, authorized=True)


def test_readiness_rejects_extra_raw_data_artifact_without_reading_it(frozen_synthetic_project, monkeypatch):
    root = frozen_synthetic_project
    path = root / "models/frozen/freeze.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["raw.csv"] = "not-read"
    path.write_text(json.dumps(manifest))
    original_read = Path.read_bytes
    def guarded_read(path):
        if path.name == "raw.csv":
            pytest.fail("Safe readiness attempted to read raw employee data")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    checks, _ = holdout_readiness(root)
    assert not checks.passed.all()
    assert "all freeze inputs bound" in checks.loc[~checks.passed, "check"].tolist()


def test_predictor_failure_leaves_started_receipt_and_blocks_retry(frozen_synthetic_project, monkeypatch):
    class Broken:
        def predict_proba(self, X):
            raise RuntimeError("synthetic prediction failure")
    monkeypatch.setattr(joblib, "load", lambda *args: Broken())
    root = frozen_synthetic_project
    with pytest.raises(RuntimeError, match="prediction failure"):
        run_final_holdout(root, authorized=True)
    receipt = json.loads((root / "results/final_holdout/assessment_receipt.json").read_text())
    assert receipt["status"] == "started"
    with pytest.raises(RuntimeError, match="previous assessment"):
        run_final_holdout(root, authorized=True)


def test_positive_class_column_follows_classes_order():
    class ReversedClassifier:
        classes_ = np.array([1, 0])
        def predict_proba(self, X):
            return np.tile([.2, .8], (len(X), 1))
    assert np.all(positive_probability(ReversedClassifier(), pd.DataFrame({"f": [1, 2]})) == .2)
