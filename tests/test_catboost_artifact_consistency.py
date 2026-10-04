"""Artifact consistency and numerical reconciliation tests for CatBoost (Step 5).

Treats `results/models/catboost/catboost_oof_predictions.csv` as the canonical source
for raw OOF metrics. Recomputes all raw metrics directly and verifies that:
- summary CSV, summary JSON, fold metrics, confusion matrix, figures, and manifest agree
- clear separation exists between raw OOF, calibrated OOF, experiments, and baselines
- generic ambiguous labels like 'CatBoost Brier score' are avoided
- automated assertions fail when artifacts disagree
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.inputs import load_saved_model_evidence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "models" / "catboost"
MODELS_DIR = PROJECT_ROOT / "models" / "catboost"
TOLERANCE = 1e-5


@pytest.fixture
def raw_oof_df():
    """Load canonical raw OOF predictions."""
    path = RESULTS_DIR / "catboost_oof_predictions.csv"
    assert path.exists(), f"Missing canonical OOF predictions at {path}"
    return pd.read_csv(path)


@pytest.fixture
def calibrated_oof_df():
    """Load cross-fitted calibrated OOF predictions."""
    path = RESULTS_DIR / "catboost_calibrated_oof_predictions.csv"
    assert path.exists(), f"Missing calibrated OOF predictions at {path}"
    return pd.read_csv(path)


def test_oof_predictions_dual_copy_parity(raw_oof_df):
    """Verify oof_predictions.csv and catboost_oof_predictions.csv are strictly identical."""
    std_oof = pd.read_csv(RESULTS_DIR / "oof_predictions.csv")
    assert len(raw_oof_df) == len(std_oof) == 80_000
    assert (raw_oof_df["Employee_ID"] == std_oof["Employee_ID"]).all()
    assert np.allclose(raw_oof_df["y_probability"], std_oof["y_probability"], atol=1e-9)
    assert (raw_oof_df["y_true"] == std_oof["y_true"]).all()


def test_recomputed_raw_metrics_match_summary_json(raw_oof_df):
    """Recompute all raw metrics directly from canonical OOF and compare with summary JSON."""
    y_true = raw_oof_df["y_true"].to_numpy(dtype=int)
    y_prob = raw_oof_df["y_probability"].to_numpy(dtype=float)
    y_pred = raw_oof_df["y_pred_0_5"].to_numpy(dtype=int)

    calc_ap = average_precision_score(y_true, y_prob)
    calc_roc = roc_auc_score(y_true, y_prob)
    calc_brier = brier_score_loss(y_true, y_prob)
    calc_logloss = log_loss(y_true, y_prob)
    calc_bal_acc = balanced_accuracy_score(y_true, y_pred)
    calc_prec = precision_score(y_true, y_pred, zero_division=0)
    calc_rec = recall_score(y_true, y_pred, zero_division=0)
    calc_f1 = f1_score(y_true, y_pred, zero_division=0)
    calc_pos_rate = float(np.mean(y_pred))

    with open(RESULTS_DIR / "catboost_summary_metrics.json", "r", encoding="utf-8") as f:
        meta_json = json.load(f)

    raw_json = meta_json["group_a_raw_catboost_oof_metrics"]
    assert abs(raw_json["raw_oof_pr_auc"] - calc_ap) < TOLERANCE
    assert abs(raw_json["raw_oof_roc_auc"] - calc_roc) < TOLERANCE
    assert abs(raw_json["raw_oof_brier_score"] - calc_brier) < TOLERANCE
    assert abs(raw_json["raw_oof_log_loss"] - calc_logloss) < TOLERANCE
    assert abs(raw_json["raw_oof_balanced_accuracy"] - calc_bal_acc) < TOLERANCE
    assert abs(raw_json["raw_oof_precision"] - calc_prec) < TOLERANCE
    assert abs(raw_json["raw_oof_recall"] - calc_rec) < TOLERANCE
    assert abs(raw_json["raw_oof_f1"] - calc_f1) < TOLERANCE
    assert abs(raw_json["raw_oof_pred_pos_rate"] - calc_pos_rate) < TOLERANCE


def test_recomputed_raw_metrics_match_summary_csv(raw_oof_df):
    """Verify Group A row in summary CSV exactly matches recomputed OOF metrics."""
    y_true = raw_oof_df["y_true"].to_numpy(dtype=int)
    y_prob = raw_oof_df["y_probability"].to_numpy(dtype=float)
    y_pred = raw_oof_df["y_pred_0_5"].to_numpy(dtype=int)

    calc_ap = average_precision_score(y_true, y_prob)
    calc_roc = roc_auc_score(y_true, y_prob)
    calc_brier = brier_score_loss(y_true, y_prob)
    calc_logloss = log_loss(y_true, y_prob)
    calc_bal_acc = balanced_accuracy_score(y_true, y_pred)
    calc_precision = precision_score(y_true, y_pred, zero_division=0)
    calc_recall = recall_score(y_true, y_pred, zero_division=0)
    calc_f1 = f1_score(y_true, y_pred, zero_division=0)
    calc_pos_rate = float(np.mean(y_pred))

    sum_csv = pd.read_csv(RESULTS_DIR / "catboost_summary_metrics.csv")
    group_a = sum_csv[sum_csv["result_group"].str.startswith("A.")].iloc[0]

    assert abs(group_a["pr_auc"] - calc_ap) < TOLERANCE
    assert abs(group_a["roc_auc"] - calc_roc) < TOLERANCE
    assert abs(group_a["brier_score"] - calc_brier) < TOLERANCE
    assert abs(group_a["log_loss"] - calc_logloss) < TOLERANCE
    assert abs(group_a["balanced_accuracy"] - calc_bal_acc) < TOLERANCE
    assert abs(group_a["precision"] - calc_precision) < TOLERANCE
    assert abs(group_a["recall"] - calc_recall) < TOLERANCE
    assert abs(group_a["f1"] - calc_f1) < TOLERANCE
    assert abs(group_a["pred_pos_rate"] - calc_pos_rate) < TOLERANCE


def test_standardized_summary_metrics_json_interface():
    """Verify load_saved_model_evidence validates CatBoost results without discrepancy."""
    preds, folds, summary = load_saved_model_evidence(PROJECT_ROOT, "CatBoost", RESULTS_DIR)
    assert len(preds) == 80_000
    assert len(folds) == 5
    assert "pooled_average_precision" in summary
    assert "pooled_roc_auc" in summary
    assert "pooled_brier_score" in summary


def test_confusion_matrix_reconciliation(raw_oof_df):
    """Verify catboost_confusion_matrix.csv matches exact counts from canonical OOF."""
    y_true = raw_oof_df["y_true"].to_numpy(dtype=int)
    y_pred = raw_oof_df["y_pred_0_5"].to_numpy(dtype=int)
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    cm_df = pd.read_csv(RESULTS_DIR / "catboost_confusion_matrix.csv", index_col=0)
    assert cm_df.iloc[0, 0] == tn_c
    assert cm_df.iloc[0, 1] == fp_c
    assert cm_df.iloc[1, 0] == fn_c
    assert cm_df.iloc[1, 1] == tp_c


def test_calibrated_oof_reconciliation(calibrated_oof_df):
    """Verify calibrated OOF predictions match Group B metrics in summary JSON and CSV."""
    y_true = calibrated_oof_df["y_true"].to_numpy(dtype=int)
    cal_prob = calibrated_oof_df["calibrated_y_probability"].to_numpy(dtype=float)

    calc_cal_ap = average_precision_score(y_true, cal_prob)
    calc_cal_roc = roc_auc_score(y_true, cal_prob)
    calc_cal_brier = brier_score_loss(y_true, cal_prob)
    calc_cal_logloss = log_loss(y_true, cal_prob)
    cal_pred = (cal_prob >= 0.50).astype(int)
    calc_cal_precision = precision_score(y_true, cal_pred, zero_division=0)
    calc_cal_recall = recall_score(y_true, cal_pred, zero_division=0)
    calc_cal_f1 = f1_score(y_true, cal_pred, zero_division=0)
    calc_cal_balanced_accuracy = balanced_accuracy_score(y_true, cal_pred)
    calc_cal_pos_rate = float(np.mean(cal_pred))

    with open(RESULTS_DIR / "catboost_summary_metrics.json", "r", encoding="utf-8") as f:
        meta_json = json.load(f)

    cal_json = meta_json["group_b_cross_fitted_calibrated_oof_metrics"]
    assert abs(cal_json["cal_oof_pr_auc"] - calc_cal_ap) < TOLERANCE
    assert abs(cal_json["cal_oof_roc_auc"] - calc_cal_roc) < TOLERANCE
    assert abs(cal_json["cal_oof_brier_score"] - calc_cal_brier) < TOLERANCE
    assert abs(cal_json["cal_oof_log_loss"] - calc_cal_logloss) < TOLERANCE
    assert abs(cal_json["cal_oof_precision"] - calc_cal_precision) < TOLERANCE
    assert abs(cal_json["cal_oof_recall"] - calc_cal_recall) < TOLERANCE
    assert abs(cal_json["cal_oof_f1"] - calc_cal_f1) < TOLERANCE
    assert abs(cal_json["cal_oof_balanced_accuracy"] - calc_cal_balanced_accuracy) < TOLERANCE
    assert abs(cal_json["cal_oof_pred_pos_rate"] - calc_cal_pos_rate) < TOLERANCE

    summary_csv = pd.read_csv(RESULTS_DIR / "catboost_summary_metrics.csv")
    calibrated_row = summary_csv.loc[
        summary_csv["result_group"].str.startswith("B.")
    ].iloc[0]
    for column, value in {
        "pr_auc": calc_cal_ap,
        "roc_auc": calc_cal_roc,
        "brier_score": calc_cal_brier,
        "log_loss": calc_cal_logloss,
        "precision": calc_cal_precision,
        "recall": calc_cal_recall,
        "f1": calc_cal_f1,
        "balanced_accuracy": calc_cal_balanced_accuracy,
        "pred_pos_rate": calc_cal_pos_rate,
    }.items():
        assert abs(calibrated_row[column] - value) < TOLERANCE


def test_calibrated_oof_identity_and_bounds_match_raw_oof(raw_oof_df, calibrated_oof_df):
    """Calibrated predictions must be cross-fitted companions, never replacements."""
    raw_sorted = raw_oof_df.sort_values("Employee_ID").reset_index(drop=True)
    calibrated_sorted = calibrated_oof_df.sort_values("Employee_ID").reset_index(drop=True)
    assert len(calibrated_sorted) == len(raw_sorted) == 80_000
    for column in ("Employee_ID", "cv_fold", "y_true"):
        assert (calibrated_sorted[column] == raw_sorted[column]).all()
    assert calibrated_sorted["Employee_ID"].is_unique
    probabilities = calibrated_sorted["calibrated_y_probability"].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert not np.allclose(
        probabilities,
        raw_sorted["y_probability"].to_numpy(dtype=float),
        atol=1e-12,
    ), "Calibrated output must be stored separately from raw OOF probabilities."


def test_baseline_benchmark_agrees_with_canonical_catboost_predictions(raw_oof_df, calibrated_oof_df):
    """The benchmark must report the same raw and calibrated canonical metrics."""
    benchmark = pd.read_csv(RESULTS_DIR / "catboost_baseline_benchmark.csv")
    expected = {
        "CatBoost (Raw Nested OOF)": raw_oof_df["y_probability"].to_numpy(dtype=float),
        "CatBoost (Calibrated OOF)": calibrated_oof_df["calibrated_y_probability"].to_numpy(dtype=float),
    }
    y_true = raw_oof_df["y_true"].to_numpy(dtype=int)
    for model_name, probabilities in expected.items():
        row = benchmark.loc[benchmark["model"] == model_name].iloc[0]
        y_pred = (probabilities >= 0.50).astype(int)
        values = {
            "pr_auc": average_precision_score(y_true, probabilities),
            "roc_auc": roc_auc_score(y_true, probabilities),
            "brier_score": brier_score_loss(y_true, probabilities),
            "log_loss": log_loss(y_true, probabilities),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "predicted_positive_rate": float(np.mean(y_pred)),
        }
        for column, value in values.items():
            assert abs(row[column] - value) < TOLERANCE


def test_fold_metrics_mean_reconciliation():
    """Verify mean of fold_metrics.csv matches fold_metrics summary in JSON."""
    fm = pd.read_csv(RESULTS_DIR / "catboost_fold_metrics.csv")
    with open(RESULTS_DIR / "catboost_summary_metrics.json", "r", encoding="utf-8") as f:
        meta_json = json.load(f)

    f_json = meta_json["fold_metrics"]
    assert abs(fm["pr_auc"].mean() - f_json["pr_auc_mean"]) < TOLERANCE
    assert abs(fm["roc_auc"].mean() - f_json["roc_auc_mean"]) < TOLERANCE
    assert abs(fm["brier_score"].mean() - f_json["brier_score_mean"]) < TOLERANCE


def test_all_manifest_artifacts_exist_and_non_empty():
    """Verify all artifacts listed in catboost_output_manifest.csv exist and are non-empty."""
    manifest_df = pd.read_csv(RESULTS_DIR / "catboost_output_manifest.csv")
    for _, row in manifest_df.iterrows():
        p = PROJECT_ROOT / row["relative_path"]
        assert p.exists(), f"Manifest artifact missing: {p}"
        assert p.stat().st_size > 0, f"Manifest artifact is empty: {p}"


def test_exact_topk_capacity_counts(raw_oof_df):
    """Verify exact top-k counts (1%, 2%, 5%, 10%, 20%, 50%) and threshold analysis file."""
    thresh_path = RESULTS_DIR / "catboost_threshold_analysis.csv"
    assert thresh_path.exists(), f"Missing threshold analysis at: {thresh_path}"
    thresh_df = pd.read_csv(thresh_path)
    
    expected_capacities = {
        "Top 1%": 800,
        "Top 2%": 1600,
        "Top 5%": 4000,
        "Top 10%": 8000,
        "Top 20%": 16000,
        "Top 50%": 40000,
    }
    
    for cap_label, expected_count in expected_capacities.items():
        row = thresh_df.loc[thresh_df["capacity_target"] == cap_label]
        assert len(row) == 1, f"Missing row for {cap_label}"
        flagged = int(row.iloc[0]["flagged_count"])
        assert flagged == expected_count, f"Flagged count mismatch for {cap_label}: expected {expected_count}, got {flagged}"
        
        # Verify deterministic top-k ranking on raw OOF dataframe
        sorted_df = raw_oof_df.sort_values(
            by=["y_probability", "Employee_ID"],
            ascending=[False, True]
        ).reset_index(drop=True)
        top_slice = sorted_df.iloc[:expected_count]
        assert len(top_slice) == expected_count
        actual_tp = int(top_slice["y_true"].sum())
        reported_tp = int(row.iloc[0]["true_positives_captured"])
        assert actual_tp == reported_tp, f"TP captured mismatch for {cap_label}: expected {actual_tp}, reported {reported_tp}"


def test_operational_confusion_matrix_reconciliation(raw_oof_df):
    """Verify catboost_operational_confusion_matrix.csv exists and matches OOF predictions."""
    csv_path = RESULTS_DIR / "catboost_operational_confusion_matrix.csv"
    assert csv_path.exists(), f"Missing operational confusion matrix at: {csv_path}"
    op_df = pd.read_csv(csv_path)
    assert len(op_df) == 4

    # Verify baseline 0.50 cutoff preserves exact contract reconciliation
    row_05 = op_df.loc[op_df["policy"] == "Default 0.50 Cutoff"].iloc[0]
    assert int(row_05["true_negatives_tn"]) == 71992
    assert int(row_05["false_positives_fp"]) == 0
    assert int(row_05["false_negatives_fn"]) == 8008
    assert int(row_05["true_positives_tp"]) == 0

    # Verify Base-Rate Prevalence row captures non-zero true positives
    row_prev = op_df.loc[op_df["policy"] == "Base-Rate Prevalence"].iloc[0]
    assert int(row_prev["true_positives_tp"]) == 4240
    assert int(row_prev["false_negatives_fn"]) == 3768
    assert int(row_prev["false_positives_fp"]) == 37848
    assert int(row_prev["true_negatives_tn"]) == 34144
    assert abs(float(row_prev["recall"]) - 0.5295) < TOLERANCE

    # Verify Top 10% Budget Capacity row reconciles with topk analysis
    row_top10 = op_df.loc[op_df["policy"] == "Top 10% Budget Capacity"].iloc[0]
    assert int(row_top10["true_positives_tp"]) == 874
    assert int(row_top10["false_positives_fp"]) == 7126
    assert int(row_top10["flagged_count"]) == 8000

    # Verify comparison figure exists
    fig_path = RESULTS_DIR / "figure_operational_confusion_matrix.png"
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


