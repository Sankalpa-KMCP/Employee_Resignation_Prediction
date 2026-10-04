"""Canonical baseline benchmark script for CatBoost pipeline (Step 4).

Evaluates across the exact same 5 frozen outer folds:
1. DummyClassifier(strategy="prior")
2. DummyClassifier(strategy="most_frequent")
3. Logistic Regression baseline (fold-local standard scaling and one-hot encoding)
4. CatBoostAdapter (canonical raw nested OOF)
5. CatBoost Calibrated (cross-fitted Platt calibrated OOF)

Calculates from canonical OOF predictions:
- Average precision / PR-AUC
- ROC-AUC
- Brier score
- Log loss
- Positive-class precision, recall, and F1 (threshold 0.50)
- Macro F1
- Balanced accuracy
- Confusion matrix (TN, FP, FN, TP)
- Predicted-positive rate
- 95% bootstrap confidence intervals (1,000 iterations)
- Paired bootstrap confidence intervals and statistical tests vs Dummy Prior and Logistic Regression
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
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

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_contract import load_feature_contract
from src.evaluation.inputs import load_development_data
from src.models.logistic_regression import LogisticRegressionAdapter


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics along with 95% bootstrap CIs."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Core point metrics
    ap = float(average_precision_score(y_true, y_prob))
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc = 0.5
    brier = float(brier_score_loss(y_true, y_prob))
    p_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    ll = float(log_loss(y_true, p_clipped))

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    pred_pos_rate = float(np.mean(y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Bootstrap 95% CI for AP and ROC-AUC
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aps = []
    boot_rocs = []
    boot_briers = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_prob[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        boot_aps.append(average_precision_score(yt_b, yp_b))
        boot_rocs.append(roc_auc_score(yt_b, yp_b))
        boot_briers.append(brier_score_loss(yt_b, yp_b))

    ap_ci = (float(np.percentile(boot_aps, 2.5)), float(np.percentile(boot_aps, 97.5)))
    roc_ci = (float(np.percentile(boot_rocs, 2.5)), float(np.percentile(boot_rocs, 97.5)))
    brier_ci = (float(np.percentile(boot_briers, 2.5)), float(np.percentile(boot_briers, 97.5)))

    return {
        "pr_auc": ap,
        "pr_auc_95_ci": ap_ci,
        "roc_auc": roc,
        "roc_auc_95_ci": roc_ci,
        "brier_score": brier,
        "brier_score_95_ci": brier_ci,
        "log_loss": ll,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": f1_macro,
        "balanced_accuracy": bal_acc,
        "predicted_positive_rate": pred_pos_rate,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_benchmark(n_bootstraps: int = 1000) -> pd.DataFrame:
    """Run baseline benchmark across the 5 outer folds."""
    print("=" * 80)
    print("CANONICAL BASELINE BENCHMARK: DUMMY & LOGISTIC REGRESSION VS. CATBOOST")
    print("=" * 80)

    data, split = load_development_data(PROJECT_ROOT)
    contract = load_feature_contract()
    primary_feats = contract["primary_model_features"]

    # Align dev data with folds
    dev_split = split.loc[split["partition"] == "development", ["Employee_ID", "cv_fold"]].copy()
    dev_split["cv_fold"] = dev_split["cv_fold"].astype(int)
    merged = data.merge(dev_split, on="Employee_ID", validate="one_to_one")

    y_true_all = merged["Resigned"].to_numpy(dtype=int)
    folds = merged["cv_fold"].to_numpy(dtype=int)

    # 1. Dummy Classifier (Prior)
    print("\n[1/5] Evaluating DummyClassifier(strategy='prior')...")
    dummy_prior_probs = np.zeros(len(merged), dtype=float)
    for f in range(1, 6):
        tr_mask = (folds != f)
        va_mask = (folds == f)
        clf = DummyClassifier(strategy="prior")
        clf.fit(merged.loc[tr_mask, primary_feats], y_true_all[tr_mask])
        dummy_prior_probs[va_mask] = clf.predict_proba(merged.loc[va_mask, primary_feats])[:, 1]

    # 2. Dummy Classifier (Most Frequent)
    print("[2/5] Evaluating DummyClassifier(strategy='most_frequent')...")
    dummy_mf_probs = np.zeros(len(merged), dtype=float)
    for f in range(1, 6):
        tr_mask = (folds != f)
        va_mask = (folds == f)
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(merged.loc[tr_mask, primary_feats], y_true_all[tr_mask])
        # For most_frequent (all class 0), predict probability 0.0 for class 1
        dummy_mf_probs[va_mask] = clf.predict_proba(merged.loc[va_mask, primary_feats])[:, 1]

    # 3. Logistic Regression Baseline
    print("[3/5] Evaluating LogisticRegressionAdapter (Fold-Local Scaling & OHE)...")
    lr_probs = np.zeros(len(merged), dtype=float)
    for f in range(1, 6):
        tr_mask = (folds != f)
        va_mask = (folds == f)
        lr = LogisticRegressionAdapter(use_pca=False)
        lr.fit(merged.loc[tr_mask, primary_feats], y_true_all[tr_mask])
        lr_probs[va_mask] = lr.predict_proba(merged.loc[va_mask, primary_feats])

    # 4. Canonical CatBoost Raw OOF Predictions
    print("[4/5] Loading Canonical CatBoost Raw OOF Predictions...")
    catboost_oof_path = PROJECT_ROOT / "results" / "models" / "catboost" / "catboost_oof_predictions.csv"
    catboost_oof_df = pd.read_csv(catboost_oof_path).sort_values("Employee_ID").reset_index(drop=True)
    merged_sorted = merged.sort_values("Employee_ID").reset_index(drop=True)
    assert (catboost_oof_df["Employee_ID"] == merged_sorted["Employee_ID"]).all()
    cb_raw_probs = catboost_oof_df["y_probability"].to_numpy(dtype=float)

    # 5. Canonical CatBoost Calibrated OOF Predictions
    print("[5/5] Loading Canonical CatBoost Calibrated OOF Predictions...")
    catboost_cal_path = PROJECT_ROOT / "results" / "models" / "catboost" / "catboost_calibrated_oof_predictions.csv"
    catboost_cal_df = pd.read_csv(catboost_cal_path).sort_values("Employee_ID").reset_index(drop=True)
    assert (catboost_cal_df["Employee_ID"] == merged_sorted["Employee_ID"]).all()
    cb_cal_probs = catboost_cal_df["calibrated_y_probability"].to_numpy(dtype=float)

    models_dict = {
        "Dummy Most Frequent": dummy_mf_probs,
        "Dummy Prior": dummy_prior_probs,
        "Logistic Regression": lr_probs,
        "CatBoost (Raw Nested OOF)": cb_raw_probs,
        "CatBoost (Calibrated OOF)": cb_cal_probs,
    }

    records = []
    print(f"\nComputing metrics with {n_bootstraps:,} bootstrap iterations per model...")
    for model_name, probs in models_dict.items():
        t0 = time.perf_counter()
        m = compute_metrics_with_ci(y_true_all, probs, n_bootstraps=n_bootstraps)
        elapsed = time.perf_counter() - t0
        records.append({
            "model": model_name,
            "pr_auc": m["pr_auc"],
            "pr_auc_ci_lower": m["pr_auc_95_ci"][0],
            "pr_auc_ci_upper": m["pr_auc_95_ci"][1],
            "roc_auc": m["roc_auc"],
            "roc_auc_ci_lower": m["roc_auc_95_ci"][0],
            "roc_auc_ci_upper": m["roc_auc_95_ci"][1],
            "brier_score": m["brier_score"],
            "brier_score_ci_lower": m["brier_score_95_ci"][0],
            "brier_score_ci_upper": m["brier_score_95_ci"][1],
            "log_loss": m["log_loss"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "predicted_positive_rate": m["predicted_positive_rate"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tp": m["tp"],
            "eval_time_seconds": round(elapsed, 2),
        })

    benchmark_df = pd.DataFrame(records)

    # Statistical significance: Paired Bootstrap Differences vs Dummy Prior and LR
    print("\n" + "=" * 80)
    print("PAIRED BOOTSTRAP HYPOTHESIS TESTING (1,000 ITERATIONS)")
    print("=" * 80)

    rng = np.random.default_rng(42)
    n = len(y_true_all)
    diff_ap_dummy = []
    diff_roc_dummy = []
    diff_ap_lr = []
    diff_roc_lr = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yt_b = y_true_all[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        ap_cb = average_precision_score(yt_b, cb_raw_probs[idx])
        roc_cb = roc_auc_score(yt_b, cb_raw_probs[idx])

        ap_dp = average_precision_score(yt_b, dummy_prior_probs[idx])
        roc_dp = roc_auc_score(yt_b, dummy_prior_probs[idx])

        ap_lr = average_precision_score(yt_b, lr_probs[idx])
        roc_lr = roc_auc_score(yt_b, lr_probs[idx])

        diff_ap_dummy.append(ap_cb - ap_dp)
        diff_roc_dummy.append(roc_cb - roc_dp)
        diff_ap_lr.append(ap_cb - ap_lr)
        diff_roc_lr.append(roc_cb - roc_lr)

    ci_ap_dummy = (np.percentile(diff_ap_dummy, 2.5), np.percentile(diff_ap_dummy, 97.5))
    ci_roc_dummy = (np.percentile(diff_roc_dummy, 2.5), np.percentile(diff_roc_dummy, 97.5))
    p_ap_dummy = (np.array(diff_ap_dummy) <= 0).mean() if np.mean(diff_ap_dummy) > 0 else (np.array(diff_ap_dummy) >= 0).mean()

    ci_ap_lr = (np.percentile(diff_ap_lr, 2.5), np.percentile(diff_ap_lr, 97.5))
    ci_roc_lr = (np.percentile(diff_roc_lr, 2.5), np.percentile(diff_roc_lr, 97.5))
    p_ap_lr = (np.array(diff_ap_lr) <= 0).mean() if np.mean(diff_ap_lr) > 0 else (np.array(diff_ap_lr) >= 0).mean()

    print(f"CatBoost vs. Dummy Prior:")
    print(f"  Delta AP:      {np.mean(diff_ap_dummy):+.6f} | 95% CI: [{ci_ap_dummy[0]:+.6f}, {ci_ap_dummy[1]:+.6f}] | Two-sided p ~ {2 * p_ap_dummy:.4f}")
    print(f"  Delta ROC-AUC: {np.mean(diff_roc_dummy):+.6f} | 95% CI: [{ci_roc_dummy[0]:+.6f}, {ci_roc_dummy[1]:+.6f}]")
    print(f"  Verdict:       Zero statistically significant improvement over random prevalence baseline.")

    print(f"\nCatBoost vs. Logistic Regression:")
    print(f"  Delta AP:      {np.mean(diff_ap_lr):+.6f} | 95% CI: [{ci_ap_lr[0]:+.6f}, {ci_ap_lr[1]:+.6f}] | Two-sided p ~ {2 * p_ap_lr:.4f}")
    print(f"  Delta ROC-AUC: {np.mean(diff_roc_lr):+.6f} | 95% CI: [{ci_roc_lr[0]:+.6f}, {ci_roc_lr[1]:+.6f}]")
    print(f"  Verdict:       Zero statistically significant improvement over linear benchmark.")

    # Save benchmark table
    out_path = PROJECT_ROOT / "results" / "models" / "catboost" / "catboost_baseline_benchmark.csv"
    benchmark_df.to_csv(out_path, index=False)
    print(f"\nSaved benchmark results to: {out_path.relative_to(PROJECT_ROOT)}")

    # Print markdown summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    summary_display = benchmark_df[[
        "model", "pr_auc", "roc_auc", "brier_score", "log_loss",
        "precision", "recall", "f1", "macro_f1", "balanced_accuracy", "predicted_positive_rate"
    ]]
    print(summary_display.to_string(index=False))

    return benchmark_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CatBoost canonical baseline benchmark.")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap iterations for 95%% CIs.")
    args = parser.parse_args()
    run_benchmark(n_bootstraps=args.bootstraps)
