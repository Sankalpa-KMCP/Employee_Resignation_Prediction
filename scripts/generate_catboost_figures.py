"""Comprehensive Diagnostic Visualizations Generator for CatBoost Pipeline (Step 10).

Generates all 10 reproducible diagnostic figures with explicit data lineage labeling:
1. Figure 01: Feature Importance Ranking (with chance-level warning)
2. Figure 02: Precision-Recall Curves (Raw OOF vs. Calibrated OOF vs. Dummy Prior)
3. Figure 03: Confusion Matrix Heatmap (Raw OOF at 0.50 threshold)
4. Figure 04: ROC Curves (Raw OOF vs. Calibrated OOF vs. 0.50 Chance Line)
5. Figure 05: Probability Distribution Comparison (Raw OOF vs. Calibrated OOF)
6. Figure 06: Reliability Diagram / Calibration Curve (Raw vs. Calibrated vs. Perfect)
7. Figure 07: Top-k Capacity Lift Curve (Raw OOF vs. Random Selection Baseline)
8. Figure 08: Inner Training vs. Validation Learning Curves (Logloss & PRAUC)
9. Figure 09: Fold-Level Score Distributions (Outer Folds 1-5 Stability)
10. Figure 10: artifact-derived CatBoost feature importance bar chart
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "models" / "catboost"
MODELS_DIR = PROJECT_ROOT / "models" / "catboost"


def generate_all_figures() -> list[Path]:
    """Generate and save all 10 diagnostic figures to results/models/catboost/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    # Load canonical OOF files
    oof_df = pd.read_csv(RESULTS_DIR / "catboost_oof_predictions.csv")
    cal_df = pd.read_csv(RESULTS_DIR / "catboost_calibrated_oof_predictions.csv")

    y_true = oof_df["y_true"].to_numpy(dtype=int)
    raw_probs = oof_df["y_probability"].to_numpy(dtype=float)
    cal_probs = cal_df["calibrated_y_probability"].to_numpy(dtype=float)
    prevalence = float(np.mean(y_true))
    raw_ap = float(average_precision_score(y_true, raw_probs))
    calibrated_ap = float(average_precision_score(y_true, cal_probs))
    raw_roc_auc = float(roc_auc_score(y_true, raw_probs))
    calibrated_roc_auc = float(roc_auc_score(y_true, cal_probs))
    raw_brier = float(brier_score_loss(y_true, raw_probs))
    calibrated_brier = float(brier_score_loss(y_true, cal_probs))

    # Set common styling
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # 1. Figure 01: Feature Importance (with chance-level warning)
    # -------------------------------------------------------------------------
    print("Generating Figure 01: Feature Importance...")
    imp_df = pd.read_csv(RESULTS_DIR / "catboost_feature_importance.csv")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    colors = ["#1f77b4" if imp > 0 else "#cccccc" for imp in imp_df["importance"][::-1]]
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=colors, alpha=0.85)
    ax.set_title(
        "CatBoost — Final Development Model Feature Importance\n"
        "[WARNING: Feature importance reflects random tree noise; uninformative when ROC-AUC ~ 0.50]",
        fontsize=11, fontweight="bold", pad=12, color="#333333"
    )
    ax.set_xlabel("Mean Absolute Feature Contribution (PredictionValuesChange)", fontsize=10)
    ax.set_ylabel("Predictor", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    f1_path = RESULTS_DIR / "figure_01_catboost_feature_importance.png"
    fig.savefig(f1_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(f1_path)

    # -------------------------------------------------------------------------
    # 2. Figure 02: Precision-Recall Curves
    # -------------------------------------------------------------------------
    print("Generating Figure 02: Precision-Recall Curves...")
    p_raw, r_raw, _ = precision_recall_curve(y_true, raw_probs)
    p_cal, r_cal, _ = precision_recall_curve(y_true, cal_probs)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.plot(r_raw, p_raw, color="#1f77b4", lw=2, label=f"CatBoost Raw OOF (AP = {raw_ap:.5f})")
    ax.plot(r_cal, p_cal, color="#2ca02c", lw=1.8, linestyle="-.", label=f"CatBoost Calibrated OOF (AP = {calibrated_ap:.5f})")
    ax.axhline(prevalence, color="#d62728", linestyle="--", lw=1.5, label=f"Dummy Prior Baseline (AP = {prevalence:.5f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("CatBoost — Precision-Recall Curves [Raw OOF vs. Calibrated OOF vs. Baseline]", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 0.30])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    f2_path = RESULTS_DIR / "figure_02_catboost_precision_recall_curves.png"
    fig.savefig(f2_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(f2_path)

    # -------------------------------------------------------------------------
    # 3. Figure 03: Confusion Matrix Heatmap (0.50 Threshold)
    # -------------------------------------------------------------------------
    print("Generating Figure 03: Confusion Matrix Heatmap...")
    cm_df = pd.read_csv(RESULTS_DIR / "catboost_confusion_matrix.csv", index_col=0)
    cm = cm_df.to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    cax = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_title("CatBoost — OOF Confusion Matrix (0.50 Decision Cutoff)\n[Raw Nested-CV OOF Predictions]", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stayed (0)", "Resigned (1)"])
    ax.set_yticklabels(["Stayed (0)", "Resigned (1)"])
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("Actual Class", fontsize=10)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.colorbar(cax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    f3_path = RESULTS_DIR / "figure_03_catboost_confusion_matrix.png"
    fig.savefig(f3_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(f3_path)

    # -------------------------------------------------------------------------
    # 4. Figure 04: ROC Curves
    # -------------------------------------------------------------------------
    print("Generating Figure 04: ROC Curves...")
    fpr_raw, tpr_raw, _ = roc_curve(y_true, raw_probs)
    fpr_cal, tpr_cal, _ = roc_curve(y_true, cal_probs)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.plot(fpr_raw, tpr_raw, color="#1f77b4", lw=2, label=f"CatBoost Raw OOF (ROC-AUC = {raw_roc_auc:.5f})")
    ax.plot(fpr_cal, tpr_cal, color="#2ca02c", lw=1.8, linestyle="-.", label=f"CatBoost Calibrated OOF (ROC-AUC = {calibrated_roc_auc:.5f})")
    ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.5, label="Chance Line (ROC-AUC = 0.50000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("CatBoost — Receiver Operating Characteristic (ROC) Curves\n[Raw OOF vs. Calibrated OOF vs. Chance]", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    f4_path = RESULTS_DIR / "figure_04_catboost_roc_curve.png"
    fig.savefig(f4_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(f4_path)

    # -------------------------------------------------------------------------
    # 5. Figure 05: Probability Distribution Comparison
    # -------------------------------------------------------------------------
    print("Generating Figure 05: Probability Distribution Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=160, sharey=False)

    ax1.hist(raw_probs[y_true == 0], bins=50, alpha=0.6, color="#1f77b4", label="Actual Stayed (0)", density=True)
    ax1.hist(raw_probs[y_true == 1], bins=50, alpha=0.6, color="#d62728", label="Actual Resigned (1)", density=True)
    ax1.axvline(0.50, color="black", linestyle="--", lw=1.2, label="Threshold 0.50")
    ax1.set_title(f"Raw OOF Predicted Probabilities\n[Brier = {raw_brier:.5f}]", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Predicted Probability", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.hist(cal_probs[y_true == 0], bins=50, alpha=0.6, color="#2ca02c", label="Actual Stayed (0)", density=True)
    ax2.hist(cal_probs[y_true == 1], bins=50, alpha=0.6, color="#ff7f0e", label="Actual Resigned (1)", density=True)
    ax2.axvline(prevalence, color="red", linestyle="--", lw=1.2, label=f"Prevalence ({prevalence:.4f})")
    ax2.set_title(f"Platt-Calibrated OOF Predicted Probabilities\n[Brier = {calibrated_brier:.5f}]", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Calibrated Probability", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("CatBoost — Out-of-Fold Probability Calibration Diagnostics", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    f5_path = RESULTS_DIR / "figure_05_catboost_probability_distribution.png"
    fig.savefig(f5_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(f5_path)

    # -------------------------------------------------------------------------
    # 6. Figure 06: Reliability Diagram
    # -------------------------------------------------------------------------
    print("Generating Figure 06: Reliability Diagram...")
    prob_true_raw, prob_pred_raw = calibration_curve(y_true, raw_probs, n_bins=10, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, cal_probs, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.5, label="Perfect Calibration")
    ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#1f77b4", lw=2, label=f"Raw OOF (Brier = {raw_brier:.4f})")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#2ca02c", lw=2, label=f"Platt Calibrated (Brier = {calibrated_brier:.4f})")
    ax.axhline(prevalence, color="red", linestyle=":", lw=1.2, label=f"Natural Prevalence ({prevalence:.4f})")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Empirical Fraction of Positives (Resigned)", fontsize=11)
    ax.set_title("CatBoost — Reliability Diagram / Calibration Curve\n[Raw OOF vs. Calibrated OOF]", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    f6_path = RESULTS_DIR / "figure_06_catboost_calibration_curve.png"
    fig.savefig(f6_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(f6_path)

    # -------------------------------------------------------------------------
    # 7. Figure 07: Top-k Capacity Lift Curve
    # -------------------------------------------------------------------------
    print("Generating Figure 07: Top-k Capacity Lift Curve...")
    thresh_df = pd.read_csv(RESULTS_DIR / "catboost_threshold_analysis.csv")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(thresh_df["capacity_fraction"] * 100, thresh_df["lift_vs_prevalence"], "o-", color="#1f77b4", lw=2.2, label="CatBoost Raw OOF Lift")
    ax.axhline(1.0, color="#d62728", linestyle="--", lw=1.8, label="Random Prior Baseline (Lift = 1.00)")
    for _, row in thresh_df.iterrows():
        cap_pct = row["capacity_fraction"] * 100
        lift = row["lift_vs_prevalence"]
        ax.annotate(f"{lift:.2f}x", (cap_pct, lift), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Targeting Capacity (% of Development Population)", fontsize=11)
    ax.set_ylabel(f"Lift vs. Natural Prevalence ({prevalence:.4f})", fontsize=11)
    ax.set_title("CatBoost — Operational Retention Outreach Lift Curve\n[Top-k Selection Performance vs. Random Outreach]", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim([0.7, 1.3])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    f7_path = RESULTS_DIR / "figure_07_catboost_topk_lift_curve.png"
    fig.savefig(f7_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(f7_path)

    # -------------------------------------------------------------------------
    # 8. Figure 08: Inner Learning Curves
    # -------------------------------------------------------------------------
    print("Generating Figure 08: Inner Learning Curves...")
    evidence_path = RESULTS_DIR / "catboost_fold_selection_evidence.json"
    with evidence_path.open("r", encoding="utf-8") as handle:
        fold_evidence = json.load(handle)

    # The adapter persists CatBoost's actual evals_result for every candidate.
    # A missing series is a data-lineage error, never an invitation to simulate one.
    curves = []
    for fold_key, fold_data in fold_evidence.items():
        if not isinstance(fold_data, dict):
            continue
        for candidate in fold_data.get("candidates", {}).values():
            if not isinstance(candidate, dict):
                continue
            evaluation = candidate.get("evals_result", {})
            learn = evaluation.get("learn", {})
            validation = evaluation.get("validation", evaluation.get("validation_0", {}))
            train_logloss = learn.get("Logloss")
            validation_logloss = validation.get("Logloss")
            train_prauc = learn.get("PRAUC")
            validation_prauc = validation.get("PRAUC")
            if all(series is not None for series in (train_logloss, validation_logloss, train_prauc, validation_prauc)):
                curves.append((train_logloss, validation_logloss, train_prauc, validation_prauc))

    # Fallback to empirical inner traces from catboost_info/catboost_training.json if available
    training_json_path = PROJECT_ROOT / "catboost_info" / "catboost_training.json"
    if not curves and training_json_path.exists():
        with training_json_path.open("r", encoding="utf-8") as tj_file:
            tj = json.load(tj_file)
            tr_pr = [it["learn"][0] for it in tj.get("iterations", [])]
            tr_ll = [it["learn"][1] for it in tj.get("iterations", [])]
            va_pr = [it["test"][0] for it in tj.get("iterations", [])]
            va_ll = [it["test"][1] for it in tj.get("iterations", [])]
            if tr_pr and va_pr:
                curves.append((tr_ll, va_ll, tr_pr, va_pr))

    if not curves:
        raise FileNotFoundError(
            "Empirical inner learning curves are unavailable in "
            "catboost_fold_selection_evidence.json or catboost_training.json. Re-run the leakage-safe "
            "nested pipeline to persist CatBoost evals_result; figures are not simulated."
        )

    min_length = min(len(series) for curve in curves for series in curve)
    iterations = np.arange(1, min_length + 1)
    train_logloss, val_logloss, train_prauc, val_prauc = (
        np.mean([np.asarray(curve[index][:min_length], dtype=float) for curve in curves], axis=0)
        for index in range(4)
    )

    fig, (ax_ll, ax_pr) = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    ax_ll.plot(iterations, train_logloss, label="Inner Train Logloss", color="#1f77b4", lw=1.8)
    ax_ll.plot(iterations, val_logloss, label="Inner Validation Logloss", color="#ff7f0e", lw=1.8, linestyle="--")
    with (RESULTS_DIR / "catboost_run_metadata.json").open("r", encoding="utf-8") as handle:
        run_metadata = json.load(handle)
    selected_candidate = run_metadata.get("candidate_id", "selected candidate")
    ax_ll.set_title(f"Logloss Learning Curve ({selected_candidate}; empirical inner traces)", fontsize=10, fontweight="bold")
    ax_ll.set_xlabel("Iteration (Tree Count)", fontsize=10)
    ax_ll.set_ylabel("Logloss", fontsize=10)
    ax_ll.grid(True, linestyle="--", alpha=0.3)
    ax_ll.legend()

    ax_pr.plot(iterations, train_prauc, label="Inner Train PRAUC", color="#1f77b4", lw=1.8)
    ax_pr.plot(iterations, val_prauc, label="Inner Validation PRAUC", color="#d62728", lw=1.8, linestyle="--")
    ax_pr.axhline(prevalence, color="gray", linestyle="--", lw=1.2, label=f"Prevalence ({prevalence:.4f})")
    ax_pr.set_title("PRAUC Learning Curve (empirical inner traces)", fontsize=10, fontweight="bold")
    ax_pr.set_xlabel("Iteration (Tree Count)", fontsize=10)
    ax_pr.set_ylabel("PRAUC", fontsize=10)
    ax_pr.grid(True, linestyle="--", alpha=0.3)
    ax_pr.legend()

    fig.suptitle("CatBoost — Inner Training vs. Validation Learning Curves [Candidate Selection Diagnostic]", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    f8_path = RESULTS_DIR / "figure_08_catboost_inner_learning_curves.png"
    fig.savefig(f8_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(f8_path)

    # -------------------------------------------------------------------------
    # 9. Figure 09: Fold-Level Score Distributions
    # -------------------------------------------------------------------------
    print("Generating Figure 09: Fold-Level Score Distributions...")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    fold_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for f in range(1, 6):
        sub_p = oof_df.loc[oof_df["cv_fold"] == f, "y_probability"]
        ax.hist(sub_p, bins=40, density=True, alpha=0.45, color=fold_colors[f - 1], label=f"Outer Fold {f} (Mean = {sub_p.mean():.4f})")

    ax.set_title("CatBoost — Out-of-Fold Score Distributions Across 5 Outer Folds\n[Raw OOF Predictions Showing Effect of Fold-Selected Candidates]", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Probability", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    f9_path = RESULTS_DIR / "figure_09_catboost_fold_score_distributions.png"
    fig.savefig(f9_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(f9_path)

    # -------------------------------------------------------------------------
    # 10. Figure 10: Artifact-derived feature importance
    # -------------------------------------------------------------------------
    print("Generating Figure 10: Artifact-Derived Feature Importance...")
    # Do not label generic feature importance as SHAP. The persisted importance
    # artifact is the only supported input until genuine fold-safe SHAP values
    # are explicitly exported by the modelling workflow.
    importance_df = pd.read_csv(RESULTS_DIR / "catboost_feature_importance.csv")
    shap_df = importance_df.rename(columns={"importance": "feature_importance"}).sort_values("feature_importance")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    colors = ["#1f77b4" if s > 0 else "#cccccc" for s in shap_df["feature_importance"]]
    ax.barh(shap_df["feature"], shap_df["feature_importance"], color=colors, alpha=0.85)
    ax.set_title("CatBoost — Feature Importance (Artifact-Derived Only)\n[Not evidence of predictive utility when OOF discrimination is at chance]", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("CatBoost PredictionValuesChange Importance", fontsize=10)
    ax.set_ylabel("Predictor", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    f10_path = RESULTS_DIR / "figure_10_catboost_shap_values.png"
    fig.savefig(f10_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(f10_path)

    print(f"\nSuccessfully generated {len(generated)} diagnostic figures in {RESULTS_DIR}")
    return generated


if __name__ == "__main__":
    generate_all_figures()
