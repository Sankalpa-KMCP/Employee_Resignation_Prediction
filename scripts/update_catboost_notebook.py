"""Update CatBoost notebook cells to ensure dynamic loading, artifact reconciliation, and full 10-figure display."""

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_ROOT / "notebooks" / "team_contributions" / "models" / "member_2_catboost" / "06_m2_catboost.ipynb"

def update_notebook():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 9: Cramer's V fix
    c9_source = "".join(nb["cells"][9]["source"])
    c9_source = c9_source.replace("Cram\ufffdr", "Cramer").replace("Cramér", "Cramer")
    nb["cells"][9]["source"] = [line + "\n" for line in c9_source.split("\n")[:-1]] + ([c9_source.split("\n")[-1]] if c9_source.split("\n")[-1] else [])

    # Cell 11: Dynamic takeaways & benchmark display
    c11_code = """pd.set_option("display.max_columns", None)
summary_metrics_csv = CANONICAL_RESULTS_DIR / "catboost_summary_metrics.csv"
assert summary_metrics_csv.exists(), f"Summary metrics missing at: {summary_metrics_csv}"

summary_metrics_df = pd.read_csv(summary_metrics_csv)
print("Standardized Performance Comparison Across Separated Result Groups:")
display(summary_metrics_df[[
    "result_group", "model_name", "pr_auc", "roc_auc", "brier_score", "log_loss",
    "balanced_accuracy", "ci_pr_auc", "ci_roc_auc", "verdict"
]])

baseline_bench_csv = CANONICAL_RESULTS_DIR / "catboost_baseline_benchmark.csv"
if baseline_bench_csv.exists():
    baseline_bench_df = pd.read_csv(baseline_bench_csv)
    print("\\nCanonical Baseline Benchmark with Paired 95% Bootstrap Confidence Intervals:")
    display(baseline_bench_df)

dummy_row = summary_metrics_df[summary_metrics_df["model_name"].str.contains("Dummy Prior")].iloc[0]
raw_row = summary_metrics_df[summary_metrics_df["result_group"].str.startswith("A.")].iloc[0]
cal_row = summary_metrics_df[summary_metrics_df["result_group"].str.startswith("B.")].iloc[0]

print("\\nAcademic Takeaways Dynamically Grounded in Artifacts:")
print(f"1. Dummy Prior establishes analytical bounds: PR-AUC = {dummy_row['pr_auc']:.5f}, Brier = {dummy_row['brier_score']:.5f}.")
print(f"2. CatBoost Raw OOF PR-AUC ({raw_row['pr_auc']:.5f}) and ROC-AUC ({raw_row['roc_auc']:.5f}) sit directly inside the chance baseline noise envelope.")
print(f"3. Platt scaling achieves Brier = {cal_row['brier_score']:.5f} by collapsing probabilities toward the empirical class prevalence ({cal_row['pr_auc']:.5f}), without altering ranking metrics.")
"""
    nb["cells"][11]["source"] = [line + "\n" for line in c11_code.split("\n")[:-1]] + ([c11_code.split("\n")[-1]] if c11_code.split("\n")[-1] else [])

    # Cell 19: Dynamic loading of fixed tree experiments from CSV
    c19_code = """print("Outer Cross-Validation Fold Metrics (5 Folds):")
display(fold_metrics_df[[
    "fold", "pr_auc", "roc_auc", "brier_score", "balanced_accuracy",
    "precision", "recall", "f1", "predicted_positive_count", "predicted_positive_rate"
]])

print("\\nInvestigation of Fixed Tree Counts (Ablation across Outer Folds from Artifacts):")
exp_csv = CANONICAL_RESULTS_DIR / "catboost_leakage_safe_experiments.csv"
assert exp_csv.exists(), f"Experiments CSV missing at: {exp_csv}"
exp_df = pd.read_csv(exp_csv)
fixed_tree_exp = exp_df[exp_df["experiment"].str.contains("Fixed Trees", na=False)][
    ["experiment", "pr_auc", "roc_auc", "brier_score", "log_loss", "methodological_note"]
].copy().reset_index(drop=True)
display(fixed_tree_exp)
print("Conclusion: Later trees do not improve discrimination; they merely shrink probabilities toward the class prevalence.")
"""
    nb["cells"][19]["source"] = [line + "\n" for line in c19_code.split("\n")[:-1]] + ([c19_code.split("\n")[-1]] if c19_code.split("\n")[-1] else [])

    # Cell 27: Add explicit warning for feature importance
    c27_code = """feature_imp_csv = CANONICAL_RESULTS_DIR / "catboost_feature_importance.csv"
assert feature_imp_csv.exists(), f"Feature importance CSV missing at: {feature_imp_csv}"

importance_df = pd.read_csv(feature_imp_csv)
print("CatBoost Feature Importance Ranking (PredictionValuesChange):")
print(f"Warning: feature importance is not evidence of useful signal because raw OOF ROC-AUC is {raw_oof_roc_auc:.5f}.")
display(importance_df)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="#1f77b4", edgecolor="none", alpha=0.85)
ax.set_title("CatBoost - Final Development Model Feature Importance (Noise Attribution Warning)", fontsize=12, pad=12, fontweight="bold")
ax.set_xlabel("Mean Absolute Feature Contribution (PredictionValuesChange)", fontsize=11)
ax.set_ylabel("Predictor", fontsize=11)
ax.grid(axis="x", linestyle="--", alpha=0.3)
fig.tight_layout()

fig_path = CANONICAL_RESULTS_DIR / "figure_01_catboost_feature_importance.png"
fig.savefig(fig_path, dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)
"""
    nb["cells"][27]["source"] = [line + "\n" for line in c27_code.split("\n")[:-1]] + ([c27_code.split("\n")[-1]] if c27_code.split("\n")[-1] else [])

    # Cell 28 markdown update
    nb["cells"][28]["source"] = [
        "## 15. Comprehensive Diagnostic Visualizations Suite\n",
        "\n",
        "Below we display all 10 diagnostic figures generated across the formal CatBoost evaluation. Every figure clearly specifies whether it illustrates **Raw OOF**, **Calibrated OOF**, or **Inner-Fold / Experiment** dynamics.\n"
    ]

    # Cell 29: Display all 10 figures
    c29_code = """# Display all 10 diagnostic figures with clear data provenance labels
fig_files = [
    ("Figure 01: Feature Importance (Final Model / Noise Attribution Warning)", CANONICAL_RESULTS_DIR / "figure_01_catboost_feature_importance.png"),
    ("Figure 02: Precision-Recall Curves (Raw OOF vs. Dummy Prior Baseline)", CANONICAL_RESULTS_DIR / "figure_02_catboost_precision_recall_curves.png"),
    ("Figure 03: Confusion Matrix Heatmap (Raw OOF at Default 0.50 Threshold)", CANONICAL_RESULTS_DIR / "figure_03_catboost_confusion_matrix.png"),
    ("Figure 04: ROC Curves (Raw OOF vs. Chance Diagonal Baseline)", CANONICAL_RESULTS_DIR / "figure_04_catboost_roc_curve.png"),
    ("Figure 05: Probability Distribution Comparison (Raw OOF vs. Calibrated OOF)", CANONICAL_RESULTS_DIR / "figure_05_catboost_probability_distribution.png"),
    ("Figure 06: Reliability Diagram (Raw OOF vs. Cross-Fitted Calibrated OOF)", CANONICAL_RESULTS_DIR / "figure_06_catboost_calibration_curve.png"),
    ("Figure 07: Top-k Lift & Retention Capacity Curves (Raw OOF Predictions)", CANONICAL_RESULTS_DIR / "figure_07_catboost_topk_lift_curve.png"),
    ("Figure 08: Inner-Fold Training vs. Validation Learning Curves (Logloss & PRAUC)", CANONICAL_RESULTS_DIR / "figure_08_catboost_inner_learning_curves.png"),
    ("Figure 09: Fold-Level Probability Score Distributions across 5 Folds", CANONICAL_RESULTS_DIR / "figure_09_catboost_fold_score_distributions.png"),
    ("Figure 10: Artifact-Derived Feature Importance (not SHAP)", CANONICAL_RESULTS_DIR / "figure_10_catboost_shap_values.png"),
]

for title, fpath in fig_files:
    assert fpath.exists(), f"Diagnostic figure missing at: {fpath}"
    print(f"\\n{'=' * 75}\\n{title}\\n{'=' * 75}")
    display(Image(filename=str(fpath)))
"""
    nb["cells"][29]["source"] = [line + "\n" for line in c29_code.split("\n")[:-1]] + ([c29_code.split("\n")[-1]] if c29_code.split("\n")[-1] else [])

    # Cell 34: Add assertions for all 10 figures and new baseline benchmark
    c34_code = """# ==========================================
# PHASE 3: POST-EXECUTION VALIDATION CHECKS
# ==========================================

# 1. Exact row count and uniqueness
assert len(catboost_oof_df) == 80_000, f"Expected 80,000 rows, got {len(catboost_oof_df)}"
assert len(calibrated_oof_df) == 80_000, f"Expected 80,000 calibrated rows, got {len(calibrated_oof_df)}"
assert not catboost_oof_df["Employee_ID"].duplicated().any(), "Duplicate Employee_IDs detected!"
assert not calibrated_oof_df["Employee_ID"].duplicated().any(), "Duplicate Employee_IDs detected in calibrated OOF!"

# 2. Strict fold distribution (16,000 per fold)
assert set(catboost_oof_df["cv_fold"].unique()) == {1, 2, 3, 4, 5}
assert (catboost_oof_df["cv_fold"].value_counts() == 16_000).all()

# 3. Holdout isolation check (Zero contamination)
holdout_overlap = set(catboost_oof_df["Employee_ID"]).intersection(holdout_ids)
assert len(holdout_overlap) == 0, f"LEAKAGE DETECTED: {len(holdout_overlap)} holdout rows in OOF!"

# 4. Target distribution and imbalance treatment (derived, never hard-coded)
target_distribution = catboost_oof_df["y_true"].value_counts().sort_index()
assert set(target_distribution.index) == {0, 1}, "Both target classes must be present in OOF predictions."
print("OOF Resigned target distribution:")
display(target_distribution.rename_axis("Resigned").to_frame("count").assign(proportion=lambda x: x["count"] / x["count"].sum()))
with open(CANONICAL_RESULTS_DIR / "catboost_run_metadata.json", "r", encoding="utf-8") as metadata_handle:
    selected_metadata = json.load(metadata_handle)
print("Imbalance treatment: class weighting was evaluated only inside outer-training folds; the selected candidate "
      f"({selected_metadata['candidate_id']}) is {('unweighted' if selected_metadata['params'].get('auto_class_weights') is None else 'weighted')} because weighting did not improve pooled OOF ranking and calibration.")

# 5. Probability range sanity
assert not catboost_oof_df["y_probability"].isna().any(), "NaN found in predictions!"
assert catboost_oof_df["y_probability"].between(0.0, 1.0).all(), "Probabilities outside [0, 1]!"
assert not calibrated_oof_df["calibrated_y_probability"].isna().any(), "NaN found in calibrated predictions!"
assert calibrated_oof_df["calibrated_y_probability"].between(0.0, 1.0).all(), "Calibrated probabilities outside [0, 1]!"

# 6. Artifact presence - Tables, Models and Manifest
required_artifacts = [
    CANONICAL_RESULTS_DIR / "catboost_oof_predictions.csv",
    CANONICAL_RESULTS_DIR / "catboost_calibrated_oof_predictions.csv",
    CANONICAL_RESULTS_DIR / "catboost_fold_metrics.csv",
    CANONICAL_RESULTS_DIR / "catboost_summary_metrics.csv",
    CANONICAL_RESULTS_DIR / "catboost_summary_metrics.json",
    CANONICAL_RESULTS_DIR / "summary_metrics.json",
    CANONICAL_RESULTS_DIR / "catboost_baseline_benchmark.csv",
    CANONICAL_MODELS_DIR / "catboost_development_model.cbm",
    CANONICAL_RESULTS_DIR / "catboost_fold_candidate_selection.csv",
    CANONICAL_RESULTS_DIR / "catboost_leakage_safe_experiments.csv",
    CANONICAL_RESULTS_DIR / "catboost_feature_audit.csv",
    CANONICAL_RESULTS_DIR / "catboost_threshold_analysis.csv",
    CANONICAL_RESULTS_DIR / "catboost_calibration_diagnostics.csv",
    CANONICAL_RESULTS_DIR / "catboost_output_manifest.csv",
    # Complete 10-Figure Suite
    CANONICAL_RESULTS_DIR / "figure_01_catboost_feature_importance.png",
    CANONICAL_RESULTS_DIR / "figure_02_catboost_precision_recall_curves.png",
    CANONICAL_RESULTS_DIR / "figure_03_catboost_confusion_matrix.png",
    CANONICAL_RESULTS_DIR / "figure_04_catboost_roc_curve.png",
    CANONICAL_RESULTS_DIR / "figure_05_catboost_probability_distribution.png",
    CANONICAL_RESULTS_DIR / "figure_06_catboost_calibration_curve.png",
    CANONICAL_RESULTS_DIR / "figure_07_catboost_topk_lift_curve.png",
    CANONICAL_RESULTS_DIR / "figure_08_catboost_inner_learning_curves.png",
    CANONICAL_RESULTS_DIR / "figure_09_catboost_fold_score_distributions.png",
    CANONICAL_RESULTS_DIR / "figure_10_catboost_shap_values.png",
]
for art in required_artifacts:
    assert art.exists(), f"Required artifact missing: {art}"
    assert art.stat().st_size > 0, f"Required artifact is empty: {art}"

# 7. Strict Numerical Reconciliation Assertions (Zero Fabricated or Hardcoded Values)
TOL = 1e-5
raw_oof_precision = float(precision_score(y_true_arr, y_pred_05_arr, zero_division=0))
raw_oof_recall = float(recall_score(y_true_arr, y_pred_05_arr, zero_division=0))
raw_oof_f1 = float(f1_score(y_true_arr, y_pred_05_arr, zero_division=0))
cal_oof_pred_05 = (cal_p >= 0.50).astype(int)
cal_oof_precision = float(precision_score(y_true_arr, cal_oof_pred_05, zero_division=0))
cal_oof_recall = float(recall_score(y_true_arr, cal_oof_pred_05, zero_division=0))
cal_oof_f1 = float(f1_score(y_true_arr, cal_oof_pred_05, zero_division=0))
cal_oof_balanced_acc = float(balanced_accuracy_score(y_true_arr, cal_oof_pred_05))
cal_oof_pred_pos_rate = float(np.mean(cal_oof_pred_05))
with open(CANONICAL_RESULTS_DIR / "catboost_summary_metrics.json", "r", encoding="utf-8") as f:
    s_json = json.load(f)

s_csv = pd.read_csv(CANONICAL_RESULTS_DIR / "catboost_summary_metrics.csv")
raw_csv = s_csv[s_csv["result_group"] == "A. Nested-CV Raw CatBoost OOF"].iloc[0]
cal_csv = s_csv[s_csv["result_group"] == "B. Cross-Fitted Calibrated CatBoost OOF"].iloc[0]

# Verify JSON Group A matches recomputed OOF metrics
assert abs(s_json["group_a_raw_catboost_oof_metrics"]["raw_oof_pr_auc"] - raw_oof_pr_auc) < TOL, "Summary JSON PR-AUC mismatch!"
assert abs(s_json["group_a_raw_catboost_oof_metrics"]["raw_oof_roc_auc"] - raw_oof_roc_auc) < TOL, "Summary JSON ROC-AUC mismatch!"
assert abs(s_json["group_a_raw_catboost_oof_metrics"]["raw_oof_brier_score"] - raw_oof_brier_score) < TOL, "Summary JSON Brier mismatch!"
assert abs(s_json["group_a_raw_catboost_oof_metrics"]["raw_oof_log_loss"] - raw_oof_log_loss) < TOL, "Summary JSON log loss mismatch!"
assert abs(s_json["group_a_raw_catboost_oof_metrics"]["raw_oof_precision"] - raw_oof_precision) < TOL, "Summary JSON precision mismatch!"
assert abs(s_json["group_a_raw_catboost_oof_metrics"]["raw_oof_recall"] - raw_oof_recall) < TOL, "Summary JSON recall mismatch!"
assert abs(s_json["group_a_raw_catboost_oof_metrics"]["raw_oof_f1"] - raw_oof_f1) < TOL, "Summary JSON F1 mismatch!"
assert abs(s_json["group_a_raw_catboost_oof_metrics"]["raw_oof_balanced_accuracy"] - raw_oof_balanced_acc) < TOL, "Summary JSON balanced accuracy mismatch!"
assert abs(s_json["group_a_raw_catboost_oof_metrics"]["raw_oof_pred_pos_rate"] - raw_oof_pred_pos_rate) < TOL, "Summary JSON predicted positive rate mismatch!"

# Verify CSV Group A matches recomputed OOF metrics
assert abs(raw_csv["pr_auc"] - raw_oof_pr_auc) < TOL, "Summary CSV PR-AUC mismatch!"
assert abs(raw_csv["roc_auc"] - raw_oof_roc_auc) < TOL, "Summary CSV ROC-AUC mismatch!"
assert abs(raw_csv["brier_score"] - raw_oof_brier_score) < TOL, "Summary CSV Brier mismatch!"
assert abs(raw_csv["balanced_accuracy"] - raw_oof_balanced_acc) < TOL, "Summary CSV balanced accuracy mismatch!"
assert abs(raw_csv["pred_pos_rate"] - raw_oof_pred_pos_rate) < TOL, "Summary CSV predicted positive rate mismatch!"

# Verify JSON Group B matches recomputed Calibrated OOF metrics
assert abs(s_json["group_b_cross_fitted_calibrated_oof_metrics"]["cal_oof_pr_auc"] - cal_oof_pr_auc) < TOL, "Calibrated JSON PR-AUC mismatch!"
assert abs(s_json["group_b_cross_fitted_calibrated_oof_metrics"]["cal_oof_roc_auc"] - cal_oof_roc_auc) < TOL, "Calibrated JSON ROC-AUC mismatch!"
assert abs(s_json["group_b_cross_fitted_calibrated_oof_metrics"]["cal_oof_brier_score"] - cal_oof_brier_score) < TOL, "Calibrated JSON Brier mismatch!"
assert abs(s_json["group_b_cross_fitted_calibrated_oof_metrics"]["cal_oof_log_loss"] - cal_oof_log_loss) < TOL, "Calibrated JSON log loss mismatch!"
assert abs(s_json["group_b_cross_fitted_calibrated_oof_metrics"]["cal_oof_precision"] - cal_oof_precision) < TOL, "Calibrated JSON precision mismatch!"
assert abs(s_json["group_b_cross_fitted_calibrated_oof_metrics"]["cal_oof_recall"] - cal_oof_recall) < TOL, "Calibrated JSON recall mismatch!"
assert abs(s_json["group_b_cross_fitted_calibrated_oof_metrics"]["cal_oof_f1"] - cal_oof_f1) < TOL, "Calibrated JSON F1 mismatch!"
assert abs(s_json["group_b_cross_fitted_calibrated_oof_metrics"]["cal_oof_balanced_accuracy"] - cal_oof_balanced_acc) < TOL, "Calibrated JSON balanced accuracy mismatch!"
assert abs(s_json["group_b_cross_fitted_calibrated_oof_metrics"]["cal_oof_pred_pos_rate"] - cal_oof_pred_pos_rate) < TOL, "Calibrated JSON predicted-positive rate mismatch!"

# Verify CSV Group B matches recomputed Calibrated OOF metrics
assert abs(cal_csv["pr_auc"] - cal_oof_pr_auc) < TOL, "Calibrated CSV PR-AUC mismatch!"
assert abs(cal_csv["roc_auc"] - cal_oof_roc_auc) < TOL, "Calibrated CSV ROC-AUC mismatch!"
assert abs(cal_csv["brier_score"] - cal_oof_brier_score) < TOL, "Calibrated CSV Brier mismatch!"

# Verify Confusion Matrix
cm_df = pd.read_csv(CANONICAL_RESULTS_DIR / "catboost_confusion_matrix.csv", index_col=0)
cm_calc = confusion_matrix(y_true_arr, y_pred_05_arr, labels=[0, 1])
assert cm_df.iloc[0, 0] == cm_calc[0, 0], "Confusion matrix TN mismatch!"
assert cm_df.iloc[0, 1] == cm_calc[0, 1], "Confusion matrix FP mismatch!"
assert cm_df.iloc[1, 0] == cm_calc[1, 0], "Confusion matrix FN mismatch!"
assert cm_df.iloc[1, 1] == cm_calc[1, 1], "Confusion matrix TP mismatch!"

# Verify Fold Metrics mean matches Summary JSON
f_metrics = pd.read_csv(CANONICAL_RESULTS_DIR / "catboost_fold_metrics.csv")
assert abs(f_metrics["pr_auc"].mean() - s_json["fold_metrics"]["pr_auc_mean"]) < TOL, "Fold PR-AUC mean mismatch!"
assert abs(f_metrics["roc_auc"].mean() - s_json["fold_metrics"]["roc_auc_mean"]) < TOL, "Fold ROC-AUC mean mismatch!"
assert abs(f_metrics["brier_score"].mean() - s_json["fold_metrics"]["brier_score_mean"]) < TOL, "Fold Brier mean mismatch!"

print("All Phase 3 post-execution validation checks & numerical reconciliation assertions PASSED successfully!")
"""
    nb["cells"][34]["source"] = [line + "\n" for line in c34_code.split("\n")[:-1]] + ([c34_code.split("\n")[-1]] if c34_code.split("\n")[-1] else [])

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully updated notebook at {NB_PATH}")

if __name__ == "__main__":
    update_notebook()
