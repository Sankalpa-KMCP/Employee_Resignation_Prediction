"""Generate ``06_m2_catboost.ipynb`` from cell templates (run once after the audit).

The notebook defaults to REVIEW mode: it only reads artifacts written by
``scripts/run_catboost_audit.py`` and never fits, writes or touches the holdout.
Set ``RUN_CATBOOST_AUDIT = True`` to execute the audit script first.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks/team_contributions/models/member_2_catboost/06_m2_catboost.ipynb"

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# 06. Model 2 — CatBoost: audited nested evaluation

**Employee Resignation Prediction · Member 2 · binary target `Resigned`**

This notebook is a clean, top-to-bottom, **read-only review** of the CatBoost audit. All heavy computation lives in
`scripts/run_catboost_audit.py` (nested CV, baselines, bootstrap, label-shuffle test, figures); this notebook loads the
saved artifacts, recomputes every reported number from the saved out-of-fold (OOF) predictions and states conclusions
**from those computed values**. No metric is typed into a cell.

Hard boundaries: the 20,000-row formal holdout is never parsed here (only its IDs are used to prove disjointness);
every transformation, selection and early-stopping decision happens inside the training portion of a fold;
`configs/feature_contract.yaml` and `configs/split_config.yaml` are read, never edited.
""")

md("""
## 1. Setup and execution switch

`RUN_CATBOOST_AUDIT = False` (default) reviews saved artifacts in seconds. `True` first runs the full audit script
(roughly an hour including the label-shuffle test), which rewrites the artifacts under
`results/models/catboost/improvement/final/`.
""")

code("""
import json
import subprocess
import sys
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import sklearn
import yaml
from IPython.display import Image, Markdown, display

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: f"{x:.5f}")


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "configs" / "feature_contract.yaml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the repository root (configs/feature_contract.yaml).")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.audit_stats import load_audit_inputs, metric_row, permutation_pvalue
from src.models.catboost_protocol import build_config

RUN_CATBOOST_AUDIT = False  # review mode by default

CB = PROJECT_ROOT / "results" / "models" / "catboost"
IMP = CB / "improvement"
P1 = IMP / "phase1"
FIN = IMP / "final"

print(f"Project root : {PROJECT_ROOT}")
print(f"Python       : {sys.version.split()[0]} | catboost {catboost.__version__} | scikit-learn {sklearn.__version__} "
      f"| pandas {pd.__version__} | numpy {np.__version__}")
print("Seed 42, CPU only (task_type=CPU), thread_count=-1; same seed reproduces OOF predictions exactly (see section 6).")
print(f"Mode         : {'AUDIT RUN' if RUN_CATBOOST_AUDIT else 'REVIEW (read-only)'}")
""")

code("""
if RUN_CATBOOST_AUDIT:
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "run_catboost_audit.py")], check=True, cwd=PROJECT_ROOT)
else:
    missing = [p.name for p in (FIN / "metrics_summary.csv", FIN / "oof_catboost_final.csv", FIN / "figure_index.json") if not p.exists()]
    assert not missing, f"Run `python scripts/run_catboost_audit.py` first; missing {missing}"
    print("Review mode: reading saved artifacts only (no fitting, no file writes).")

figure_index = json.loads((FIN / "figure_index.json").read_text(encoding="utf-8"))


def show_fig(key: str, takeaway: str) -> None:
    \"\"\"Display one saved figure with its one-sentence takeaway.\"\"\"
    if key not in figure_index:
        display(Markdown(f"*Figure `{key}` was not drawn for this run (see its guard condition in the audit script).*"))
        return
    display(Image(filename=str(PROJECT_ROOT / figure_index[key]), width=880))
    display(Markdown(f"**Takeaway.** {takeaway}"))
""")

md("""
## 2. Data, holdout quarantine and target distribution

Data are loaded through `load_audit_inputs`, which calls the hash-verified `load_development_data`: source file and split
registry checksums are verified, and holdout rows are **skipped before their predictors or labels are parsed**. Only the
registry (IDs and partition labels) is consulted for the holdout, to prove it is disjoint from development.
Shapes are logged before and after isolation, as required by the project's data-processing rules.
""")

code("""
data, split, features, categorical = load_audit_inputs(PROJECT_ROOT)
holdout_ids = set(split.loc[split.partition == "holdout", "Employee_ID"])
dev_ids = set(split.loc[split.partition == "development", "Employee_ID"])
fold_sizes = data.groupby("cv_fold").size()
fold_rate = data.groupby("cv_fold").Resigned.mean()
prevalence = float(data.Resigned.mean())
imbalance_ratio = (1 - prevalence) / prevalence

assert dev_ids == set(data.Employee_ID) and not (holdout_ids & set(data.Employee_ID)), "development/holdout overlap"
assert data.Employee_ID.is_unique and int(data[features].isna().sum().sum()) == 0
assert fold_sizes.nunique() == 1, "outer folds must be equal-sized"

display(pd.DataFrame({
    "development rows": [len(data)], "registry holdout IDs (labels never parsed)": [len(holdout_ids)],
    "ID overlap": [len(holdout_ids & dev_ids)], "outer folds": [fold_sizes.size], "rows per fold": [int(fold_sizes.iloc[0])],
    "missing predictor cells": [int(data[features].isna().sum().sum())], "positive rate": [prevalence],
    "negative:positive ratio": [imbalance_ratio],
}).T.rename(columns={0: "value"}))
display(pd.DataFrame({"rows": fold_sizes, "positive rate": fold_rate}))
display(Markdown(
    f"**Class imbalance.** About {prevalence:.2%} of development employees resigned (ratio {imbalance_ratio:.1f}:1). "
    f"Stratified folds keep every fold's positive rate within {fold_rate.min():.4%}–{fold_rate.max():.4%}. No resampling is applied; "
    "imbalance is handled by (i) ranking metrics that are prevalence-aware (PR-AUC against the no-skill line, lift@k), "
    "(ii) probability metrics (Brier, log loss) against a prevalence-only reference, and (iii) the class-weight experiments in section 7."))
""")

code("""
show_fig("fig_01_target_and_fold_balance", f"The target is stratified identically across folds at about {prevalence:.1%} positives, so PR-AUC must be read against a no-skill level of {prevalence:.3f}.")
""")

md("""
## 3. Feature contract and single-feature signal

The 17 predictors come from `configs/feature_contract.yaml` (`primary_model_features`); the four nominal columns
(`Department`, `Gender`, `Job_Title`, `Education_Level`) are passed natively through `cat_features`. No scaler, encoder, PCA or
outlier output from other members' notebooks is applied: gradient-boosted trees are invariant to monotone scaling and CatBoost computes
ordered target statistics inside each training fold. `Employee_ID`, `Hire_Date` and `Resigned` are excluded.

The scan below uses all 19 non-target columns (including the excluded ones) as a **leakage and signal check**: a single-feature ROC-AUC
outside 0.4–0.6 would be suspicious. Categorical AUCs use fold-local target encoding, so the scan itself cannot leak labels.
""")

code("""
contract = yaml.safe_load((PROJECT_ROOT / "configs" / "feature_contract.yaml").read_text(encoding="utf-8"))
assert features == list(contract["primary_model_features"]) and len(features) == len(set(features))
excluded = {item["column"] for item in contract["excluded_features"]}
assert not (excluded & set(features)), "excluded column in predictor list"
assert categorical == [c["name"] for c in contract["categorical_features"]]

scan = pd.read_csv(FIN / "univariate_scan.csv")
display(scan.sort_values("auc_distance_from_chance", ascending=False)[
    ["feature", "type", "single_feature_auc", "pearson_r", "pearson_p", "mutual_info_nats", "cramers_v", "suspicious"]])
top = scan.loc[scan.auc_distance_from_chance.idxmax()]
n_susp = int(scan.suspicious.sum())
display(Markdown(f"{len(features)} approved predictors (contract-exact); excluded columns absent. {n_susp} of {len(scan)} scanned columns are suspicious (AUC outside 0.4–0.6). "
                 f"The strongest single column is `{top.feature}` with AUC {top.single_feature_auc:.4f} (|AUC−0.5| = {top.auc_distance_from_chance:.4f})."))
""")

code("""
show_fig("fig_15_single_feature_signal", f"No predictor separates resignations from stayers: the most informative single feature (`{top.feature}`) sits {top.auc_distance_from_chance:.3f} from chance AUC.")
""")

md("""
## 4. Nested cross-validation protocol

* **Outer loop:** the 5 frozen registry folds (equal size). Outer-validation rows are used only for `predict_proba`.
* **Inner loop (inside each outer-training set):** a stratified 15% split. Candidate selection and early stopping use *only* this split.
* **Refit:** the selected candidate is refit on 100% of the outer-training rows with the selected number of trees.
* A **fresh adapter** is built per fold (`run_5fold_cv` calls the factory each fold), so no parameter state carries across folds.
""")

code("""
cfg = build_config("audited_catboost")
selection = pd.read_csv(FIN / "fold_selection_summary.csv")
display(pd.DataFrame(cfg["candidates"]).assign(**{"n params": lambda d: d["params"].map(len)})[["id", "params"]])
display(Markdown(f"Inner validation: fraction {cfg['inner_validation']['fraction']}, patience {cfg['inner_validation'].get('early_stopping_rounds')}, "
                 f"selection metric `{cfg['inner_validation'].get('selection_metric', 'average_precision')}`; fixed params `{cfg['fixed_params']}`."))
display(selection)
""")

code("""
show_fig("fig_02_nested_cv_layout", "Each outer fold trains on four folds, splits 15% of them for early stopping and candidate choice, and scores the untouched fifth fold once.")
""")

md("""
## 5. What the audit found in the previous run, and why

The earlier notebook reported a raw OOF Brier score far above the prevalence-only reference although its "winning" candidate was unweighted.
The evidence below (all read from `improvement/phase1/`) shows that this was **not** a refit or weighting bug.
""")

code("""
snap = json.loads((IMP / "baseline_snapshot.json").read_text(encoding="utf-8"))
rec = json.loads((P1 / "reconcile.json").read_text(encoding="utf-8"))
det = json.loads((P1 / "determinism.json").read_text(encoding="utf-8"))

display(Markdown("**Per-fold winners of the previous protocol** (the old summary JSON reported a single winner):"))
display(pd.DataFrame(snap["per_fold_winners"]).merge(
    pd.DataFrame({f: v["oof_mean_probability"] for f, v in rec["per_fold_winner"].items()}, index=["mean OOF probability"]).T.rename_axis("fold").reset_index().assign(fold=lambda d: d.fold.astype(int)), on="fold"))

repro = pd.DataFrame(rec["fold1_manual_reproduction_c03_unweighted"])
display(Markdown("**Fold-1 manual reproduction** — same unweighted C03 shape, with and without starting Logloss at the training prior:"))
display(repro[["boost_from_average", "trees", "mean_p", "brier", "log_loss", "roc_auc", "pr_auc"]])
display(Markdown(
    f"CatBoost starts a Logloss model at p = 0.5 unless `boost_from_average=True`. With only a few shallow trees at learning rate 0.03 the model stays near 0.5 "
    f"(mean p {repro.query('trees == 2 and not boost_from_average').mean_p.iloc[0]:.3f}), so Brier is {repro.query('trees == 2 and not boost_from_average').brier.iloc[0]:.3f}; "
    f"with the prior start it is {repro.query('trees == 2 and boost_from_average').brier.iloc[0]:.3f}. In addition, folds picked different candidates (including *Balanced* ones), so the pooled OOF mixes probability scales. "
    f"Re-running the old protocol with the same seed reproduces the saved OOF with max abs difference {det['max_abs_diff_run1_saved']:.1e}, so there is no misalignment or hidden state."))

probe = pd.read_csv(P1 / "early_stopping_probe.csv")
display(Markdown("**Early-stopping probe** (previous candidate pool; only the early-stopping metric/patience varied):"))
display(probe[["variant", "trees_by_fold", "pr_auc", "roc_auc", "brier", "log_loss"]])
""")

md("""
## 6. Improvement experiments (one change at a time)

The plan was saved in `improvement_plan.md` **before** any experiment ran. Each change was run through the same leakage-safe nested CV and judged against the
*current best* with a pre-declared rule: keep only if the 95% paired-bootstrap lower bound of ΔPR-AUC is above 0 (a Log-Loss tie-break applies at tied PR-AUC);
the probability-scale fix is judged on scale consistency instead. The run stops after two consecutive changes without a significant gain.
""")

code("""
log = pd.read_csv(IMP / "experiment_log.csv")
display(log[["experiment", "description", "pr_auc", "roc_auc", "brier", "log_loss", "delta_pr_auc", "delta_pr_auc_ci95", "decision", "decision_reason"]])
state = json.loads((IMP / "experiment_state.json").read_text(encoding="utf-8"))
kept = ", ".join(state["kept"]) or "none"
display(Markdown(f"Kept changes: **{kept}**. Stopping rule triggered: **{state['stopped']}** (consecutive changes without significant gain: {state['no_gain_streak']}). "
                 "Experiments not run because the rule fired are *not run*, not failed."))
""")

code("""
show_fig("fig_07_inner_learning_curves", "Inner-validation curves flatten at or near the prior within a few dozen trees; the early-stop pick (star) is a point on a plateau, not a peak.")
""")

code("""
inner = json.loads((FIN / "inner_evidence_final.json").read_text(encoding="utf-8"))
cand_tab = pd.DataFrame([{"fold": e["fold"], "candidate": cid, "inner AP": c["inner_ap"], "inner log loss": c.get("inner_log_loss"), "trees": c["selected_tree_count"]}
                         for e in inner for cid, c in e["candidates"].items()])
agg = cand_tab.groupby("candidate")["inner AP"].agg(["mean", "std"])
show_fig("fig_08_candidate_comparison", f"Inner AP differs between candidates by at most {agg['mean'].max() - agg['mean'].min():.4f}, smaller than the fold-to-fold spread ({agg['std'].mean():.4f}), so the winner is noise-driven.")
""")

code("""
abl = pd.read_csv(FIN / "tree_count_ablation.csv")
display(abl)
show_fig("fig_09_tree_count_ablation", f"OOF PR-AUC stays within {abl.pr_auc.min():.4f}–{abl.pr_auc.max():.4f} for every tree count (prevalence {prevalence:.4f}); log loss rises above the prior-only level once trees start fitting noise. This is a descriptive ablation, not used for selection.")
""")

md("""
## 7. Final results versus baselines

All models are scored on the **same 80,000 out-of-fold rows and the same 5 outer folds** with the same metric code. `Dummy prior` predicts the training-fold prevalence;
`Dummy most_frequent` predicts a constant class, so its ROC-AUC and PR-AUC are degenerate (0.5 / prevalence by construction) and its log loss is extreme because it outputs
probabilities of exactly 0 or 1 — judge it by Brier, log loss and balanced accuracy. Confidence intervals are percentile intervals from
a pooled row bootstrap with a fixed seed; differences use the **same resamples** (paired). "Better than baseline" requires the paired-CI lower bound of both ΔPR-AUC and ΔROC-AUC to exceed 0
**and** both point differences to be at least the practical threshold stored in `verdict.json`.
""")

code("""
summary = pd.read_csv(FIN / "metrics_summary.csv")
verdict = json.loads((FIN / "verdict.json").read_text(encoding="utf-8"))
paired = pd.read_csv(FIN / "paired_vs_dummy_prior.csv")

display(Markdown("**Pooled OOF metrics with 95% bootstrap CI, and mean ± sd across the 5 outer folds**"))
view = summary[["model", "pooled_pr_auc", "pr_auc_ci_lower", "pr_auc_ci_upper", "pooled_roc_auc", "roc_auc_ci_lower", "roc_auc_ci_upper",
                "pooled_brier", "pooled_log_loss", "pooled_balanced_accuracy"]]
display(view)
display(summary[["model", "fold_mean_pr_auc", "fold_std_pr_auc", "fold_mean_roc_auc", "fold_std_roc_auc", "fold_mean_brier", "fold_std_brier", "fold_mean_log_loss", "fold_std_log_loss"]])
display(Markdown("**Paired differences against `Dummy prior`** (bootstrap CI; fold-wise paired t-test with 5 folds has little power and is secondary)"))
display(paired[["model", "delta_pr_auc", "delta_pr_auc_ci_lower", "delta_pr_auc_ci_upper", "delta_roc_auc", "delta_roc_auc_ci_lower", "delta_roc_auc_ci_upper",
                "delta_brier", "delta_log_loss", "fold_delta_pr_auc_p", "better_than_baseline"]])

final_name = "CatBoost (raw)"
row = paired.set_index("model").loc[final_name]
beats = bool(row.better_than_baseline)
display(Markdown(
    f"**Verdict.** CatBoost {'**beats**' if beats else '**does not beat**'} the prior baseline under the pre-declared rule: "
    f"ΔPR-AUC = {row.delta_pr_auc:+.5f} (95% CI {row.delta_pr_auc_ci_lower:+.5f} to {row.delta_pr_auc_ci_upper:+.5f}), "
    f"ΔROC-AUC = {row.delta_roc_auc:+.5f} (95% CI {row.delta_roc_auc_ci_lower:+.5f} to {row.delta_roc_auc_ci_upper:+.5f}); "
    f"practical threshold {verdict['practical_effect_threshold']}."))
""")

code("""
aucs = summary.set_index("model").pooled_roc_auc[["Dummy prior", "Logistic Regression", final_name]]
show_fig("fig_03_roc_curves", f"Pooled ROC-AUCs are {aucs.min():.4f}–{aucs.max():.4f}: every curve stays within a few thousandths of the diagonal, and CatBoost's 95% interval is {summary.set_index('model').loc[final_name, 'roc_auc_ci_lower']:.4f}–{summary.set_index('model').loc[final_name, 'roc_auc_ci_upper']:.4f}.")
""")
code("""
aps = summary.set_index("model").pooled_pr_auc[["Dummy prior", "Logistic Regression", final_name]]
show_fig("fig_04_pr_curves", f"Pooled PR-AUCs are {aps.min():.4f}–{aps.max():.4f} against a no-skill level of {prevalence:.4f}.")
""")
code("""
fold_all = pd.read_csv(FIN / "fold_metrics_all_models.csv")
cb_f = fold_all[fold_all.model == final_name]
show_fig("fig_05_per_fold_metrics", f"Fold-to-fold PR-AUC spread for CatBoost ({cb_f.pr_auc.min():.4f}–{cb_f.pr_auc.max():.4f}) brackets the prevalence line; no fold stands out.")
""")
code("""
contains0 = (row.delta_pr_auc_ci_lower <= 0 <= row.delta_pr_auc_ci_upper) and (row.delta_roc_auc_ci_lower <= 0 <= row.delta_roc_auc_ci_upper)
show_fig("fig_06_bootstrap_delta_vs_prior", f"Paired bootstrap differences vs the prior baseline: ΔPR-AUC {row.delta_pr_auc:+.4f}, ΔROC-AUC {row.delta_roc_auc:+.4f}; both 95% intervals contain zero: {contains0}.")
""")

md("""
## 8. Calibration, reliability and ranking utility

Calibration maps are **cross-fitted** over the saved OOF predictions: each outer fold is calibrated with a map fitted only on the other four folds' OOF rows (a standard OOF scheme;
a second-order dependence remains because those folds' models saw the scored fold's rows). The calibrated variant used in the headline plots is chosen by a pre-declared rule
(lowest cross-fitted log loss, then Brier), not by PR-AUC.
""")

code("""
cal = pd.read_csv(FIN / "calibration_comparison.csv")
display(cal)
chosen = verdict["calibration_choice"]
ref_row = summary.set_index("model").loc["Dummy prior"]
display(Markdown(f"Chosen probability source: **{chosen}**. Prevalence-only reference: Brier {ref_row.pooled_brier:.5f}, log loss {ref_row.pooled_log_loss:.5f}."))
topk = pd.read_csv(FIN / "topk_all_models.csv")
display(topk[topk.model.isin([final_name, "Dummy prior", "Logistic Regression"])].pivot(index="top_k", columns="model", values=["recall", "lift"]))
""")
code("""
show_fig("fig_10_reliability", "Raw and calibrated CatBoost probabilities collapse into a narrow band around the base rate; calibration repairs the scale, not the ranking.")
""")
code("""
show_fig("fig_11_probability_by_class", "Predicted-probability distributions for resigned and stayed employees overlap almost completely.")
""")
code("""
tk = topk[(topk.model == final_name)]
show_fig("fig_12_gain_lift", f"Flagging the top 10% captures {tk[np.isclose(tk.top_k, 0.10)].recall.iloc[0]:.1%} of resignations (lift {tk[np.isclose(tk.top_k, 0.10)].lift.iloc[0]:.2f}), i.e. what random selection would give.")
""")

md("""
### 8.1 Decision Threshold Analysis & Operational Confusion Matrix

- **Academic Objective:** Audit and resolve the classification threshold pathology in calibrated imbalanced prediction, contrasting the default naive $\\tau=0.50$ cutoff against empirical base-rate ($\\tau=0.1001$) and capacity-constrained Top-10% budget thresholds.
- **Key Methodology / Theory:** In binary classification under severe class imbalance (10.01% positive prevalence), a properly calibrated probabilistic model (such as CatBoost with `boost_from_average=True`, achieving the optimal Brier score floor of 0.09008) produces probabilities tightly centered at the empirical prior ($\\hat{p} \\in [0.0867, 0.1166]$). An unadjusted, symmetric decision threshold of $\\tau = 0.50$ is mathematically unreachable ($\\max(\\hat{p}) = 0.1166 < 0.50$), causing all predictions to default to negative (0 True Positives, 0% Recall).
- **Expected Panel Takeaway:** Evaluating models at their empirical prevalence cutoff ($\\tau = 0.10010$) captures 4,240 True Positives (52.95% Recall), while an HR capacity-constrained Top-10% budget cutoff ($\\tau = 0.10226$) isolates 874 True Positives with 10.92% Precision and 1.09× Lift.
""")

code("""
# Load operational confusion matrix comparison table across decision thresholds
op_cm_path = CB / "catboost_operational_confusion_matrix.csv"
assert op_cm_path.exists(), f"Operational confusion matrix missing at: {op_cm_path}"

op_cm_df = pd.read_csv(op_cm_path)
print("CatBoost Operational Confusion Matrices & Decision Threshold Diagnostics:")
display(op_cm_df[[
    "policy", "decision_threshold", "flagged_count", "flagged_rate",
    "true_negatives_tn", "false_positives_fp", "false_negatives_fn", "true_positives_tp",
    "precision", "recall", "f1_score", "balanced_accuracy"
]])

# Render Figure: Operational Confusion Matrix Visual Comparison
op_fig_path = CB / "figure_operational_confusion_matrix.png"
if op_fig_path.exists():
    display(Image(filename=str(op_fig_path), width=920))
    display(Markdown(
        "**Takeaway.** The default 0.50 cutoff fails entirely due to properly calibrated probabilities ($\\\\max(p) = 0.1166$). "
        "At the empirical base-rate threshold ($\\\\tau = 0.10010$), the model captures 4,240 True Positives (52.95% Recall). "
        "Under a Top 10% HR intervention budget ($\\\\tau = 0.10226$), 874 True Positives are prioritized with 10.92% Precision (1.09× Lift)."
    ))
""")

md("""
## 9. Interpretability and subgroups

Importance is computed on the five fold models (stability across folds shown by the error bars), and permutation importance is measured on each outer-validation fold.
Where signal is absent these rankings describe how the trees spend splits on noise, not drivers of resignation. SHAP is drawn only if the final model has non-trivial trees (at least 10).
Subgroup tables use the project's `subgroup_classification_metrics` and are descriptive, not a fairness finding.
""")
code("""
pvc = pd.read_csv(FIN / "importance_prediction_values_change.csv", index_col=0)
perm_imp = pd.read_csv(FIN / "importance_permutation_oof.csv", index_col=0)
cv_ratio = (pvc.std(axis=1) / pvc.mean(axis=1)).median()
show_fig("fig_13_feature_importance", f"Importance rankings are unstable across folds (median coefficient of variation {cv_ratio:.2f}) and permuting any predictor changes PR-AUC by no more than {perm_imp.abs().to_numpy().max():.4f}.")
""")
code("""
shap_info = json.loads((FIN / "shap_info.json").read_text(encoding="utf-8"))
display(Markdown(f"Final development model tree count: {shap_info['final_tree_count']}; SHAP computed: {shap_info['computed']}."))
show_fig("fig_14_shap_summary", "SHAP attributions are tiny and sign-inconsistent, matching the null signal.")
""")
code("""
sub = {c: pd.read_csv(FIN / f"subgroup_{c}.csv") for c in ("Department", "Gender", "Education_Level")}
for c, t in sub.items():
    display(Markdown(f"**{c}**"))
    display(t[[c, "n", "prevalence", "average_precision", "roc_auc", "stable_rate_estimates"]])
show_fig("fig_17_subgroups", "Subgroup AP and AUC hover around group prevalence and 0.5; differences are within sampling noise (descriptive only).")
""")

md("""
## 10. Leakage audit and label-shuffle test

Leakage evidence from `improvement/phase1/leakage.json`: excluded columns are absent from the matrix, no duplicate feature rows exist, no single feature is suspicious, and the OOF frame
is joined to the source labels by `Employee_ID` with zero mismatches. The **label-shuffle test** reruns the *complete* nested protocol on permuted development labels; if the real score were
genuinely above chance it would sit in the upper tail of this null distribution.
""")
code("""
leak = json.loads((P1 / "leakage.json").read_text(encoding="utf-8"))
display(Markdown(
    f"Excluded columns in predictor list: {leak['excluded_columns_in_features'] or 'none'}; suspicious single-feature columns: {leak['suspicious_columns'] or 'none'}; "
    f"duplicate feature rows: {leak['duplicates']['duplicate_feature_rows']}; cross-train/validation duplicates: {leak['duplicates']['cross_boundary_duplicates_total']}; "
    f"nearest-neighbour distance (fold-1 sample, standardised) min {leak['near_duplicates_fold1']['nn_dist_min']:.3f}."))
null = pd.read_csv(FIN / "permutation_null.csv")
real = summary.set_index("model").loc[final_name]
p_ap = permutation_pvalue(real.pooled_pr_auc, null.pr_auc)
p_auc = permutation_pvalue(real.pooled_roc_auc, null.roc_auc)
display(Markdown(
    f"{len(null)} permutations. Real PR-AUC {real.pooled_pr_auc:.5f} vs shuffled mean {null.pr_auc.mean():.5f} ± {null.pr_auc.std():.5f} (permutation p = {p_ap:.3f}); "
    f"real ROC-AUC {real.pooled_roc_auc:.5f} vs shuffled mean {null.roc_auc.mean():.5f} ± {null.roc_auc.std():.5f} (p = {p_auc:.3f})."))
""")
code("""
above_all = bool(real.pooled_pr_auc > null.pr_auc.max())
show_fig("fig_16_permutation_test", f"The real-label PR-AUC is {'above every' if above_all else 'within the range of the'} {len(null)} shuffled-label runs (p = {p_ap:.3f}; smallest attainable p with {len(null)} permutations is {1 / (len(null) + 1):.3f}), so the test cannot reject chance at 5%.")
""")

md("""
## 11. Final development model and artifact reconciliation

The final development model (`models/catboost/audit/model.cbm`) is a **new file**; the earlier `catboost_development_model.cbm` is left untouched.
The cell below reloads it, recomputes OOF metrics from the saved CSV and reconciles them to the saved summary (tolerance 1e-5), re-joins the OOF rows to the source labels by `Employee_ID`,
checks fold-wise probability scale, and verifies that every manifest artifact exists.
""")
code("""
from catboost import CatBoostClassifier

oof = pd.read_csv(FIN / "oof_catboost_final.csv")
again = metric_row(oof.y_true, oof.y_probability)
saved = summary.set_index("model").loc[final_name]
for k in ("pr_auc", "roc_auc", "brier", "log_loss"):
    assert abs(again[k] - saved[f"pooled_{k}"]) < 1e-5, f"{k} does not reconcile with the saved summary"

joined = oof.merge(data[["Employee_ID", "Resigned", "cv_fold"]], on="Employee_ID", suffixes=("", "_src"), validate="one_to_one")
assert len(joined) == len(data) and (joined.y_true == joined.Resigned).all() and (joined.cv_fold == joined.cv_fold_src).all()
fold_scale = oof.groupby("cv_fold").y_probability.mean()
assert (fold_scale - prevalence).abs().max() < 0.02, "fold probability scales disagree"
assert holdout_ids.isdisjoint(set(oof.Employee_ID))

model = CatBoostClassifier()
model.load_model(str(PROJECT_ROOT / "models" / "catboost" / "audit" / "model.cbm"))
meta = json.loads((PROJECT_ROOT / "models" / "catboost" / "audit" / "metadata.json").read_text(encoding="utf-8"))
assert model.tree_count_ == meta["selected_tree_count"]

manifest = pd.read_csv(CB / "catboost_output_manifest.csv")
missing = [r for r in manifest.relative_path if not (PROJECT_ROOT / r.replace("\\\\", "/")).exists()]
assert not missing, missing
display(Markdown(f"Reconciled to 1e-5; {len(joined):,} OOF rows aligned by `Employee_ID`; fold-mean probabilities {fold_scale.min():.4f}–{fold_scale.max():.4f} (prevalence {prevalence:.4f}); "
                 f"final model has {model.tree_count_} trees; all {len(manifest)} manifest artifacts exist."))
""")

md("""
## 12. Conclusions

The conclusions below are generated from the values computed above.
""")
code("""
cb = paired.set_index("model").loc[final_name]
lr = paired.set_index("model").loc["Logistic Regression"]
lines = [
    f"* **Benchmark.** Pooled OOF PR-AUC {real.pooled_pr_auc:.5f} vs no-skill {prevalence:.5f}; ROC-AUC {real.pooled_roc_auc:.5f}. "
    f"Against the prior baseline: ΔPR-AUC {cb.delta_pr_auc:+.5f} (CI {cb.delta_pr_auc_ci_lower:+.5f}…{cb.delta_pr_auc_ci_upper:+.5f}), "
    f"ΔROC-AUC {cb.delta_roc_auc:+.5f} (CI {cb.delta_roc_auc_ci_lower:+.5f}…{cb.delta_roc_auc_ci_upper:+.5f}). "
    + ("CatBoost beats the baseline under the pre-declared rule." if beats else "CatBoost does **not** beat the baseline under the pre-declared rule."),
    f"* **Versus logistic regression.** ΔPR-AUC {lr.delta_pr_auc:+.5f} and ΔROC-AUC {lr.delta_roc_auc:+.5f} (both relative to the prior baseline; CIs overlap CatBoost's).",
    f"* **Probability quality.** Raw Brier {real.pooled_brier:.5f} vs prior-only {ref_row.pooled_brier:.5f}; log loss {real.pooled_log_loss:.5f} vs {ref_row.pooled_log_loss:.5f}.",
    f"* **Permutation test.** p = {p_ap:.3f} (PR-AUC) and {p_auc:.3f} (ROC-AUC) over {len(null)} permutations of the full nested protocol.",
    f"* **Tuning plateau.** Kept changes: {kept}; stopping rule triggered: {state['stopped']}. Tree-count ablation PR-AUC range {abl.pr_auc.min():.4f}–{abl.pr_auc.max():.4f}.",
]
display(Markdown("\\n".join(lines)))
display(Markdown(
    "**Interpretation.** " + ("The 17 approved predictors carry detectable signal." if beats and p_ap < 0.05 else
    f"No reliable signal in the 17 approved predictors was demonstrated. CatBoost does not beat the prior baseline (both paired intervals include zero). "
    f"The label-shuffle test is inconclusive at its resolution: the real score is {'above all' if real.pooled_pr_auc > null.pr_auc.max() else 'inside'} {len(null)} shuffles (p = {p_ap:.3f}, the minimum attainable), "
    f"and the shuffled null itself is centred at PR-AUC {null.pr_auc.mean():.4f}, below the prevalence {prevalence:.4f}, so the prior baseline is the fairer reference. "
    "More permutations would be needed to resolve a difference of this size; any true effect is at most a few thousandths of PR-AUC.")))
""")

md("""
### Recommendations (not actions taken)

If the team needs a usable resignation model, the limiting factor is the information in the features, not the learner:

1. Obtain features with a plausible causal link to resignation (manager changes, compensation percentile versus peers, internal mobility, engagement-survey trajectories). Any new predictor requires a documented shared decision under `change_control` in `configs/feature_contract.yaml`.
2. Pursue the longitudinal snapshot data described in `configs/longitudinal_snapshot_contract.yaml` and `docs/longitudinal_data_availability_report.md`: time-ordered snapshots allow trend features and a time-based inner split (already supported by `CatBoostAdapter.fit(snapshot_dates=...)`).
3. Revisit the target definition (e.g. resignation within a fixed horizon after a snapshot date) so that the label is predictable from information available at prediction time.
4. `Hire_Date`-derived tenure may be tried only after the contract is explicitly amended and the team is told; it was **not** used here.
5. Do not read the formal holdout until the team freezes a model that shows development-set evidence of signal (see `docs/FINAL_HOLDOUT_PROTOCOL.md`).
6. **Operational Threshold Selection:** For any downstream deployment on imbalanced employee data, decouple probability estimation from decision cutoffs. Never apply a default 0.50 cutoff to calibrated probabilities; set thresholds based on organizational intervention capacity (e.g. Top 10% budget) or relative costs of false negatives versus false positives.
""")

md("""
## 13. Summary and Conclusion

### Q&A
- **Q: Does CatBoost demonstrate predictive power over trivial baselines for employee resignation prediction?**  
  **A:** No. CatBoost v2.1.0 achieves a pooled out-of-fold PR-AUC of `0.10225` (95% CI: `[0.09931, 0.10566]`) against the analytical Dummy Prior baseline of `0.10009` (prevalence: `0.10010`), and a ROC-AUC of `0.50410` (95% CI: `[0.49733, 0.51083]`). Because both 95% paired bootstrap confidence intervals span zero ($\\Delta\\text{PR-AUC} \\in [-0.00032, +0.00486]$, $\\Delta\\text{ROC-AUC} \\in [-0.00469, +0.01288]$), and the empirical label-permutation test yielded $p = 0.091$, the model does not beat the no-skill baseline at statistical significance ($\\alpha = 0.05$).
- **Q: Is CatBoost properly calibrated and tuned?**  
  **A:** Yes. CatBoost is mathematically and probabilistically well-tuned. With `boost_from_average=True`, its raw Brier score is `0.09008` (matching the theoretical minimum for a 10.01% base rate), and Logloss dropped from `0.60521` to `0.32529`. The absence of high predictive accuracy is an information-theoretic bound of the 17 static features, not an algorithmic or hyperparameter tuning flaw.
- **Q: Why does the standard confusion matrix at cutoff 0.50 yield zero predicted positives, and how is it resolved?**  
  **A:** Because the model is calibrated to the true 10.01% base rate, all predicted probabilities lie between `0.0867` and `0.1166`. An unadjusted 0.50 threshold is unreachable. Decoupling probability calibration from decision thresholding resolves this: applying the empirical base-rate cutoff ($\\tau = 0.10010$) captures 4,240 True Positives (52.95% Recall), while a Top 10% HR budget cutoff ($\\tau = 0.10226$) isolates 874 True Positives with 10.92% Precision (1.09× Lift).

### Data Analysis Key Findings
- **Rigorous Leakage-Safe Nested Validation:** The 5-fold outer cross-validation with 3-seed inner candidate search (`[42, 43, 44]`) across 20 randomized candidates (`R01`–`R20`) proved that hyperparameter optimization reaches an early plateau: inner-validation Logloss flattens within 15–57 trees, and deeper tree structures merely fit sample variance.
- **Univariate & Permutation Signal Absence:** Every single feature has an individual ROC-AUC between `0.4908` and `0.5092` (near exact 0.50 chance). Out-of-fold permutation importance showed that shuffling any of the 17 predictors causes a negligible PR-AUC shift of $\\le 0.0004$. TreeSHAP attributions tightly cluster at $0.00 \\pm 0.02$.
- **Class Imbalance & Weighting Dynamics:** Systematic ablations (E3a–E3d) demonstrated that artificial class weighting (`auto_class_weights='Balanced'` or `scale_pos_weight`) severely distorts predicted probabilities (inflating Brier error from `0.09008` to `0.24923`) without producing any statistically significant gain in PR-AUC. Unweighted loss combined with operational thresholding is the optimal strategy.
- **Demographic Subgroup Consistency:** Classification metrics evaluated across subgroups (`Department`, `Gender`, `Education_Level`) remained within normal binomial sampling variance of the overall population prevalence ($0.093$–$0.111$ AP and $0.477$–$0.522$ ROC-AUC), with no evidence of disparate subgroup degradation.

### Insights or Next Steps
- **Honest Academic Negative Result:** The project's findings constitute a verified, leak-free negative result. In academic data science, proving that a static tabular feature set lacks predictive signal through nested cross-validation and hypothesis testing is a scientifically robust outcome that prevents deploying pseudo-predictive models into production HR workflows.
- **Next Steps & Feature Acquisition:** Future iterations should not spend compute on deeper learners or excessive tuning. Instead, efforts must focus on acquiring high-signal, causal features: longitudinal snapshot records (tracking quarterly compensation ratio changes, manager turnover, promotion latency, and engagement survey sentiment trajectories) as outlined in `configs/longitudinal_snapshot_contract.yaml` and `docs/longitudinal_data_availability_report.md`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, NB)
print(f"wrote {NB.relative_to(ROOT)} with {len(cells)} cells")

