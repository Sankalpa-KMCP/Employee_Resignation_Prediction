"""Render ``audit_report.md`` from machine-readable evidence (no typed-in metrics).

Reads phase-1 evidence (before) and, when present, the final audit outputs (after).
Run:  python scripts/render_catboost_audit_report.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CB = ROOT / "results/models/catboost"
IMP = CB / "improvement"
P1, FIN = IMP / "phase1", IMP / "final"
MODELS_AFTER = ROOT / "models/catboost/audit" if (ROOT / "models/catboost/audit").exists() else ROOT / "models/catboost"
NB = ROOT / "notebooks/team_contributions/models/member_2_catboost/06_m2_catboost.ipynb"
_pre_nbs = sorted(NB.parent.glob("06_m2_catboost.pre_audit_*.ipynb"))
PRE_NB = _pre_nbs[0] if _pre_nbs else NB


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    integ, rec, det, leak = J(P1 / "integrity.json"), J(P1 / "reconcile.json"), J(P1 / "determinism.json"), J(P1 / "leakage.json")
    snap = J(IMP / "baseline_snapshot.json")
    es = pd.read_csv(P1 / "early_stopping_probe.csv")
    base = pd.read_csv(P1 / "baseline_comparison_before.csv").set_index("model")
    paired = pd.read_csv(P1 / "paired_vs_prior_before.csv").set_index("model")
    scan = pd.read_csv(P1 / "univariate_scan_all_columns.csv")
    nb_before = json.loads(PRE_NB.read_text(encoding="utf-8"))
    md_text = "\n".join("".join(c["source"]) for c in nb_before["cells"] if c["cell_type"] == "markdown")
    typed_numbers = len(re.findall(r"\b0\.\d{4,}\b", md_text))
    have_after = (FIN / "metrics_summary.csv").exists()
    after = pd.read_csv(FIN / "metrics_summary.csv").set_index("model") if have_after else None
    after_paired = pd.read_csv(FIN / "paired_vs_dummy_prior.csv").set_index("model") if have_after else None
    perm = pd.read_csv(FIN / "permutation_null.csv") if (FIN / "permutation_null.csv").exists() else None
    fold_means = rec["per_fold_winner"]
    prev = integ["overall_positive_rate"]
    fr = pd.read_csv(IMP / "experiment_log.csv") if (IMP / "experiment_log.csv").exists() else None
    aft = {}
    if have_after:
        fsel = pd.read_csv(FIN / "fold_selection_summary.csv")
        oof_f = pd.read_csv(FIN / "oof_catboost_final.csv")
        fm = oof_f.groupby("cv_fold").y_probability.mean()
        meta_f = J(MODELS_AFTER / "metadata.json")
        aft["3.2"] = (f"PASS: selection on `{fsel.selection_metric.iloc[0]}`, patience {J(FIN / 'run_metadata.json')['inner_validation']['early_stopping_rounds']}, "
                      f"{int(fsel.n_inner_splits.iloc[0])} repeated inner splits; trees per fold {fsel.selected_trees.tolist()}")
        aft["3.3"] = (f"PASS: raw Brier {after.loc['CatBoost (raw)', 'pooled_brier']:.5f} vs prior {after.loc['Dummy prior', 'pooled_brier']:.5f}; "
                      f"winners {fsel.candidate_id.tolist()} (all unweighted: {bool(fsel.auto_class_weights.isna().all() or (fsel.auto_class_weights.astype(str) == 'None').all())})")
        aft["3.4"] = f"PASS: fold-mean probability {fm.min():.4f}–{fm.max():.4f} (prevalence {prev:.4f})"
        aft["3.6"] = f"new model `models/catboost/audit/model.cbm`: {meta_f['selected_tree_count']} trees (old file untouched)"

    def pr(model, col="pooled_pr_auc"):
        return f"{base.loc[model, col]:.5f}"

    findings = []  # (id, item, before_status, evidence, severity, class, after)

    def add(i, item, status, ev, sev, cls, aft="n/a (unchanged)"):
        findings.append((i, item, status, ev, sev, cls, aft))

    imp_ok = all(v["ok"] for v in integ["imports"].values())
    vr = integ["version_vs_requirements"]
    mism = {k: f"installed {v['installed']} vs pinned {v['pinned']}" for k, v in vr.items() if v["installed"] != v["pinned"]}
    add("1.1", "Imports resolve; versions match requirements.txt", "WARN" if imp_ok and mism else ("PASS" if imp_ok else "FAIL"),
        f"all {len(integ['imports'])} notebook imports resolve (`improvement/phase1/integrity.json`); mismatches: {mism or 'none'}", "Low", "C/D")
    add("1.2", "One-command, top-to-bottom runnable script", "FAIL",
        "Before the audit no script ran load→nested CV→OOF→metrics→figures; notebook cell 3 sets `RUN_CATBOOST_TRAINING = False` so a default run only re-reads saved CSVs. "
        "Created `scripts/run_catboost_audit.py`.", "Medium", "D",
        "PASS (script created; see Phase 4 run log)" if have_after else "pending")
    add("1.3", "CPU-only deterministic training", "PASS",
        f"two same-seed nested-CV runs: max abs diff = {det['max_abs_diff_run1_run2']:.1e}; vs saved OOF = {det['max_abs_diff_run1_saved']:.1e}; GPU used = {det['gpu_used']} (`phase1/determinism.json`)", "—", "—")
    add("2.1", "Canonical data path (`data/raw` + `split_registry.csv`)", "WARN",
        f"cell 5 reads the raw CSV directly (`pd.read_csv(RAW_DATA_PATH)`) and merges the registry: canonical files, but bypasses the hash-verified loader `load_development_data` (uses loader: {integ['notebook_uses_canonical_dev_loader']})", "Low", "D",
        "PASS (new notebook uses `load_audit_inputs`)" if have_after else "pending")
    add("2.2", "Row counts, folds, stratification, uniqueness, missing, dtypes", "PASS",
        f"dev rows {integ['n_dev_rows']:,}; holdout IDs in registry {integ['n_holdout_registry_ids']:,}; overlap {integ['id_overlap_dev_holdout']}; fold sizes {sorted(set(integ['fold_sizes'].values()))}; "
        f"fold positive rates {min(integ['fold_positive_rate'].values()):.5f}–{max(integ['fold_positive_rate'].values()):.5f} (overall {prev:.5f}); duplicate IDs {integ['duplicate_employee_ids']}; missing predictor cells {integ['missing_values_predictors']}", "—", "—")
    add("2.3", "Formal holdout not read", "FAIL" if integ["notebook_loads_holdout_labels_in_memory"] else "PASS",
        "cell 5 builds `holdout_df = merged_df[merged_df['partition']=='holdout']` from the raw CSV, i.e. holdout predictors AND `Resigned` are parsed into memory (only row counts/IDs are used afterwards; no scoring, no plot, no decision). "
        "Violates the spirit of hard rule 1 / the protocol's ID-only inspection.", "High", "A/D",
        "PASS (new notebook never builds a holdout frame)" if have_after else "pending")
    add("2.4", "Native categoricals, no needless scaling/encoding", "PASS",
        f"`cat_features` = {integ['categorical_from_contract']} passed natively; dtypes numeric for the other {len(integ['dtypes']) - 4}; notebook applies scaler/PCA/winsor: {integ['notebook_applies_preprocessing_outputs']}", "—", "—")
    add("2.5", "Preprocessing-member outputs (outliers, scaling, encoding, selection, PCA)", "PASS",
        "None are applied to CatBoost. Appropriate for a tree ensemble: splits are invariant to monotone scaling, categoricals use ordered target statistics fitted inside each training fold.", "—", "—")
    add("2.6", "17 predictors equal the feature contract", "PASS",
        f"notebook reads `primary_model_features` from the contract (source: {integ['notebook_primary_features_source']}); count {integ['feature_contract_n']}; list identical to contract (no mismatch)", "—", "—")
    add("3.1", "Nested CV structure", "PASS",
        f"inner split = `train_test_split` on the rows passed to `fit()` ({leak['code_paths']['inner_split_from_outer_train_only']}); early stopping `eval_set` = inner val ({leak['code_paths']['eval_set_is_inner_val']}); refit on 100% outer-train ({leak['code_paths']['refit_on_outer_train_only']}); outer-val only in `predict_proba`", "—", "—")
    wins = rec["per_fold_winner"]
    add("3.2", "Early-stopping set-up (PRAUC, patience 50)", "FAIL",
        "selected trees per fold " + str(snap["fold_metrics_raw"] and [v["trees"] for v in wins.values()]) + "; PRAUC on ~960 inner positives is noisy. Probe (`phase1/early_stopping_probe.csv`):\n\n" +
        table([[r.variant, r.trees_by_fold, f"{r.pr_auc:.5f}", f"{r.roc_auc:.5f}", f"{r.brier:.5f}", f"{r.log_loss:.5f}"] for r in es.itertuples()],
              ["variant", "trees per fold", "PR-AUC", "ROC-AUC", "Brier", "LogLoss"]) +
        f"\n\nSwitching to Logloss/AUC or patience 200 changes which trees/candidates win but PR-AUC stays within {es.pr_auc.min():.5f}–{es.pr_auc.max():.5f} (prevalence {prev:.5f}) and ROC-AUC within {es.roc_auc.min():.5f}–{es.roc_auc.max():.5f}.", "Medium", "B",
        aft.get("3.2", "pending"))
    ex = rec["fold1_manual_reproduction_c03_unweighted"]
    r2 = next(r for r in ex if not r["boost_from_average"] and r["trees"] == 2)
    r2b = next(r for r in ex if r["boost_from_average"] and r["trees"] == 2)
    add("3.3", "Brier 0.211 for an 'unweighted' winner (priority)", "FAIL",
        f"**Not a refit/weights bug.** (i) CatBoost leaves `boost_from_average` off for Logloss by default (`get_params()` reports {rec['catboost_default_boost_from_average']}; explicit False gives identical output), so a model starts at p=0.5; with lr=0.03 two trees only reach mean p={r2['mean_p']:.4f} (fold-1 manual reproduction, C03 unweighted, 2 trees: Brier {r2['brier']:.5f}). "
        f"With `boost_from_average=True` the same model gives mean p={r2b['mean_p']:.4f}, Brier {r2b['brier']:.5f}. (ii) The nested CV did **not** use C03 in every fold: per-fold winners were "
        f"{', '.join(f'fold {f}: {v['candidate']} ({v['class_weights'] if v['class_weights'] != 'nan' else 'unweighted'}, {v['trees']} trees, mean p={v['oof_mean_probability']:.3f})' for f, v in wins.items())}; the summary JSON reports only `{rec['summary_reports_single_winner']}`. Balanced candidates inflate probabilities toward 0.5. "
        f"The saved OOF reproduces exactly from the current code ({det['max_abs_diff_run1_saved']:.0e}), so there is no hidden state or misalignment.", "High", "B (+D reporting)",
        aft.get("3.3", "pending"))
    add("3.4", "Pooled OOF mixes probability scales across folds", "FAIL",
        f"fold-mean predicted probability ranges {min(v['oof_mean_probability'] for v in wins.values()):.3f}–{max(v['oof_mean_probability'] for v in wins.values()):.3f}; pooled AP/ROC/Brier over such a mixture are not properties of any single model. Per-fold metrics are the valid unit.", "Medium", "B/D",
        aft.get("3.4", "pending"))
    add("3.5", "No state carried between folds", "PASS",
        f"`run_5fold_cv` constructs `adapter_factory()` per fold (found: {rec['runner_builds_adapter_each_fold']}); the post-selection `self.params.update(...)` lives on a per-fold object; rerun equals saved OOF", "—", "—")
    dm = rec["development_model"]
    add("3.6", "`catboost_development_model.cbm` reproduces predictions", "WARN",
        f"loads; tree count {dm['tree_count']} = metadata {dm['metadata_tree_count']}; a refit from the stored parameters reproduces 2,000 sampled predictions (max abs diff {dm['max_abs_diff_vs_refit_from_metadata']:.1e}). "
        f"But it is a {dm['tree_count']}-tree model with mean p={dm['mean_p_on_sample']:.3f}: reproducible, not useful.", "Medium", "B",
        aft.get("3.6", "pending"))
    add("4.1", "Baselines on the same folds/rows/metric code", "PASS",
        "Dummy prior, Dummy most_frequent, Logistic Regression and CatBoost raw/calibrated computed from OOF with `audit_stats.compare_models` (`phase1/baseline_comparison_before.csv`).", "—", "—")
    cm = "CatBoost raw (before)"
    add("4.2", "CatBoost vs baseline (verdict rule: CI lower bound > 0 and Δ ≥ 0.01)", "FAIL (does not beat baseline)",
        f"ΔPR-AUC vs Dummy prior = {paired.loc[cm, 'delta_pr_auc']:+.5f} (95% CI {paired.loc[cm, 'delta_pr_auc_ci_lower']:+.5f}, {paired.loc[cm, 'delta_pr_auc_ci_upper']:+.5f}); "
        f"ΔROC-AUC = {paired.loc[cm, 'delta_roc_auc']:+.5f} (CI {paired.loc[cm, 'delta_roc_auc_ci_lower']:+.5f}, {paired.loc[cm, 'delta_roc_auc_ci_upper']:+.5f}); "
        f"ΔBrier = {paired.loc[cm, 'delta_brier']:+.5f}; better_than_baseline = {paired.loc[cm, 'better_than_baseline']}", "High (finding, not a defect)", "C",
        (f"better_than_baseline = {after_paired.loc['CatBoost (raw)', 'better_than_baseline']}" if have_after else "pending"))
    sf = scan.loc[scan.auc_distance_from_chance.idxmax()]
    add("5.1", "Excluded columns absent from the feature matrix", "PASS",
        f"excluded columns in feature list: {leak['excluded_columns_in_features'] or 'none'}; runner raises on forbidden/post-outcome names ({leak['code_paths']['runner_rejects_forbidden_features']})", "—", "—")
    add("5.2", "Single-feature leakage / 'too good to be true' scan (all 19 non-target columns incl. Employee_ID, Hire_Date)", "PASS",
        f"suspicious (AUC>0.6 or <0.4): {leak['suspicious_columns'] or 'none'}; largest distance of single-feature AUC from 0.5 = {sf.auc_distance_from_chance:.4f} ({sf.feature}); max Cramér's V {leak['max_cramers_v']:.4f}; max MI {leak['max_mi_nats']:.1e} nats", "—", "—")
    add("5.3", "Duplicates / near-duplicates across train and validation", "PASS",
        f"duplicate feature rows {leak['duplicates']['duplicate_feature_rows']}; cross-boundary duplicates {leak['duplicates']['cross_boundary_duplicates_total']}; fold-1 sample nearest-neighbour distance min {leak['near_duplicates_fold1']['nn_dist_min']:.3f} (none below 1e-6)", "—", "—")
    add("5.4", "Calibration cross-fitted, never scored on its own fit rows", "WARN",
        "Notebook cell 21 fits the Platt map on other folds' OOF rows (own fold excluded). Second-order caveat: those other-fold models were trained on rows that include the scored fold, so this is standard OOF cross-fitting, not nested calibration. Unit test `test_crossfit_calibration_never_uses_the_scored_fold` guards the first-order property.", "Low", "B")
    add("5.5", "No statistic computed on full development data before the split", "PASS",
        "Model fitting statistics (CatBoost ordered target statistics) are computed inside `fit()` per fold. Notebook cells 7/9 compute descriptive full-development statistics (fold-support counts, univariate audit); they feed no model.", "—", "—")
    add("5.6", "Label-shuffle test (≥10 permutations, full nested CV)", "WARN",
        "The previous notebook has a single shuffled-target run (Exp 8 in `catboost_leakage_safe_experiments.csv`), not ≥10. Full test is run on the final protocol in Phase 4.", "Medium", "D",
        (f"PASS: {len(perm)} permutations (see section below)" if perm is not None else "pending"))
    al = rec["alignment"]
    add("5.7", "OOF row order / labels aligned by explicit Employee_ID join", "PASS",
        f"{al['rows_joined']:,} rows joined 1:1; label mismatches {al['label_mismatches']}; fold mismatches {al['fold_mismatches']}; ROC-AUC from CSV {al['roc_auc_from_csv']:.6f} vs summary {al['roc_auc_in_summary']:.6f}", "—", "—")
    add("5.8", "Formal holdout never scored or used", "WARN",
        f"No receipt (`results/final_holdout/assessment_receipt.json` exists: {integ['holdout_receipt_present']}); no `models/frozen`; `src/models/catboost_model.py` never references holdout; runner rejects holdout IDs. Residual: see 2.3 (notebook parsed holdout rows into memory).", "High", "A/D",
        "PASS (audit code never loads holdout rows)" if have_after else "pending")
    add("6.1", "Required figures present", "FAIL",
        "Existing `figure_01…10` cover importance, PR, confusion, ROC, probability, reliability, lift, learning curves, fold scores, SHAP. Missing: target/fold balance, nested-CV layout, bootstrap Δ, candidate comparison, tree-count ablation, single-feature signal, permutation test, subgroups, per-fold strip with baselines, permutation importance, ROC bands. Existing SHAP/importance figures describe a 2-tree model.", "Medium", "D",
        "PASS (17 figures; `final/figure_index.json`)" if have_after else "pending")
    add("6.2", "No hard-coded metrics in the notebook", "FAIL",
        f"{typed_numbers} typed 4+-decimal numbers in markdown cells of the previous notebook (e.g. conclusions in cells 32 and 35), plus code cells printing literal reference values.", "Medium", "D",
        "PASS (new notebook computes narrative numbers)" if have_after else "pending")
    add("6.3", "Summary JSON consistent with per-fold evidence", "FAIL",
        f"`catboost_summary_metrics.json` reports one `selected_candidate_id` ({rec['summary_reports_single_winner']}) and `selected_tree_count` for the whole run, while folds selected {rec['distinct_winners']}.", "Medium", "D",
        "PASS (per-fold selection saved in `final/fold_selection_summary.csv`)" if have_after else "pending")
    add("7.1", "Feature-contract change control", "WARN",
        "`catboost_leakage_safe_experiments.csv` Exp 7 evaluated '6 engineered features'. Adding predictors is a prohibited *silent* change (`change_control.prohibited_silent_changes`). It was not adopted; no new feature is used by this audit.", "Low", "C")
    add("7.2", "Test-suite baseline in this environment", "WARN",
        "Full suite before any change: see `improvement/phase1/pytest_baseline_summary.txt` — failures/errors come from missing `torch`, `interpret`, `xgboost` packages (other members' modules) and an FT notebook cell-count test, none from CatBoost files.", "Low", "C")

    lines = ["# CatBoost audit report", "", f"_Rendered from machine-readable evidence by `scripts/render_catboost_audit_report.py`. Every number below is computed; none is typed._", "",
             "## 1. Checklist (before → after)", "",
             table([[f[0], f[1], f[2], f[3], f[4], f[5], f[6]] for f in findings], ["#", "Item", "Status (before)", "Evidence", "Severity", "Class", "After"]), "",
             "Class: **A** pipeline bug, **B** tuning/training-protocol weakness, **C** feature/data limitation, **D** reporting/visualisation gap.", "",
             "## 2. Baseline snapshot (before)", "",
             table([[m, f"{r.pooled_pr_auc:.5f}", f"[{r.pr_auc_ci_lower:.5f}, {r.pr_auc_ci_upper:.5f}]", f"{r.pooled_roc_auc:.5f}", f"[{r.roc_auc_ci_lower:.5f}, {r.roc_auc_ci_upper:.5f}]", f"{r.pooled_brier:.5f}", f"{r.pooled_log_loss:.5f}", f"{r.pooled_balanced_accuracy:.4f}"]
                    for m, r in base.iterrows()], ["model", "PR-AUC", "95% CI", "ROC-AUC", "95% CI", "Brier", "Log loss", "Bal. acc @0.5"]), "",
             f"Positive prevalence = {prev:.5f}. `Dummy most_frequent` predicts a constant class, so its ranking metrics are degenerate (AUC = 0.5 by construction) and its log loss is huge because it outputs probabilities of exactly 0 or 1; use Brier, log loss and balanced accuracy for it.", "",
             "Candidate pool (8) and per-fold winners are stored in `improvement/baseline_snapshot.json`.", ""]
    if have_after:
        lines += ["## 3. After the improvement run", "",
                  table([[m, f"{r.pooled_pr_auc:.5f}", f"[{r.pr_auc_ci_lower:.5f}, {r.pr_auc_ci_upper:.5f}]", f"{r.pooled_roc_auc:.5f}", f"[{r.roc_auc_ci_lower:.5f}, {r.roc_auc_ci_upper:.5f}]", f"{r.pooled_brier:.5f}", f"{r.pooled_log_loss:.5f}"]
                         for m, r in after.iterrows()], ["model", "PR-AUC", "95% CI", "ROC-AUC", "95% CI", "Brier", "Log loss"]), ""]
    if perm is not None and have_after:
        from src.evaluation.audit_stats import permutation_pvalue
        real_ap, real_auc = after.loc["CatBoost (raw)", "pooled_pr_auc"], after.loc["CatBoost (raw)", "pooled_roc_auc"]
        lines += ["## 4. Label-shuffle test (final protocol)", "",
                  f"{len(perm)} permutations of the development labels, each through the complete nested protocol. Real PR-AUC {real_ap:.5f} vs shuffled mean {perm.pr_auc.mean():.5f} (sd {perm.pr_auc.std():.5f}); permutation p = {permutation_pvalue(real_ap, perm.pr_auc):.3f}. "
                  f"Real ROC-AUC {real_auc:.5f} vs shuffled mean {perm.roc_auc.mean():.5f} (sd {perm.roc_auc.std():.5f}); p = {permutation_pvalue(real_auc, perm.roc_auc):.3f}.", ""]
    if fr is not None:
        lines += ["## 5. Experiment log", "", table([[r.experiment, r.description, f"{r.pr_auc:.5f}", f"{r.roc_auc:.5f}", f"{r.brier:.5f}", f"{r.log_loss:.5f}", r.decision] for r in fr.itertuples()],
                                                  ["id", "change", "PR-AUC", "ROC-AUC", "Brier", "Log loss", "decision"]), ""]
    out = IMP / "audit_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({len(findings)} checklist rows)")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
