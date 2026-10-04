"""Phase 0/1 audit of the CatBoost nested-CV pipeline (diagnostic only, no tuning).

Writes machine-readable evidence to ``results/models/catboost/improvement/phase1/``.
``audit_report.md`` is rendered from that evidence by ``scripts/render_catboost_audit_report.py``
so every number in the report is computed, not typed.

Usage (from project root):  python scripts/catboost_audit_phase1.py [--stage NAME ...]
Stages: snapshot integrity reconcile determinism early_stopping leakage baselines
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.audit_stats import (  # noqa: E402
    compare_models, duplicate_audit, load_audit_inputs, metric_row, near_duplicate_audit, per_fold_metrics,
    univariate_scan,
)
from src.models.catboost_model import CatBoostAdapter  # noqa: E402
from src.models.config import load_model_config  # noqa: E402
from src.models.runner import run_5fold_cv  # noqa: E402

CB_DIR = PROJECT_ROOT / "results" / "models" / "catboost"
OUT = CB_DIR / "improvement"
P1 = OUT / "phase1"
SEED = 42


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


def run_cv(data, split, feats, factory):
    frame = data.drop(columns=["cv_fold"], errors="ignore")
    return run_5fold_cv(factory, frame, split, feats, expected_model_name="CatBoost Classifier")


def stage_snapshot(ctx):
    """Phase 0: record candidate pool, winners, tree counts and metrics recomputed from saved OOF."""
    oof = pd.read_csv(CB_DIR / "catboost_oof_predictions.csv")
    cal = pd.read_csv(CB_DIR / "catboost_calibrated_oof_predictions.csv")
    sel = pd.read_csv(CB_DIR / "catboost_fold_candidate_selection.csv")
    cfg = load_model_config("catboost", strict_schema=True)
    summary = json.loads((CB_DIR / "catboost_summary_metrics.json").read_text(encoding="utf-8"))
    winners = sel[sel["is_fold_winner"]][["fold", "candidate_id", "selected_trees", "auto_class_weights", "inner_ap"]]
    snap = {
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "git_head": __import__("subprocess").run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip(),
        "candidate_pool": [{"id": c["id"], **c["params"]} for c in cfg["candidates"]],
        "inner_validation": cfg["inner_validation"],
        "fixed_params": cfg.get("fixed_params", {}),
        "reported_winning_candidate": summary.get("selected_candidate_id"),
        "reported_tree_count": summary["winning_candidate_params"]["selected_tree_count"],
        "per_fold_winners": winners.to_dict(orient="records"),
        "metrics_recomputed_from_saved_oof": {
            "raw": metric_row(oof.y_true, oof.y_probability),
            "calibrated": metric_row(cal.y_true, cal.calibrated_y_probability),
        },
        "fold_metrics_raw": per_fold_metrics(oof).to_dict(orient="records"),
        "raw_probability_summary": {"min": float(oof.y_probability.min()), "mean": float(oof.y_probability.mean()),
                                     "max": float(oof.y_probability.max())},
        "oof_sha256": hashlib.sha256((CB_DIR / "catboost_oof_predictions.csv").read_bytes()).hexdigest(),
    }
    jdump(snap, OUT / "baseline_snapshot.json")
    return snap


def stage_integrity(ctx):
    """Checklist item 2 + imports (item 1): canonical loading, fold sizes, dtypes, contract match."""
    data, split, feats, cats = ctx["data"], ctx["split"], ctx["feats"], ctx["cats"]
    import yaml
    contract = yaml.safe_load((PROJECT_ROOT / "configs/feature_contract.yaml").read_text(encoding="utf-8"))
    nb = json.loads((PROJECT_ROOT / "notebooks/team_contributions/models/member_2_catboost/06_m2_catboost.ipynb").read_text(encoding="utf-8"))
    nb_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ids = set(split.Employee_ID)
    dev_ids = set(split.loc[split.partition == "development", "Employee_ID"])
    hold_ids = set(split.loc[split.partition == "holdout", "Employee_ID"])
    fold_sizes = data.groupby("cv_fold").size().to_dict()
    fold_pos = data.groupby("cv_fold").Resigned.mean().to_dict()
    ev = {
        "n_dev_rows": len(data), "n_holdout_registry_ids": len(hold_ids),
        "id_overlap_dev_holdout": len(dev_ids & hold_ids), "dev_ids_equal_loaded_ids": dev_ids == set(data.Employee_ID),
        "fold_sizes": fold_sizes, "fold_positive_rate": fold_pos, "overall_positive_rate": float(data.Resigned.mean()),
        "duplicate_employee_ids": int(data.Employee_ID.duplicated().sum()),
        "missing_values_predictors": int(data[feats].isna().sum().sum()),
        "dtypes": {c: str(data[c].dtype) for c in feats},
        "feature_contract_n": len(feats), "feature_contract_features": feats, "categorical_from_contract": cats,
        "notebook_primary_features_source": "feature_contract['primary_model_features']" if "primary_model_features" in nb_code else "UNKNOWN",
        "notebook_loads_holdout_labels_in_memory": "holdout_df = merged_df[merged_df[\"partition\"] == \"holdout\"]" in nb_code,
        "notebook_uses_canonical_dev_loader": "load_development_data" in nb_code,
        "notebook_reads_raw_csv_directly": "pd.read_csv(RAW_DATA_PATH)" in nb_code,
        "notebook_default_mode_is_review": "RUN_CATBOOST_TRAINING = False" in nb_code,
        "notebook_applies_preprocessing_outputs": any(t in nb_code.lower() for t in ("standardscaler", "onehotencoder", "pca(", "winsor")),
        "holdout_receipt_present": (PROJECT_ROOT / "results/final_holdout/assessment_receipt.json").exists(),
        "frozen_pipeline_present": (PROJECT_ROOT / "models/frozen").exists(),
    }
    # Imports (item 1): every module the notebook imports must resolve
    mods = ["catboost", "sklearn", "numpy", "pandas", "matplotlib", "seaborn", "yaml", "scipy",
            "src.evaluation.metrics", "src.evaluation.oof_validation", "src.models.catboost_model",
            "src.models.config", "src.models.runner"]
    req = dict(l.strip().split("==") for l in (PROJECT_ROOT / "requirements.txt").read_text().splitlines() if "==" in l and not l.startswith("#"))
    imports = {}
    for m in mods:
        try:
            mod = importlib.import_module(m)
            imports[m] = {"ok": True, "version": getattr(mod, "__version__", None)}
        except Exception as exc:  # noqa: BLE001
            imports[m] = {"ok": False, "error": repr(exc)}
    pin = {"catboost": "catboost", "sklearn": "scikit-learn", "numpy": "numpy", "pandas": "pandas",
           "matplotlib": "matplotlib", "seaborn": "seaborn", "yaml": "pyyaml"}
    ev["imports"] = imports
    ev["version_vs_requirements"] = {k: {"installed": imports[k]["version"], "pinned": req.get(v)} for k, v in pin.items()}
    jdump(ev, P1 / "integrity.json")
    return ev


def stage_reconcile(ctx):
    """Item 3 priority: why is raw Brier 0.21 for an 'unweighted' winner?"""
    data, split, feats = ctx["data"], ctx["split"], ctx["feats"]
    oof = pd.read_csv(CB_DIR / "catboost_oof_predictions.csv")
    sel = pd.read_csv(CB_DIR / "catboost_fold_candidate_selection.csv")
    fold_mean_p = oof.groupby("cv_fold").y_probability.mean().to_dict()
    winners = sel[sel.is_fold_winner].set_index("fold")
    ev = {
        "per_fold_winner": {int(f): {"candidate": r.candidate_id, "class_weights": str(r.auto_class_weights),
                                      "trees": int(r.selected_trees), "oof_mean_probability": fold_mean_p[int(f)]}
                            for f, r in winners.iterrows()},
        "distinct_winners": sorted(winners.candidate_id.unique()),
        "summary_reports_single_winner": json.loads((CB_DIR / "catboost_summary_metrics.json").read_text())["selected_candidate_id"],
    }
    # Join OOF to source labels by Employee_ID (alignment audit)
    j = oof.merge(data[["Employee_ID", "Resigned", "cv_fold"]], on="Employee_ID", validate="one_to_one", suffixes=("", "_src"))
    ev["alignment"] = {"rows_joined": len(j), "label_mismatches": int((j.y_true != j.Resigned).sum()),
                       "fold_mismatches": int((j.cv_fold != j.cv_fold_src).sum()),
                       "oof_sorted_by_id": bool(oof.Employee_ID.is_monotonic_increasing),
                       "roc_auc_from_csv": metric_row(j.Resigned, j.y_probability)["roc_auc"]}
    summ = json.loads((CB_DIR / "catboost_summary_metrics.json").read_text())
    ev["alignment"]["roc_auc_in_summary"] = summ["group_a_raw_catboost_oof_metrics"]["raw_oof_roc_auc"]
    # Manual reproduction of the Brier mechanism on fold 1 (fixed config, no tuning)
    from catboost import CatBoostClassifier
    tr, va = data[data.cv_fold != 1], data[data.cv_fold == 1]
    cats = ctx["cats"]
    Xtr, Xva = tr[feats].copy(), va[feats].copy()
    for c in cats:
        Xtr[c], Xva[c] = Xtr[c].astype(str), Xva[c].astype(str)
    ci = [feats.index(c) for c in cats]
    rows = []
    for bfa in (False, True):
        for n_trees in (2, 4, 50, 300):
            m = CatBoostClassifier(iterations=n_trees, depth=4, learning_rate=0.03, l2_leaf_reg=5, loss_function="Logloss",
                                   boost_from_average=bfa, random_seed=SEED, verbose=False, allow_writing_files=False,
                                   thread_count=-1, cat_features=ci).fit(Xtr, tr.Resigned)
            p = m.predict_proba(Xva)[:, 1]
            rows.append({"boost_from_average": bfa, "trees": n_trees, "mean_p": float(p.mean()), **metric_row(va.Resigned, p)})
    ev["fold1_manual_reproduction_c03_unweighted"] = rows
    ev["catboost_default_boost_from_average"] = CatBoostClassifier(loss_function="Logloss").get_params().get("boost_from_average")
    ev["boost_from_average_in_repo_config"] = "boost_from_average" in (PROJECT_ROOT / "configs/models/catboost.yaml").read_text()
    # Fresh adapter per fold (state carry-over)
    src = (PROJECT_ROOT / "src/models/runner.py").read_text()
    ev["runner_builds_adapter_each_fold"] = "adapter = adapter_factory()" in src
    # Development model file
    from src.models.catboost_model import CatBoostAdapter as A
    mdl = CatBoostClassifier(); mdl.load_model(str(PROJECT_ROOT / "models/catboost/catboost_development_model.cbm"))
    meta = json.loads((PROJECT_ROOT / "models/catboost/metadata.json").read_text())
    X_all = data[feats].copy()
    for c in cats:
        X_all[c] = X_all[c].astype(str)
    sample = X_all.sample(2000, random_state=SEED)
    p_saved = mdl.predict_proba(sample)[:, 1]
    p = meta["params"]
    rebuilt = CatBoostClassifier(iterations=meta["selected_tree_count"], depth=p["depth"], learning_rate=p["learning_rate"],
                                 l2_leaf_reg=p["l2_leaf_reg"], loss_function="Logloss", random_seed=SEED, verbose=False,
                                 allow_writing_files=False, thread_count=-1, cat_features=ci,
                                 **({"auto_class_weights": p["auto_class_weights"]} if p.get("auto_class_weights") else {}))
    rebuilt.fit(X_all, data.Resigned)
    p_new = rebuilt.predict_proba(sample)[:, 1]
    ev["development_model"] = {"tree_count": int(mdl.tree_count_), "metadata_tree_count": meta.get("selected_tree_count"),
                                "max_abs_diff_vs_refit_from_metadata": float(np.abs(p_saved - p_new).max()),
                                "mean_p_on_sample": float(p_saved.mean())}
    jdump(ev, P1 / "reconcile.json")
    return ev


def stage_determinism(ctx):
    """Run the canonical nested CV twice with the same seed; diff both runs and the saved OOF."""
    data, split, feats = ctx["data"], ctx["split"], ctx["feats"]
    cfg = load_model_config("catboost", strict_schema=True)
    factory = lambda: CatBoostAdapter(config=cfg, candidate_id="auto")  # noqa: E731
    runs, times = [], []
    for _ in range(2):
        t0 = time.perf_counter()
        runs.append(run_cv(data, split, feats, factory))
        times.append(time.perf_counter() - t0)
    a, b = (r.predictions.sort_values("Employee_ID").y_probability.to_numpy() for r in runs)
    saved = pd.read_csv(CB_DIR / "catboost_oof_predictions.csv").sort_values("Employee_ID").y_probability.to_numpy()
    ev = {"max_abs_diff_run1_run2": float(np.abs(a - b).max()), "max_abs_diff_run1_saved": float(np.abs(a - saved).max()),
          "identical_run1_run2": bool(np.array_equal(a, b)), "runtime_seconds": times,
          "run1_metrics": metric_row(runs[0].predictions.y_true, runs[0].predictions.y_probability),
          "gpu_used": any(str(f["adapter_metadata"]["params"].get("task_type")).upper() != "CPU" for f in runs[0].metadata["fold_metadata"]),
          "fold_winners_run1": [f["adapter_metadata"]["candidate_id"] for f in runs[0].metadata["fold_metadata"]]}
    runs[0].predictions.to_csv(P1 / "oof_rerun_canonical_protocol.csv", index=False)
    jdump(ev, P1 / "determinism.json")
    return ev


def stage_early_stopping(ctx, only=None):
    """Probe: does eval_metric / patience / boost_from_average change tree counts and OOF metrics?"""
    data, split, feats = ctx["data"], ctx["split"], ctx["feats"]
    cfg = load_model_config("catboost", strict_schema=True)
    variants = {
        "baseline_PRAUC_p50": ({}, 50),
        "PRAUC_p200": ({}, 200),
        "Logloss_p100": ({"eval_metric": "Logloss"}, 100),
        "Logloss_p200": ({"eval_metric": "Logloss"}, 200),
        "AUC_p100": ({"eval_metric": "AUC"}, 100),
        "AUC_p200": ({"eval_metric": "AUC"}, 200),
        "Logloss_p200_bfa": ({"eval_metric": "Logloss", "boost_from_average": True}, 200),
    }
    rows = pd.read_csv(P1 / "early_stopping_probe.csv").to_dict("records") if (P1 / "early_stopping_probe.csv").exists() else []
    done = {r["variant"] for r in rows}
    for name, (params, patience) in variants.items():
        if name in done or (only and name not in only):
            continue
        t0 = time.perf_counter()
        r = run_cv(data, split, feats, lambda p=params, pa=patience: CatBoostAdapter(
            config=cfg, candidate_id="auto", params=copy.deepcopy(p), early_stopping_rounds=pa))
        trees = [f["adapter_metadata"]["selected_tree_count"] for f in r.metadata["fold_metadata"]]
        winners = [f["adapter_metadata"]["candidate_id"] for f in r.metadata["fold_metadata"]]
        rows.append({"variant": name, "patience": patience, "params": json.dumps(params), "trees_by_fold": trees,
                     "median_trees": float(np.median(trees)), "winners_by_fold": winners,
                     "runtime_s": time.perf_counter() - t0,
                     **metric_row(r.predictions.y_true, r.predictions.y_probability)})
        print(f"  {name}: trees={trees} winners={winners} PR-AUC={rows[-1]['pr_auc']:.5f} ROC={rows[-1]['roc_auc']:.5f} Brier={rows[-1]['brier']:.5f}")
        pd.DataFrame(rows).to_csv(P1 / "early_stopping_probe.csv", index=False)
    return rows


def stage_leakage(ctx):
    """Item 5: exclusions, univariate scan on all dev columns, duplicates, holdout paths, calibration/pre-split statistics."""
    root = PROJECT_ROOT
    full, split, feats, cats = load_audit_inputs(root, all_columns=True)
    contract_excl = {"Employee_ID", "Hire_Date", "Resigned"}
    leaked = contract_excl & set(feats)
    scan = univariate_scan(full, [c for c in full.columns if c not in ("Resigned", "cv_fold")], cats)
    scan.to_csv(P1 / "univariate_scan_all_columns.csv", index=False)
    numeric = [c for c in feats if c not in cats]
    ev = {"excluded_columns_in_features": sorted(leaked), "scan_rows": len(scan),
          "suspicious_columns": scan[scan.suspicious].feature.tolist(),
          "max_auc_distance_from_chance": float(scan.auc_distance_from_chance.max()),
          "max_auc_feature": scan.sort_values("auc_distance_from_chance").iloc[-1].feature,
          "max_cramers_v": float(scan.cramers_v.max()), "max_mi_nats": float(scan.mutual_info_nats.max()),
          "duplicates": duplicate_audit(full, feats), "near_duplicates_fold1": near_duplicate_audit(full, numeric)}
    # Code-path evidence for holdout + calibration + pre-split statistics
    adapter_src = (root / "src/models/catboost_model.py").read_text()
    runner_src = (root / "src/models/runner.py").read_text()
    ev["code_paths"] = {
        "adapter_mentions_holdout": "holdout" in adapter_src.lower(),
        "runner_rejects_holdout_ids": "CRITICAL LEAKAGE ERROR" in runner_src,
        "runner_rejects_forbidden_features": "Forbidden predictor columns" in runner_src,
        "inner_split_from_outer_train_only": "train_test_split(\n                    X_proc" in adapter_src,
        "eval_set_is_inner_val": "eval_set=(X_in_val, y_in_val)" in adapter_src,
        "refit_on_outer_train_only": "final_model.fit(X_proc, y_arr" in adapter_src,
    }
    jdump(ev, P1 / "leakage.json")
    return ev


def stage_baselines(ctx):
    """Item 4 (before): same 5 folds, same rows, same metric code - baselines vs the CURRENT CatBoost OOF."""
    final = CB_DIR / "improvement" / "final"
    cal = pd.read_csv(CB_DIR / "catboost_calibrated_oof_predictions.csv")
    cur = pd.read_csv(CB_DIR / "catboost_oof_predictions.csv")
    cal_df = cur.copy()
    cal_df["y_probability"] = cal.set_index("Employee_ID").loc[cur.Employee_ID, "calibrated_y_probability"].to_numpy()
    frames = {"Dummy prior": pd.read_csv(final / "oof_dummy_prior.csv"),
              "Dummy most_frequent": pd.read_csv(final / "oof_dummy_most_frequent.csv"),
              "Logistic Regression": pd.read_csv(final / "oof_logistic_regression.csv"),
              "CatBoost raw (before)": cur, "CatBoost calibrated (before)": cal_df}
    frames = {k: v.sort_values("Employee_ID").reset_index(drop=True) for k, v in frames.items()}
    summary, paired, topk, fold_tab, boot = compare_models(frames, "Dummy prior")
    summary.to_csv(P1 / "baseline_comparison_before.csv", index=False)
    paired.to_csv(P1 / "paired_vs_prior_before.csv", index=False)
    topk.to_csv(P1 / "topk_before.csv", index=False)
    print(summary[["model", "pooled_pr_auc", "pooled_roc_auc", "pooled_brier", "pooled_log_loss"]].to_string())
    print(paired[["model", "delta_pr_auc", "delta_pr_auc_ci_lower", "delta_pr_auc_ci_upper", "delta_roc_auc", "better_than_baseline"]].to_string())


def main():
    stages = {"snapshot": stage_snapshot, "integrity": stage_integrity, "reconcile": stage_reconcile,
              "determinism": stage_determinism, "baselines": stage_baselines, "early_stopping": stage_early_stopping, "leakage": stage_leakage}
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="*", default=list(stages))
    ap.add_argument("--variants", nargs="*", default=None, help="early_stopping variants to run (resumes; skips finished)")
    args = ap.parse_args()
    data, split, feats, cats = load_audit_inputs(PROJECT_ROOT)
    ctx = {"data": data, "split": split, "feats": feats, "cats": cats}
    np.random.seed(SEED)
    for name in args.stage:
        print(f"== {name}")
        stages[name](ctx, args.variants) if name == "early_stopping" else stages[name](ctx)


if __name__ == "__main__":
    main()
