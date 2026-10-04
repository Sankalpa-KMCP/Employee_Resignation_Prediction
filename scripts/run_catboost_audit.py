"""End-to-end CatBoost audit, runnable from the project root with one command.

    python scripts/run_catboost_audit.py

Seeds -> config -> canonical development data -> nested 5-fold CV -> saved OOF ->
baselines (Dummy prior / most_frequent / Logistic Regression) -> metrics with
bootstrap CIs and paired comparisons -> diagnostics -> label-shuffle test ->
figures -> manifest. Everything is computed from saved OOF predictions; no metric
is typed in. The formal holdout is never loaded: data come from the hash-verified
development loader, which skips holdout rows before parsing predictors or labels.

Options: --stages run baselines metrics diagnostics permutation figures manifest
         --n-permutations N (default 10; use --n-permutations 0 to skip)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import audit_plots as plots  # noqa: E402
from src.evaluation.audit_stats import (  # noqa: E402
    DEFAULT_KS, ci95, compare_models, crossfit_calibration, load_audit_inputs, metric_row,
    per_fold_metrics, permutation_pvalue, univariate_scan,
)
from src.evaluation.fairness import subgroup_classification_metrics  # noqa: E402
from src.models.catboost_model import CatBoostAdapter  # noqa: E402
from src.models.catboost_protocol import build_config, run_protocol  # noqa: E402
from src.models.logistic_regression import LogisticRegressionAdapter  # noqa: E402
from src.models.runner import run_5fold_cv  # noqa: E402

SEED = 42
CONFIG_NAME = "audited_catboost"  # cumulative kept changes of the improvement plan
N_BOOT = 1000
MIN_EFFECT = 0.01  # pre-declared practical-significance threshold for dPR-AUC and dROC-AUC
CB = PROJECT_ROOT / "results" / "models" / "catboost"
OUT = CB / "improvement" / "final"
FIG = OUT / "figures"
MODELS_OUT = PROJECT_ROOT / "models" / "catboost" / "audit"


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


class DummyAdapter:
    """DummyClassifier wrapper conforming to the ModelAdapter protocol."""

    def __init__(self, strategy: str, name: str) -> None:
        from sklearn.dummy import DummyClassifier

        self.name, self.model_id, self._m = name, name.lower().replace(" ", "_"), DummyClassifier(strategy=strategy)

    def fit(self, X, y):
        self._m.fit(np.zeros((len(X), 1)), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X):
        return self._m.predict_proba(np.zeros((len(X), 1)))[:, 1]


def stage_run(ctx) -> None:
    """Nested CV with the configured protocol; save OOF and per-fold evidence/models/importances."""
    data, split, feats, cats = ctx["data"], ctx["split"], ctx["feats"], ctx["cats"]
    cfg = build_config(CONFIG_NAME)
    t0 = time.perf_counter()
    run = run_protocol(data, split, feats, cfg)
    run.oof.to_csv(OUT / "oof_catboost_final.csv", index=False)
    run.fold_metrics.to_csv(OUT / "fold_metrics_catboost_final.csv", index=False)
    rows, evidence, pvc, perm = [], [], {}, {}
    rng = np.random.default_rng(SEED)
    from sklearn.metrics import average_precision_score

    for k, (adapter, meta) in enumerate(zip(run.adapters, run.fold_metadata), start=1):
        inner = meta["adapter_metadata"]["inner_evaluation"]
        rows.append({"fold": k, "candidate_id": adapter.candidate_id, "selected_trees": adapter.selected_tree_count_,
                     "selection_metric": adapter.selection_metric, "n_inner_splits": inner.get("n_inner_splits", 1),
                     **{p: adapter.params.get(p) for p in ("depth", "learning_rate", "l2_leaf_reg", "auto_class_weights", "boost_from_average", "eval_metric")}})
        cands = inner.get("all_candidates", {})
        evidence.append({"fold": k, "selected": adapter.candidate_id, "candidates": {
            cid: {**{kk: vv for kk, vv in c.items() if kk != "evals_result"},
                  "curves": {m: v for m, v in c.get("evals_result", {}).get("validation", {}).items()}} for cid, c in cands.items()}})
        pvc[f"fold_{k}"] = pd.Series(adapter.get_feature_importance(), index=feats)
        va = data[data.cv_fold == k]
        Xv = va[feats].copy()
        base = average_precision_score(va.Resigned, adapter.predict_proba(Xv))
        drops = {}
        for col in feats:
            ds = []
            for _ in range(3):
                Xs = Xv.copy()
                Xs[col] = rng.permutation(Xs[col].to_numpy())
                ds.append(base - average_precision_score(va.Resigned, adapter.predict_proba(Xs)))
            drops[col] = float(np.mean(ds))
        perm[f"fold_{k}"] = pd.Series(drops)
    pd.DataFrame(rows).to_csv(OUT / "fold_selection_summary.csv", index=False)
    jdump(evidence, OUT / "inner_evidence_final.json")
    pd.DataFrame(pvc).to_csv(OUT / "importance_prediction_values_change.csv")
    pd.DataFrame(perm).to_csv(OUT / "importance_permutation_oof.csv")
    # Final development model: same protocol on all 80,000 development rows (new file; older model untouched)
    dev = CatBoostAdapter(config=cfg, candidate_id="auto")
    X_all = data[feats]
    dev.fit(X_all, data.Resigned.to_numpy())
    dev.save(MODELS_OUT)
    jdump({"fit_seconds_total": time.perf_counter() - t0, "config_candidates": [c["id"] for c in cfg["candidates"]],
           "inner_validation": cfg["inner_validation"], "fixed_params": cfg["fixed_params"],
           "final_dev_model_trees": dev.selected_tree_count_, "final_dev_model_candidate": dev.candidate_id,
           "catboost_task_type": dev.params.get("task_type"), "seed": SEED}, OUT / "run_metadata.json")
    print(f"  nested CV + final model in {time.perf_counter() - t0:.0f}s")


def stage_baselines(ctx) -> None:
    data, split, feats, cats = ctx["data"], ctx["split"], ctx["feats"], ctx["cats"]
    frame = data.drop(columns=["cv_fold"])
    specs = {"dummy_prior": lambda: DummyAdapter("prior", "Dummy Prior"),
             "dummy_most_frequent": lambda: DummyAdapter("most_frequent", "Dummy Most Frequent"),
             "logistic_regression": lambda: LogisticRegressionAdapter(name="Logistic Regression", categorical_features=cats)}
    for key, factory in specs.items():
        res = run_5fold_cv(factory, frame, split, feats)
        res.predictions.to_csv(OUT / f"oof_{key}.csv", index=False)
        print(f"  {key}: pooled AP={metric_row(res.predictions.y_true, res.predictions.y_probability)['pr_auc']:.5f}")


def _load_oofs() -> dict[str, pd.DataFrame]:
    """Row-aligned OOF frames keyed by display name; alignment on Employee_ID is asserted."""
    final = pd.read_csv(OUT / "oof_catboost_final.csv").sort_values("Employee_ID").reset_index(drop=True)
    frames = {
        "Dummy prior": pd.read_csv(OUT / "oof_dummy_prior.csv"),
        "Dummy most_frequent": pd.read_csv(OUT / "oof_dummy_most_frequent.csv"),
        "Logistic Regression": pd.read_csv(OUT / "oof_logistic_regression.csv"),
        "CatBoost (baseline raw)": pd.read_csv(CB / "catboost_oof_predictions.csv"),
        "CatBoost (raw)": final,
    }
    for method in ("platt", "isotonic"):
        f = final.copy()
        f["y_probability"] = crossfit_calibration(final, method)
        frames[f"CatBoost ({method})"] = f
    out = {}
    for name, f in frames.items():
        f = f.sort_values("Employee_ID").reset_index(drop=True)
        assert (f.Employee_ID.to_numpy() == final.Employee_ID.to_numpy()).all(), f"{name}: Employee_ID order mismatch"
        assert (f.y_true.to_numpy() == final.y_true.to_numpy()).all(), f"{name}: label mismatch after join"
        assert (f.cv_fold.to_numpy() == final.cv_fold.to_numpy()).all(), f"{name}: fold mismatch"
        out[name] = f
    return out


def stage_metrics(ctx) -> None:
    oofs = _load_oofs()
    ref = "Dummy prior"
    summary, paired, topk, fold_tab, boot = compare_models(oofs, ref, N_BOOT, SEED, MIN_EFFECT, DEFAULT_KS)
    pd.concat(fold_tab, names=["model"]).reset_index(level=0).to_csv(OUT / "fold_metrics_all_models.csv", index=False)
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    topk.to_csv(OUT / "topk_all_models.csv", index=False)
    paired.to_csv(OUT / "paired_vs_dummy_prior.csv", index=False)
    y = oofs["CatBoost (raw)"].y_true.to_numpy()
    d_pr = boot["CatBoost (raw)"]["pr_auc"] - boot["CatBoost (baseline raw)"]["pr_auc"]
    d_roc = boot["CatBoost (raw)"]["roc_auc"] - boot["CatBoost (baseline raw)"]["roc_auc"]
    cal = summary[summary.model.isin(["CatBoost (raw)", "CatBoost (platt)", "CatBoost (isotonic)"])][
        ["model", "pooled_brier", "pooled_log_loss", "pooled_pr_auc", "pooled_roc_auc"]]
    cal.to_csv(OUT / "calibration_comparison.csv", index=False)
    chosen = cal.sort_values(["pooled_log_loss", "pooled_brier"]).iloc[0].model
    np.save(OUT / "boot_delta_pr_vs_prior.npy", boot["CatBoost (raw)"]["pr_auc"] - boot[ref]["pr_auc"])
    np.save(OUT / "boot_delta_roc_vs_prior.npy", boot["CatBoost (raw)"]["roc_auc"] - boot[ref]["roc_auc"])
    np.save(OUT / "boot_delta_pr_vs_baseline_catboost.npy", d_pr)
    jdump({"calibration_choice_rule": "lowest cross-fitted pooled log loss, then Brier",
           "calibration_choice": chosen,
           "practical_effect_threshold": MIN_EFFECT,
           "final_raw_vs_baseline_raw": {"delta_pr_auc_ci": ci95(d_pr), "delta_roc_auc_ci": ci95(d_roc)},
           "n_bootstrap": N_BOOT, "bootstrap_seed": SEED,
           "verdict_rule": "better only if the 95% paired-bootstrap lower bound of delta PR-AUC and delta ROC-AUC vs Dummy prior is > 0 "
                           "AND both point differences >= threshold"}, OUT / "verdict.json")
    # reconciliation: recompute from the CSV on disk
    disk = pd.read_csv(OUT / "oof_catboost_final.csv")
    again = metric_row(disk.y_true, disk.y_probability)
    row = summary[summary.model == "CatBoost (raw)"].iloc[0]
    for k in ("pr_auc", "roc_auc", "brier", "log_loss"):
        assert abs(again[k] - row[f"pooled_{k}"]) < 1e-5, f"reconciliation failed for {k}"
    print("  metrics reconciled to 1e-5 against the saved OOF CSV")


def stage_diagnostics(ctx) -> None:
    from catboost import CatBoostClassifier

    data, feats, cats = ctx["data"], ctx["feats"], ctx["cats"]
    sel = pd.read_csv(OUT / "fold_selection_summary.csv")
    cfg = build_config(CONFIG_NAME)
    pool = {c["id"]: c["params"] for c in cfg["candidates"]}
    rows = []
    grid = [1, 2, 5, 10, 25, 50, 100, 200, 400]
    probs = {n: np.zeros(len(data)) for n in grid}
    from src.evaluation.audit_stats import metric_row as mrow

    data = data.reset_index(drop=True)
    for _, r in sel.iterrows():
        tr, va = data[data.cv_fold != r.fold], data[data.cv_fold == r.fold]
        Xt, Xv = tr[feats].copy(), va[feats].copy()
        for c in cats:
            Xt[c], Xv[c] = Xt[c].astype(str), Xv[c].astype(str)
        ci = [feats.index(c) for c in cats]
        # Descriptive ablation (outer-val used only to draw the curve, never to select): winner's FULL parameters, fixed tree grid
        kw = {**cfg["fixed_params"], **pool[r.candidate_id], "iterations": max(grid), "random_seed": SEED,
              "verbose": False, "allow_writing_files": False}
        kw.pop("early_stopping_rounds", None)
        m = CatBoostClassifier(**kw, cat_features=ci).fit(Xt, tr.Resigned)
        for n in grid:
            probs[n][va.index.to_numpy()] = m.predict_proba(Xv, ntree_end=n)[:, 1]
    for n in grid:
        rows.append({"trees": n, **mrow(data.Resigned.to_numpy(), probs[n])})
    pd.DataFrame(rows).to_csv(OUT / "tree_count_ablation.csv", index=False)
    full, _, _, _ = load_audit_inputs(PROJECT_ROOT, all_columns=True)
    scan = univariate_scan(full, [c for c in full.columns if c not in ("Resigned", "cv_fold")], cats)
    scan.to_csv(OUT / "univariate_scan.csv", index=False)
    oof = pd.read_csv(OUT / "oof_catboost_final.csv").merge(data[["Employee_ID", "Department", "Gender", "Education_Level"]], on="Employee_ID")
    for col in ("Department", "Gender", "Education_Level"):
        subgroup_classification_metrics(oof, col).to_csv(OUT / f"subgroup_{col}.csv", index=False)
    # SHAP only if the final model has non-trivial trees
    meta = json.loads((MODELS_OUT / "metadata.json").read_text(encoding="utf-8"))
    shap_info = {"final_tree_count": meta["selected_tree_count"], "computed": False}
    if meta["selected_tree_count"] >= 10:
        from catboost import Pool

        model = CatBoostClassifier()
        model.load_model(str(MODELS_OUT / "model.cbm"))
        samp = data.sample(5000, random_state=SEED)
        Xs = samp[feats].copy()
        for c in cats:
            Xs[c] = Xs[c].astype(str)
        sv = model.get_feature_importance(Pool(Xs, cat_features=cats), type="ShapValues")[:, :-1]
        np.save(OUT / "shap_values.npy", sv)
        Xs.to_csv(OUT / "shap_sample_features.csv", index=False)
        shap_info["computed"] = True
    jdump(shap_info, OUT / "shap_info.json")


def _perm_worker(i: int, threads: int) -> dict:
    """One label permutation through the COMPLETE nested protocol (own process, own data load)."""
    seed_everything()
    data, split, feats, _ = load_audit_inputs(PROJECT_ROOT)
    cfg = build_config(CONFIG_NAME, fixed_overrides={"thread_count": threads})
    rng = np.random.default_rng(SEED + 1000 + i)
    t0 = time.perf_counter()
    run = run_protocol(data, split, feats, cfg, target_override=rng.permutation(data.Resigned.to_numpy()))
    m = metric_row(run.oof.y_true, run.oof.y_probability)
    return {"perm": i, "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "seconds": time.perf_counter() - t0, "threads": threads}


def stage_permutation(ctx, n_perm: int, workers: int = 1, threads: int = -1) -> None:
    """Label-shuffle test: permute development labels, rerun the full protocol, collect the null.

    Resumable (finished permutations in permutation_null.csv are skipped). With workers > 1 the
    permutations run in parallel processes; thread count changes speed only, not the protocol.
    """
    if n_perm <= 0:
        return
    path = OUT / "permutation_null.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["perm", "pr_auc", "roc_auc", "seconds", "threads"])
    todo = [i for i in range(n_perm) if i not in set(done.perm)]

    def record(row):
        nonlocal done
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(path, index=False)
        print(f"  permutation {row['perm']}: AP={row['pr_auc']:.5f} AUC={row['roc_auc']:.5f} ({row['seconds']:.0f}s) [{len(done)}/{n_perm}]", flush=True)

    if workers <= 1:
        for i in todo:
            record(_perm_worker(i, threads))
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_perm_worker, i, threads if threads > 0 else 1) for i in todo]
        for fut in as_completed(futures):
            record(fut.result())


def stage_figures(ctx) -> None:
    data, feats = ctx["data"], ctx["feats"]
    oofs = _load_oofs()
    y = oofs["CatBoost (raw)"].y_true.to_numpy()
    summ = pd.read_csv(OUT / "metrics_summary.csv").set_index("model")
    verdict = json.loads((OUT / "verdict.json").read_text(encoding="utf-8"))
    chosen = verdict["calibration_choice"]
    cfg = build_config(CONFIG_NAME)
    made = {}
    made["fig_01_target_and_fold_balance"] = plots.fig_target_distribution(FIG / "fig_01_target_and_fold_balance.png", data)
    n_dev = len(data)
    made["fig_02_nested_cv_layout"] = plots.fig_nested_cv_diagram(FIG / "fig_02_nested_cv_layout.png", n_dev, 20000, int(data.groupby("cv_fold").size().iloc[0]), cfg["inner_validation"]["fraction"])
    show = {"CatBoost (final)": oofs[chosen if chosen != "CatBoost (raw)" else "CatBoost (raw)"], "CatBoost (baseline raw)": oofs["CatBoost (baseline raw)"],
            "Dummy prior": oofs["Dummy prior"], "Logistic Regression": oofs["Logistic Regression"]}
    pr = {k: v.y_probability.to_numpy() for k, v in show.items()}
    src = {"CatBoost (final)": chosen, "CatBoost (baseline raw)": "CatBoost (baseline raw)", "Dummy prior": "Dummy prior", "Logistic Regression": "Logistic Regression"}
    aucs = {k: (summ.loc[src[k], "pooled_roc_auc"], summ.loc[src[k], "roc_auc_ci_lower"], summ.loc[src[k], "roc_auc_ci_upper"]) for k in show}
    aps = {k: (summ.loc[src[k], "pooled_pr_auc"], summ.loc[src[k], "pr_auc_ci_lower"], summ.loc[src[k], "pr_auc_ci_upper"]) for k in show}
    made["fig_03_roc_curves"] = plots.fig_roc(FIG / "fig_03_roc_curves.png", y, pr, aucs)
    made["fig_04_pr_curves"] = plots.fig_pr(FIG / "fig_04_pr_curves.png", y, pr, aps)
    ft = {n: per_fold_metrics(oofs[src[n]]) for n in show}
    made["fig_05_per_fold_metrics"] = plots.fig_fold_strip(FIG / "fig_05_per_fold_metrics.png", ft, float(y.mean()))
    made["fig_06_bootstrap_delta_vs_prior"] = plots.fig_bootstrap_delta(
        FIG / "fig_06_bootstrap_delta_vs_prior.png",
        {"ΔPR-AUC vs Dummy prior (CatBoost raw)": np.load(OUT / "boot_delta_pr_vs_prior.npy"),
         "ΔROC-AUC vs Dummy prior (CatBoost raw)": np.load(OUT / "boot_delta_roc_vs_prior.npy")},
        f"Paired bootstrap ({verdict['n_bootstrap']} resamples, seed {verdict['bootstrap_seed']})")
    ev = json.loads((OUT / "inner_evidence_final.json").read_text(encoding="utf-8"))
    f1 = ev[0]
    curves = {cid: {m: v for m, v in c["curves"].items() if m in ("Logloss", "PRAUC", "AUC")} for cid, c in f1["candidates"].items()}
    sel_trees = {cid: c.get("selected_tree_count") for cid, c in f1["candidates"].items()}
    # the plotted metric names depend on eval_metric; Logloss is always in the loss curve
    made["fig_07_inner_learning_curves"] = plots.fig_learning_curves(FIG / "fig_07_inner_learning_curves.png", curves, sel_trees, "outer fold 1")
    cand_rows = [{"fold": e["fold"], "candidate_id": cid, "inner_ap": c["inner_ap"], "inner_log_loss": c.get("inner_log_loss", np.nan)}
                 for e in ev for cid, c in e["candidates"].items()]
    cand = pd.DataFrame(cand_rows)
    made["fig_08_candidate_comparison"] = plots.fig_candidate_comparison(FIG / "fig_08_candidate_comparison.png", cand, "inner_ap", "inner-validation average precision", float(y.mean()), "prevalence")
    abl = pd.read_csv(OUT / "tree_count_ablation.csv")
    made["fig_09_tree_count_ablation"] = plots.fig_tree_ablation(FIG / "fig_09_tree_count_ablation.png", abl, float(y.mean()), float(summ.loc["Dummy prior", "pooled_log_loss"]))
    made["fig_10_reliability"] = plots.fig_reliability(FIG / "fig_10_reliability.png", y, {"CatBoost (raw)": oofs["CatBoost (raw)"].y_probability.to_numpy(), "CatBoost (calibrated)": oofs[chosen].y_probability.to_numpy()})
    made["fig_11_probability_by_class"] = plots.fig_prob_by_class(FIG / "fig_11_probability_by_class.png", y, oofs["CatBoost (raw)"].y_probability.to_numpy())
    tk = pd.read_csv(OUT / "topk_all_models.csv")
    tb = {n: tk[tk.model == src[n]].reset_index(drop=True) for n in show}
    made["fig_12_gain_lift"] = plots.fig_gain_lift(FIG / "fig_12_gain_lift.png", y, pr, tb)
    pvc, perm = pd.read_csv(OUT / "importance_prediction_values_change.csv", index_col=0), pd.read_csv(OUT / "importance_permutation_oof.csv", index_col=0)
    made["fig_13_feature_importance"] = plots.fig_importance(FIG / "fig_13_feature_importance.png", pvc, perm)
    if (OUT / "shap_values.npy").exists():
        Xs = pd.read_csv(OUT / "shap_sample_features.csv")
        numeric = [c for c in feats if c not in ctx["cats"]]
        made["fig_14_shap_summary"] = plots.fig_shap(FIG / "fig_14_shap_summary.png", np.load(OUT / "shap_values.npy"), Xs, numeric)
    made["fig_15_single_feature_signal"] = plots.fig_single_feature_signal(FIG / "fig_15_single_feature_signal.png", pd.read_csv(OUT / "univariate_scan.csv").query("feature in @feats"))
    if (OUT / "permutation_null.csv").exists():
        null = pd.read_csv(OUT / "permutation_null.csv")
        real = float(summ.loc["CatBoost (raw)", "pooled_pr_auc"])
        pval = permutation_pvalue(real, null.pr_auc.to_numpy())
        made["fig_16_permutation_test"] = plots.fig_permutation_test(FIG / "fig_16_permutation_test.png", null.pr_auc.to_numpy(), real, "PR-AUC", pval)
    sub = {c: pd.read_csv(OUT / f"subgroup_{c}.csv") for c in ("Department", "Gender", "Education_Level")}
    made["fig_17_subgroups"] = plots.fig_subgroups(FIG / "fig_17_subgroups.png", sub)
    jdump({k: str(v.relative_to(PROJECT_ROOT)) for k, v in made.items()}, OUT / "figure_index.json")
    print(f"  wrote {len(made)} figures")


def stage_manifest(ctx) -> None:
    """Append (never delete) new artifacts to catboost_output_manifest.csv."""
    manifest_path = CB / "catboost_output_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    new = []
    for path in sorted(OUT.parent.rglob("*")):
        if path.is_file() and path.suffix in (".csv", ".json", ".png", ".md", ".npy"):
            new.append(path)
    for path in sorted(MODELS_OUT.glob("*")):
        new.append(path)
    rows = [{"artifact_name": p.name, "relative_path": str(p.relative_to(PROJECT_ROOT)), "size_bytes": p.stat().st_size, "exists": True} for p in new]
    add = pd.DataFrame(rows)
    add = add[~add.relative_path.isin(manifest.relative_path)]
    pd.concat([manifest, add], ignore_index=True).to_csv(manifest_path, index=False)
    print(f"  manifest: +{len(add)} rows (existing rows untouched)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=["run", "baselines", "metrics", "diagnostics", "permutation", "figures", "manifest"])
    ap.add_argument("--n-permutations", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes for the permutation test")
    ap.add_argument("--threads-per-worker", type=int, default=-1, help="CatBoost threads per permutation worker (-1 = all; 1 when parallel)")
    args = ap.parse_args()
    seed_everything()
    OUT.mkdir(parents=True, exist_ok=True)
    data, split, feats, cats = load_audit_inputs(PROJECT_ROOT)
    assert len(data) == 80_000 and set(split.partition) == {"development", "holdout"}
    assert (data.groupby("cv_fold").size() == 16_000).all(), "outer folds must hold exactly 16,000 rows"
    assert not ({"Employee_ID", "Hire_Date", "Resigned", "cv_fold", "partition"} & set(feats)), "excluded column in feature list"
    ctx = {"data": data, "split": split, "feats": feats, "cats": cats}
    for stage in args.stages:
        print(f"== {stage}")
        if stage == "permutation":
            stage_permutation(ctx, args.n_permutations, args.workers, args.threads_per_worker)
        else:
            {"run": stage_run, "baselines": stage_baselines, "metrics": stage_metrics, "diagnostics": stage_diagnostics,
             "figures": stage_figures, "manifest": stage_manifest}[stage](ctx)


if __name__ == "__main__":
    main()
