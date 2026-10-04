"""Tests for the CatBoost audit toolkit: leakage guards, baselines, reconciliation, determinism.

No expected metric is typed in: every number is recomputed from arrays built in the test
or from the saved OOF files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import audit_stats as st
from src.models.catboost_model import CatBoostAdapter

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/models/catboost/improvement/final"


def _toy(n=600, seed=0, signal=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["x", "y", "z"], n)})
    y = (rng.random(n) < 0.12).astype(int)
    if signal:
        X["a"] = X["a"] + 3 * y
    return X, y


# ----------------------------------------------------------------------------- metrics / baselines
def test_topk_exact_capacity_and_constant_scores_have_lift_one():
    y = np.r_[np.ones(100), np.zeros(900)].astype(int)
    table = st.topk_table(y, np.full(1000, 0.1), ks=(0.05, 0.10, 0.50))
    assert list(table.n_selected) == [50, 100, 500]  # ties must not inflate capacity
    assert np.allclose(table.lift, 1.0, atol=0.45)  # random tie-break: near 1, never systematically above


def test_topk_perfect_ranker_has_maximal_recall():
    y = np.r_[np.ones(100), np.zeros(900)].astype(int)
    table = st.topk_table(y, y.astype(float), ks=(0.10,))
    assert table.recall.iloc[0] == 1.0


def test_paired_bootstrap_of_identical_models_is_exactly_zero():
    y = (np.random.default_rng(1).random(2000) < 0.1).astype(int)
    p = np.random.default_rng(2).random(2000)
    boot = st.bootstrap_pooled(y, {"a": p, "b": p.copy()}, n_boot=50, seed=3)
    for m in st.BOOT_METRICS:
        assert np.array_equal(boot["a"][m], boot["b"][m])


def test_bootstrap_is_seed_deterministic():
    y = (np.random.default_rng(1).random(1500) < 0.1).astype(int)
    p = np.random.default_rng(2).random(1500)
    a = st.bootstrap_pooled(y, {"m": p}, n_boot=20, seed=7)["m"]["pr_auc"]
    b = st.bootstrap_pooled(y, {"m": p}, n_boot=20, seed=7)["m"]["pr_auc"]
    assert np.array_equal(a, b)


def test_paired_fold_comparison_matches_scipy():
    from scipy import stats

    a, b = [0.11, 0.12, 0.10, 0.13, 0.12], [0.10, 0.10, 0.10, 0.10, 0.11]
    out = st.paired_fold_comparison(a, b)
    ref = stats.ttest_rel(a, b)
    assert out["p_value"] == pytest.approx(ref.pvalue)
    assert out["mean_diff"] == pytest.approx(np.mean(np.subtract(a, b)))


def test_verdict_requires_ci_and_practical_effect():
    assert st.better_than_baseline((0.02, 0.04), (0.02, 0.05), 0.03, 0.03)
    assert not st.better_than_baseline((-0.001, 0.04), (0.02, 0.05), 0.03, 0.03)  # CI includes 0
    assert not st.better_than_baseline((0.0005, 0.002), (0.0004, 0.002), 0.001, 0.001)  # real but trivial effect


def test_permutation_pvalue_is_add_one():
    assert st.permutation_pvalue(1.0, [0.1] * 9) == pytest.approx(1 / 10)
    assert st.permutation_pvalue(0.0, [0.1] * 9) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- leakage guards
def test_crossfit_calibration_never_uses_the_scored_fold():
    rng = np.random.default_rng(0)
    n = 3000
    folds = np.repeat(np.arange(1, 6), n // 5)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    oof = pd.DataFrame({"y_probability": p, "y_true": y, "cv_fold": folds})
    base = st.crossfit_calibration(oof, "isotonic")
    flipped = oof.copy()
    flipped.loc[flipped.cv_fold == 1, "y_true"] = 1 - flipped.loc[flipped.cv_fold == 1, "y_true"]
    after = st.crossfit_calibration(flipped, "isotonic")
    # Changing fold 1 labels must not change fold 1's own calibrated scores (it was not in its fit set)
    assert np.array_equal(base[folds == 1], after[folds == 1])
    assert not np.array_equal(base[folds == 2], after[folds == 2])


def test_univariate_scan_flags_planted_leak_but_not_noise():
    X, y = _toy(4000, signal=False)
    df = X.assign(Resigned=y, cv_fold=np.arange(len(y)) % 5 + 1)
    df["leak"] = y + np.random.default_rng(5).normal(scale=0.01, size=len(y))
    scan = st.univariate_scan(df, ["a", "b", "c", "leak"], ["c"]).set_index("feature")
    assert scan.loc["leak", "suspicious"]
    assert not scan.loc[["a", "b", "c"], "suspicious"].any()


def test_duplicate_audit_detects_cross_boundary_duplicate():
    X, y = _toy(200)
    df = X.assign(Resigned=y, cv_fold=np.arange(200) % 5 + 1)
    df.loc[3, ["a", "b", "c"]] = df.loc[0, ["a", "b", "c"]].to_numpy()
    out = st.duplicate_audit(df, ["a", "b", "c"])
    assert out["duplicate_feature_rows"] == 1
    assert out["cross_boundary_duplicates_total"] >= 1


def test_audit_inputs_exclude_forbidden_columns_and_holdout():
    data, split, feats, cats = st.load_audit_inputs(ROOT)
    assert not ({"Employee_ID", "Hire_Date", "Resigned", "cv_fold", "partition"} & set(feats))
    holdout_ids = set(split.loc[split.partition == "holdout", "Employee_ID"])
    assert holdout_ids.isdisjoint(set(data.Employee_ID))
    assert (data.groupby("cv_fold").size() == data.groupby("cv_fold").size().iloc[0]).all()


# ----------------------------------------------------------------------------- adapter behaviour
def _adapter(**kw):
    params = {"iterations": 25, "depth": 3, "learning_rate": 0.2, "loss_function": "Logloss", "random_seed": 42,
              "verbose": False, "allow_writing_files": False, "thread_count": 1}
    return CatBoostAdapter(params=params, categorical_features=["c"], inner_val_fraction=0.2, early_stopping_rounds=5,
                           refit_outer_train=True, **kw)


def test_new_adapter_options_default_to_previous_behaviour():
    a = _adapter()
    assert a.selection_metric == "average_precision" and a.inner_seeds is None and a.final_seeds is None
    with pytest.raises(ValueError):
        _adapter(selection_metric="accuracy")


def test_same_seed_gives_identical_predictions():
    X, y = _toy(800, signal=True)
    p1 = _adapter().fit(X, y).predict_proba(X)
    p2 = _adapter().fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)


def test_seed_averaging_predicts_the_mean_of_member_models():
    X, y = _toy(800, signal=True)
    ens = _adapter(final_seeds=[1, 2, 3]).fit(X, y)
    assert len(ens._models) == 3
    X_str = X.assign(c=X["c"].astype(str))
    members = np.mean([m.predict_proba(X_str)[:, 1] for m in ens._models], axis=0)
    assert np.allclose(ens.predict_proba(X), members)


def test_repeated_inner_splits_are_recorded_and_use_only_training_rows():
    X, y = _toy(900, signal=True)
    cfg = {"model": {"name": "CatBoost Classifier", "model_id": "catboost"}, "reproducibility": {"random_seed": 42},
           "inner_validation": {"fraction": 0.2, "early_stopping_rounds": 5, "refit_outer_train": True,
                                "selection_metric": "log_loss", "repeat_seeds": [1, 2]},
           "fixed_params": {"loss_function": "Logloss", "verbose": False, "allow_writing_files": False, "thread_count": 1},
           "candidates": [{"id": "A", "params": {"iterations": 20, "depth": 2, "learning_rate": 0.2}},
                          {"id": "B", "params": {"iterations": 20, "depth": 3, "learning_rate": 0.2}}]}
    ad = CatBoostAdapter(config=cfg, candidate_id="auto", categorical_features=["c"])
    ad.fit(X, y)
    ev = ad.get_selection_evidence()
    assert ev["metric"] == "log_loss"
    assert all(c["n_inner_splits"] == 2 for c in ev["all_candidates"].values())
    assert ev["inner_train_rows"] + ev["inner_val_rows"] == len(X)  # inner split carved from the rows passed to fit()


# ----------------------------------------------------------------------------- saved-artifact reconciliation
@pytest.mark.skipif(not (FINAL / "metrics_summary.csv").exists(), reason="audit not run yet")
def test_saved_metrics_reconcile_with_saved_oof_to_1e5():
    summary = pd.read_csv(FINAL / "metrics_summary.csv").set_index("model")
    oof = pd.read_csv(FINAL / "oof_catboost_final.csv")
    again = st.metric_row(oof.y_true, oof.y_probability)
    for k in ("pr_auc", "roc_auc", "brier", "log_loss"):
        assert abs(again[k] - summary.loc["CatBoost (raw)", f"pooled_{k}"]) < 1e-5


@pytest.mark.skipif(not (FINAL / "oof_catboost_final.csv").exists(), reason="audit not run yet")
def test_final_oof_is_aligned_with_source_labels_by_employee_id():
    data, split, feats, cats = st.load_audit_inputs(ROOT)
    oof = pd.read_csv(FINAL / "oof_catboost_final.csv").merge(data[["Employee_ID", "Resigned", "cv_fold"]], on="Employee_ID", suffixes=("", "_src"), validate="one_to_one")
    assert len(oof) == len(data)
    assert (oof.y_true == oof.Resigned).all() and (oof.cv_fold == oof.cv_fold_src).all()


@pytest.mark.skipif(not (FINAL / "oof_catboost_final.csv").exists(), reason="audit not run yet")
def test_probability_scale_is_consistent_across_folds_for_the_final_protocol():
    """Regression guard for the 0.5-start artefact: fold mean probability must sit near prevalence."""
    oof = pd.read_csv(FINAL / "oof_catboost_final.csv")
    means = oof.groupby("cv_fold").y_probability.mean()
    assert (means - oof.y_true.mean()).abs().max() < 0.02


@pytest.mark.skipif(not (FINAL / "figure_index.json").exists(), reason="audit not run yet")
def test_figures_exist_and_are_registered_in_manifest():
    manifest = pd.read_csv(ROOT / "results/models/catboost/catboost_output_manifest.csv")
    registered = set(manifest.relative_path.str.replace("\\", "/", regex=False))
    for rel in json.loads((FINAL / "figure_index.json").read_text(encoding="utf-8")).values():
        assert (ROOT / rel).is_file()
        assert rel.replace("\\", "/") in registered


def test_manifest_has_no_missing_files():
    manifest = pd.read_csv(ROOT / "results/models/catboost/catboost_output_manifest.csv")
    missing = [r for r in manifest.relative_path if not (ROOT / r.replace("\\", "/")).exists()]
    assert not missing, missing


# ----------------------------------------------------------------------------- audited config and experiment log
def test_previous_canonical_catboost_config_is_unchanged():
    from src.models.config import load_model_config

    cfg = load_model_config("catboost", strict_schema=True)
    assert [c["id"] for c in cfg["candidates"]] == [f"C0{i}" for i in range(1, 9)]
    assert cfg["inner_validation"]["early_stopping_rounds"] == 50  # the audited protocol lives in audited_catboost.yaml


def test_audited_config_is_valid_and_matches_the_kept_experiment_state():
    import importlib.util
    from src.models.config import load_model_config

    audited = load_model_config("audited_catboost", strict_schema=True)
    state_path = ROOT / "results/models/catboost/improvement/experiment_state.json"
    if not state_path.exists():
        pytest.skip("experiments not run")
    spec = importlib.util.spec_from_file_location("exp", ROOT / "scripts/run_catboost_experiments.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    expected = exp.spec_to_config(json.loads(state_path.read_text(encoding="utf-8"))["spec"])
    for key in ("candidates", "inner_validation", "fixed_params"):
        assert audited[key] == expected[key]
    assert audited["audit"]["kept_experiments"] == json.loads(state_path.read_text(encoding="utf-8"))["kept"]


def test_experiment_log_decisions_follow_the_pre_declared_rule():
    log_path = ROOT / "results/models/catboost/improvement/experiment_log.csv"
    if not log_path.exists():
        pytest.skip("experiments not run")
    log = pd.read_csv(log_path)
    for r in log[log.kind == "perf"].itertuples():
        lo_pr = json.loads(r.delta_pr_auc_ci95)[0]
        ll_hi = json.loads(r.delta_log_loss_ci95)[1]
        should_keep = lo_pr > 0 or ll_hi < 0
        assert (r.decision == "kept") == should_keep, r.experiment
    for r in log[log.kind == "fix"].itertuples():
        scale_ok = (abs(r.fold_mean_prob_min - r.prevalence) < 0.02) and (abs(r.fold_mean_prob_max - r.prevalence) < 0.02)
        lo_pr = json.loads(r.delta_pr_auc_ci95)[0]
        assert (r.decision == "kept") == (scale_ok and lo_pr > -0.005), r.experiment


def test_experiment_spec_merge_does_not_mutate_the_input():
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp", ROOT / "scripts/run_catboost_experiments.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    base = {"fixed_overrides": {"a": 1}, "inner_overrides": {}, "candidates": None, "ensemble": None}
    merged = exp.merge(base, {"fixed_overrides": {"b": 2}})
    assert base["fixed_overrides"] == {"a": 1} and merged["fixed_overrides"] == {"a": 1, "b": 2}
