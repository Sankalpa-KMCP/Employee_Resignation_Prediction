"""Phase 3: run pre-declared CatBoost improvement experiments one at a time.

The plan (``results/models/catboost/improvement/improvement_plan.md``) is written
before this script is used. Each experiment is a nested-CV run through the
leakage-safe runner, then judged against the CURRENT BEST with pre-declared rules:

* kind="fix"  - kept only if probability-scale checks pass and PR-AUC is not
                materially worse (paired-bootstrap lower bound > -0.005).
* kind="perf" - kept only if the paired-bootstrap 95% CI lower bound of
                delta PR-AUC is > 0; otherwise a significantly lower Log Loss
                breaks the tie; anything else is reverted.

State lives in ``experiment_state.json``; every run appends to ``experiment_log.csv``.
Usage: python scripts/run_catboost_experiments.py E1 [E2 ...]
"""

from __future__ import annotations

import argparse
import copy
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
    bootstrap_pooled, ci95, load_audit_inputs, metric_row, paired_fold_comparison, per_fold_metrics,
)
from src.models.catboost_protocol import build_config, run_protocol  # noqa: E402
from src.models.config import load_model_config  # noqa: E402

OUT = PROJECT_ROOT / "results" / "models" / "catboost" / "improvement"
STATE = OUT / "experiment_state.json"
LOG = OUT / "experiment_log.csv"
N_BOOT = 1000
SEED = 42
MAX_NO_GAIN = 2  # stopping rule: two consecutive justified changes without significant gain


def shapes(weight: dict) -> list[dict]:
    """Three bounded tree shapes (taken from the existing candidate pool) with one weighting strategy."""
    base = [("S1", dict(depth=6, learning_rate=0.05, l2_leaf_reg=3)),
            ("S2", dict(depth=4, learning_rate=0.03, l2_leaf_reg=5)),
            ("S3", dict(depth=8, learning_rate=0.05, l2_leaf_reg=5))]
    return [{"id": cid, "params": {"iterations": 600, **p, **weight}} for cid, p in base]


# Each experiment is a delta on the cumulative spec of the current best.
EXPERIMENTS = {
    "E1": dict(kind="fix", desc="boost_from_average=True (start Logloss at the prior, not p=0.5)",
               delta={"fixed_overrides": {"boost_from_average": True}}),
    "E1b": dict(kind="fix", desc="boost_from_average=True re-tested on top of the current best (plan Amendment 1)",
                delta={"fixed_overrides": {"boost_from_average": True}}),
    "E2": dict(kind="perf", desc="early stopping + candidate selection on inner Log Loss, patience 200",
               delta={"fixed_overrides": {"eval_metric": "Logloss"},
                      "inner_overrides": {"selection_metric": "log_loss", "early_stopping_rounds": 200}}),
    "E3a": dict(kind="perf", desc="class weights: none (3 shapes)", delta={"candidates": shapes({"auto_class_weights": None})}),
    "E3b": dict(kind="perf", desc="class weights: Balanced (3 shapes)", delta={"candidates": shapes({"auto_class_weights": "Balanced"})}),
    "E3c": dict(kind="perf", desc="class weights: SqrtBalanced (3 shapes)", delta={"candidates": shapes({"auto_class_weights": "SqrtBalanced"})}),
    "E3d": dict(kind="perf", desc="class weights: scale_pos_weight=3 (3 shapes)", delta={"candidates": shapes({"scale_pos_weight": 3.0})}),
    "E4": dict(kind="perf", desc="random search, 20 trials, 3 repeated inner splits",
               delta={"candidates": "RANDOM40", "inner_overrides": {"repeat_seeds": [42, 43, 44]}}),
    "E5": dict(kind="perf", desc="seed averaging, 5 final seeds", delta={"ensemble": {"final_seeds": [42, 43, 44, 45, 46]}}),
}


def random_pool(n: int = 20, seed: int = SEED) -> list[dict]:
    """Bounded random search space (inner split only): depth 2-8, l2 1-30, strength, bagging, borders."""
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(n):
        pool.append({"id": f"R{i + 1:02d}", "params": {
            "iterations": 600, "depth": int(rng.integers(2, 9)),
            "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
            "l2_leaf_reg": float(np.round(np.exp(rng.uniform(np.log(1), np.log(30))), 2)),
            "random_strength": float(np.round(rng.uniform(0.2, 3.0), 2)),
            "bagging_temperature": float(np.round(rng.uniform(0, 2), 2)),
            "border_count": int(rng.choice([32, 64, 128, 254])),
            "one_hot_max_size": int(rng.choice([2, 4, 10])),
            "rsm": float(rng.choice([0.5, 0.8, 1.0])),
            "auto_class_weights": None}})
    return pool


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"spec": {"fixed_overrides": {}, "inner_overrides": {}, "candidates": None, "ensemble": None},
            "best_oof": "oof_E0_baseline.csv", "no_gain_streak": 0, "kept": [], "stopped": False}


def spec_to_config(spec: dict) -> dict:
    cands = spec.get("candidates")
    if cands == "RANDOM40":
        cands = random_pool()
    return build_config(candidates=cands, fixed_overrides=spec.get("fixed_overrides"),
                        inner_overrides=spec.get("inner_overrides"), ensemble=spec.get("ensemble"))


def merge(spec: dict, delta: dict) -> dict:
    out = copy.deepcopy(spec)
    for k, v in delta.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def evaluate(exp_id: str, spec: dict, run, ref: pd.DataFrame, prevalence: float, kind: str, ref_name: str) -> dict:
    new = run.oof.sort_values("Employee_ID").reset_index(drop=True)
    ref = ref.sort_values("Employee_ID").reset_index(drop=True)
    assert (new.Employee_ID.to_numpy() == ref.Employee_ID.to_numpy()).all() and (new.y_true.to_numpy() == ref.y_true.to_numpy()).all()
    y = new.y_true.to_numpy()
    boot = bootstrap_pooled(y, {"new": new.y_probability.to_numpy(), "ref": ref.y_probability.to_numpy()}, N_BOOT, SEED)
    d = {m: boot["new"][m] - boot["ref"][m] for m in ("pr_auc", "roc_auc", "brier", "log_loss")}
    m_new, m_ref = metric_row(y, new.y_probability), metric_row(y, ref.y_probability)
    fm_new, fm_ref = per_fold_metrics(new), per_fold_metrics(ref)
    paired = paired_fold_comparison(fm_new.pr_auc, fm_ref.pr_auc)
    fold_means = new.groupby("cv_fold").y_probability.mean()
    row = {"experiment": exp_id, "reference": ref_name, "kind": kind,
           "pr_auc": m_new["pr_auc"], "roc_auc": m_new["roc_auc"], "brier": m_new["brier"], "log_loss": m_new["log_loss"],
           "pr_auc_ci95": json.dumps([round(x, 6) for x in ci95(boot["new"]["pr_auc"])]),
           "roc_auc_ci95": json.dumps([round(x, 6) for x in ci95(boot["new"]["roc_auc"])]),
           "brier_ci95": json.dumps([round(x, 6) for x in ci95(boot["new"]["brier"])]),
           "log_loss_ci95": json.dumps([round(x, 6) for x in ci95(boot["new"]["log_loss"])]),
           "delta_pr_auc": m_new["pr_auc"] - m_ref["pr_auc"], "delta_pr_auc_ci95": json.dumps([round(x, 6) for x in ci95(d["pr_auc"])]),
           "delta_log_loss": m_new["log_loss"] - m_ref["log_loss"], "delta_log_loss_ci95": json.dumps([round(x, 6) for x in ci95(d["log_loss"])]),
           "delta_brier": m_new["brier"] - m_ref["brier"],
           "fold_paired_mean_diff_pr_auc": paired["mean_diff"], "fold_paired_p_value": paired["p_value"],
           "fold_mean_prob_min": float(fold_means.min()), "fold_mean_prob_max": float(fold_means.max()),
           "prevalence": prevalence, "runtime_s": run.fit_seconds,
           "trees_by_fold": json.dumps([a.selected_tree_count_ for a in run.adapters]),
           "winners_by_fold": json.dumps([a.candidate_id for a in run.adapters]),
           "config": json.dumps(spec, default=str)}
    lo_pr = ci95(d["pr_auc"])[0]
    lo_ll_hi = ci95(d["log_loss"])[1]
    if kind == "fix":
        scale_ok = bool(np.all(np.abs(fold_means - prevalence) < 0.02))
        row["decision"] = "kept" if (scale_ok and lo_pr > -0.005) else "reverted"
        row["decision_reason"] = f"scale_ok={scale_ok}; delta_pr_lower={lo_pr:.5f} (> -0.005 required)"
    else:
        if lo_pr > 0:
            row["decision"], row["decision_reason"] = "kept", f"delta PR-AUC CI lower bound {lo_pr:.5f} > 0"
        elif lo_ll_hi < 0:
            row["decision"], row["decision_reason"] = "kept", "PR-AUC tie; Log Loss significantly lower (CI upper < 0)"
        else:
            row["decision"], row["decision_reason"] = "reverted", f"no significant gain: delta PR-AUC CI lower {lo_pr:.5f} <= 0"
    return row


def write_final_config() -> Path:
    """Write the cumulative kept spec as ``configs/models/audited_catboost.yaml`` (old config untouched)."""
    import yaml

    state = load_state()
    cfg = spec_to_config(state["spec"])
    cfg.pop("raw_config", None)
    cfg["model"]["config_version"] = "2.1.0-audited"
    cfg["audit"] = {"source": "scripts/run_catboost_experiments.py", "kept_experiments": state["kept"],
                    "experiment_log": "results/models/catboost/improvement/experiment_log.csv"}
    path = PROJECT_ROOT / "configs" / "models" / "audited_catboost.yaml"
    header = ("# Audited CatBoost protocol: cumulative kept changes from the improvement plan.\n"
              "# Generated by `python scripts/run_catboost_experiments.py --write-config`; do not edit by hand.\n"
              "# configs/models/catboost.yaml (previous canonical protocol) is intentionally left unchanged.\n")
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*")
    ap.add_argument("--write-config", action="store_true")
    args = ap.parse_args()
    if args.write_config:
        print(f"wrote {write_final_config()}")
        return
    data, split, feats, _ = load_audit_inputs(PROJECT_ROOT)
    prevalence = float(data.Resigned.mean())
    state = load_state()
    for exp_id in args.experiments:
        if state["stopped"] and exp_id not in ("E0_baseline",):
            print(f"stopping rule already triggered; refusing to run {exp_id}")
            continue
        if exp_id == "E0_baseline":
            spec, kind, desc, ref_name = state["spec"], "baseline", "current repo protocol", "none"
        else:
            e = EXPERIMENTS[exp_id]
            spec, kind, desc, ref_name = merge(state["spec"], e["delta"]), e["kind"], e["desc"], state["best_oof"]
        print(f"== {exp_id}: {desc}")
        t0 = time.perf_counter()
        run = run_protocol(data, split, feats, spec_to_config(spec))
        wall = time.perf_counter() - t0
        run.oof.to_csv(OUT / f"oof_{exp_id}.csv", index=False)
        if exp_id == "E0_baseline":
            m = metric_row(run.oof.y_true, run.oof.y_probability)
            row = {"experiment": exp_id, "kind": kind, **m, "decision": "baseline", "runtime_s": run.fit_seconds,
                   "trees_by_fold": json.dumps([a.selected_tree_count_ for a in run.adapters]),
                   "winners_by_fold": json.dumps([a.candidate_id for a in run.adapters]), "config": json.dumps(spec, default=str)}
            state["best_oof"] = f"oof_{exp_id}.csv"
        else:
            ref = pd.read_csv(OUT / state["best_oof"])
            row = evaluate(exp_id, spec, run, ref, prevalence, kind, ref_name)
        row["description"], row["wall_s"] = desc, wall
        # fold winners' per-candidate inner evidence for the figures
        evidence = [{"fold": i + 1, "all_candidates": {k: {kk: vv for kk, vv in v.items()} for k, v in
                     f["adapter_metadata"]["inner_evaluation"].get("all_candidates", {}).items()}}
                    for i, f in enumerate(run.fold_metadata)]
        (OUT / f"inner_evidence_{exp_id}.json").write_text(json.dumps(evidence, default=str), encoding="utf-8")
        log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        log.to_csv(LOG, index=False)
        if row["decision"] == "kept" and exp_id != "E0_baseline":
            state["spec"], state["best_oof"] = spec, f"oof_{exp_id}.csv"
            state["kept"].append(exp_id)
            state["no_gain_streak"] = 0 if kind == "perf" else state["no_gain_streak"]
        elif row["decision"] == "reverted" and kind == "perf":
            # E3a-E3d are four variants of ONE change (class weighting): the streak grows once, after E3d.
            if not exp_id.startswith("E3") or (exp_id == "E3d" and not any(k.startswith("E3") for k in state["kept"])):
                state["no_gain_streak"] += 1
        if state["no_gain_streak"] >= MAX_NO_GAIN:
            state["stopped"] = True
        STATE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        print(f"   {row['decision']}: {row.get('decision_reason', '')} | PR-AUC={row['pr_auc']:.5f} "
              f"Brier={row['brier']:.5f} LogLoss={row['log_loss']:.5f}")
        if state["stopped"]:
            print("   STOPPING RULE: two consecutive justified changes gave no significant gain.")


if __name__ == "__main__":
    main()
