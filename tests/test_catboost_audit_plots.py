"""Smoke tests: every audit figure function renders a 300-dpi PNG from synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.evaluation import audit_plots as plots
from src.evaluation.audit_stats import topk_table


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.1).astype(int)
    data = pd.DataFrame({"Resigned": y, "cv_fold": np.arange(n) % 5 + 1})
    return {"y": y, "p": rng.random(n) * 0.02 + 0.09, "q": np.full(n, y.mean()), "data": data}


def _is_300dpi(path):
    with Image.open(path) as im:
        dpi = im.info.get("dpi", (0, 0))
    return round(dpi[0]) == 300


def test_distribution_layout_roc_pr_and_strip(tmp_path, synth):
    y, p, q = synth["y"], synth["p"], synth["q"]
    out = [
        plots.fig_target_distribution(tmp_path / "a.png", synth["data"]),
        plots.fig_nested_cv_diagram(tmp_path / "b.png", 80000, 20000, 16000, 0.15),
        plots.fig_roc(tmp_path / "c.png", y, {"CatBoost (final)": p, "Dummy prior": q}, {"CatBoost (final)": (0.5, 0.49, 0.51), "Dummy prior": (0.5, 0.5, 0.5)}, n_boot=20),
        plots.fig_pr(tmp_path / "d.png", y, {"CatBoost (final)": p, "Dummy prior": q}, {"CatBoost (final)": (0.1, 0.09, 0.11), "Dummy prior": (0.1, 0.1, 0.1)}),
        plots.fig_fold_strip(tmp_path / "e.png", {"CatBoost (final)": pd.DataFrame({"pr_auc": [0.1, 0.11, 0.09, 0.1, 0.1], "roc_auc": [0.5] * 5}),
                                                  "Dummy prior": pd.DataFrame({"pr_auc": [0.1] * 5, "roc_auc": [0.5] * 5})}, 0.1),
    ]
    assert all(o.exists() and _is_300dpi(o) for o in out)


def test_bootstrap_curves_candidates_ablation(tmp_path, synth):
    rng = np.random.default_rng(1)
    out = [
        plots.fig_bootstrap_delta(tmp_path / "f.png", {"d1": rng.normal(0, 0.002, 500), "d2": rng.normal(0, 0.005, 500)}, "t"),
        plots.fig_learning_curves(tmp_path / "g.png", {"C1": {"Logloss": list(np.linspace(0.4, 0.33, 50)), "PRAUC": list(rng.random(50) * 0.01 + 0.1)}}, {"C1": 20}, "fold 1"),
        plots.fig_candidate_comparison(tmp_path / "h.png", pd.DataFrame({"candidate_id": ["a", "a", "b", "b"], "inner_ap": [0.1, 0.11, 0.105, 0.1]}), "inner_ap", "AP", 0.1, "prevalence"),
        plots.fig_tree_ablation(tmp_path / "i.png", pd.DataFrame({"trees": [1, 10, 100], "pr_auc": [0.1, 0.1, 0.099], "log_loss": [0.33, 0.33, 0.34]}), 0.1, 0.325),
    ]
    assert all(o.exists() and _is_300dpi(o) for o in out)


def test_reliability_probability_gain_importance_shap_scan_perm_subgroups(tmp_path, synth):
    y, p, q = synth["y"], synth["p"], synth["q"]
    rng = np.random.default_rng(2)
    tb = {"CatBoost (final)": topk_table(y, p), "Dummy prior": topk_table(y, q)}
    feats = ["a", "b", "c"]
    imp = pd.DataFrame(rng.random((3, 5)), index=feats, columns=[f"fold_{i}" for i in range(1, 6)])
    scan = pd.DataFrame({"feature": feats, "single_feature_auc": [0.5, 0.51, 0.49], "auc_distance_from_chance": [0.0, 0.01, 0.01],
                         "pearson_r": [0.0, 0.01, np.nan], "cramers_v": [0.0, 0.0, 0.01]})
    sub = {"Gender": pd.DataFrame({"Gender": ["F", "M"], "average_precision": [0.1, 0.11], "roc_auc": [0.5, 0.51], "prevalence": [0.1, 0.1]})}
    out = [
        plots.fig_reliability(tmp_path / "j.png", y, {"CatBoost (raw)": p, "CatBoost (calibrated)": q + rng.normal(0, 1e-4, len(q))}),
        plots.fig_prob_by_class(tmp_path / "k.png", y, p),
        plots.fig_gain_lift(tmp_path / "l.png", y, {}, tb),
        plots.fig_importance(tmp_path / "m.png", imp, imp - 0.2),
        plots.fig_shap(tmp_path / "n.png", rng.normal(size=(200, 3)), pd.DataFrame(rng.random((200, 3)), columns=feats), ["a", "b"]),
        plots.fig_single_feature_signal(tmp_path / "o.png", scan),
        plots.fig_permutation_test(tmp_path / "p.png", list(rng.normal(0.1, 0.001, 10)), 0.1005, "PR-AUC", 0.4),
        plots.fig_subgroups(tmp_path / "q.png", sub),
    ]
    assert all(o.exists() and _is_300dpi(o) for o in out)
