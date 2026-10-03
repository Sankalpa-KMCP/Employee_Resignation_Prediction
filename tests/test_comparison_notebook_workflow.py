"""Comparison notebook safety, canonical metrics and export compatibility."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/group/02_model_comparison.ipynb"


def read_notebook():
    return json.loads(PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def verified_review():
    import IPython.display
    import joblib
    import pickle
    import torch
    from src.models import runner, ft_transformer_execution, ebm_experiments

    def forbidden(*args, **kwargs):
        raise AssertionError("Default comparison attempted training, weight loading or export.")

    namespace = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "run_5fold_cv", forbidden)
        patch.setattr(ft_transformer_execution, "run", forbidden)
        patch.setattr(ebm_experiments, "run_experiments", forbidden)
        patch.setattr(joblib, "load", forbidden)
        patch.setattr(torch, "load", forbidden)
        patch.setattr(pickle, "load", forbidden)
        patch.setattr(pd.DataFrame, "to_csv", forbidden)
        patch.setattr(Path, "write_text", forbidden)
        patch.setattr(Path, "write_bytes", forbidden)
        patch.setattr(Path, "mkdir", forbidden)
        patch.setattr(plt.Figure, "savefig", forbidden)
        patch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
        patch.setattr(plt, "show", lambda: None)
        patch.chdir(PATH.parent)
        for cell in read_notebook()["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), "comparison-cell", "exec"), namespace)
    yield namespace
    plt.close("all")


def test_default_comparison_is_complete_and_read_only(verified_review):
    cells = read_notebook()["cells"]
    assert len(cells) == 30
    assert sum(c["cell_type"] == "code" for c in cells) == 15
    assert sum(c["cell_type"] == "markdown" for c in cells) == 15
    assert not verified_review["EXPORT_RESULTS"]
    assert len(verified_review["rendered_figures"]) == 8
    assert len(verified_review["comparison"]) == 6
    assert len(verified_review["folds"]) == 30
    assert len(verified_review["baselines"]) == 2


def test_comparison_preserves_canonical_values_and_scientific_references(verified_review):
    ns = verified_review
    assert ns["summary"]["ranking_leader"] == "XGBoost"
    assert ns["mean_fold_leader"].model == "TabNet"
    assert np.isclose(ns["leader"].pooled_average_precision, 0.10118149850917421)
    assert np.isclose(ns["mean_fold_leader"].cv_ap_mean, 0.101700, atol=1e-6)
    assert np.isclose(ns["prevalence"], 0.1001)
    assert np.isclose(ns["majority_accuracy"], 0.8999)
    assert np.isclose(ns["constant_brier"], 0.1001 * 0.8999)
    for model, normalized in ns["normalized_matrices"].items():
        assert np.allclose(normalized.sum(axis=1), 1)
        row = ns["classification"].set_index("model").loc[model]
        assert np.isclose(normalized[1, 1], row.recall)
        if row.predicted_positive_rate == 0:
            assert np.isnan(ns["classification_display"].set_index("model").loc[model, "precision"])
    catboost = ns["protocol_inventory"].set_index("model").loc["CatBoost"]
    assert catboost.candidate_budget == "not recorded"
    assert "catboost_run_metadata" not in catboost.evidence_source


def test_explicit_export_round_trips_and_keeps_existing_schema(verified_review, tmp_path):
    from PIL import Image

    ns = verified_review
    previous_out = ns["out"]
    try:
        ns.update(EXPORT_RESULTS=True, out=tmp_path / "comparison")
        exec("".join(read_notebook()["cells"][28]["source"]), ns)
        for filename, key in [("model_comparison.csv", "comparison"), ("fold_metrics_long.csv", "folds"),
                              ("baseline_comparison.csv", "baselines")]:
            saved = pd.read_csv(ns["out"] / filename)
            pd.testing.assert_frame_equal(saved, ns[key], check_dtype=False, rtol=1e-12, atol=1e-12)
            original = pd.read_csv(ROOT / "results/comparison" / filename)
            pd.testing.assert_frame_equal(saved, original, check_dtype=False, rtol=1e-12, atol=1e-12)
        assert json.loads((ns["out"] / "comparison_summary.json").read_text()) == ns["summary"]
        assert not ns["summary"]["final_model_frozen"]
        assert not ns["summary"]["formal_holdout_accessed"]
        pngs = list(ns["out"].glob("*.png"))
        assert len(pngs) == 8
        for path in pngs:
            with Image.open(path) as image:
                image.verify()
        assert set(ns["rendered_figures"]).issuperset({"figure_01_average_precision_comparison.png",
                                                       "figure_02_roc_auc_comparison.png",
                                                       "figure_03_fold_ap_stability.png"})
    finally:
        ns.update(EXPORT_RESULTS=False, out=previous_out)


@pytest.mark.parametrize("corruption", ["missing_model", "id", "target", "fold", "nan", "infinity", "negative_probability", "high_probability"])
def test_invalid_model_inputs_stop_comparison(verified_review, monkeypatch, corruption):
    import IPython.display
    from src.evaluation import inputs

    frames = verified_review["frames"].copy()
    if corruption == "missing_model":
        frames.pop("CatBoost")
    else:
        damaged = frames["XGBoost"].copy()
        index = damaged.index[0]
        if corruption == "id":
            damaged.loc[index, "Employee_ID"] = next(iter(verified_review["holdout_ids"]))
        elif corruption == "target":
            damaged.loc[index, "y_true"] = 1 - int(damaged.loc[index, "y_true"])
        elif corruption == "fold":
            damaged.loc[index, "cv_fold"] = 1 if damaged.loc[index, "cv_fold"] != 1 else 2
        else:
            damaged.loc[index, "y_probability"] = {"nan": np.nan, "infinity": np.inf,
                                                    "negative_probability": -0.1, "high_probability": 1.1}[corruption]
        frames["XGBoost"] = damaged
    monkeypatch.setattr(inputs, "load_group_inputs", lambda root: (verified_review["registry"], frames))
    monkeypatch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
    monkeypatch.chdir(PATH.parent)
    namespace = {}
    cells = read_notebook()["cells"]
    exec("".join(cells[1]["source"]), namespace)
    with pytest.raises(ValueError):
        exec("".join(cells[3]["source"]), namespace)
