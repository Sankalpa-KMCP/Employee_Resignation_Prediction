"""Protect the FT notebooks' saved-review and isolated-training workflows."""

from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.evaluation.inputs import load_development_data
from src.models import ft_transformer_execution as execution
from src.models.experiment_outputs import create_run_directory


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "notebooks/team_contributions/models/member_1_ft_transformer"


def notebook(name):
    return json.loads((DIRECTORY / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["05_m1_ft_transformer.ipynb", "05_m1_ft_transformer_historical.ipynb"])
def test_ft_saved_review_cannot_train_or_export(monkeypatch, name):
    import IPython.display
    from src.models import ft_transformer, runner, experiment_outputs
    import torch
    import joblib
    import pickle

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved FT review attempted training or artifact export.")

    monkeypatch.setattr(execution, "run", forbidden)
    monkeypatch.setattr(ft_transformer.FTTransformerAdapter, "fit", forbidden)
    monkeypatch.setattr(runner, "run_5fold_cv", forbidden)
    monkeypatch.setattr(experiment_outputs, "create_run_directory", forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_csv", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(plt.Figure, "savefig", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.chdir(DIRECTORY)
    namespace = {}
    try:
        for cell in notebook(name)["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), name, "exec"), namespace)
        if name == "05_m1_ft_transformer.ipynb":
            assert len(namespace["FIGURES"]) == 6
            assert namespace["prevalence"] == pytest.approx(0.1001)
            assert namespace["pooled"]["average_precision"] == pytest.approx(0.09864796531239689)
            assert namespace["pooled"]["roc_auc"] == pytest.approx(0.4979796229231931)
            assert namespace["summary"]["cv_mean_ap"] == pytest.approx(0.09806541461164146)
            assert namespace["normalized_cm"].sum(axis=1) == pytest.approx([1, 1])
            assert namespace["observed"]["count"].gt(0).all()
    finally:
        plt.close("all")


def test_ft_notebook_cell_structure_preserves_executable_history():
    main = notebook("05_m1_ft_transformer.ipynb")["cells"]
    historical = notebook("05_m1_ft_transformer_historical.ipynb")["cells"]
    assert len(main) == 24
    assert sum(c["cell_type"] == "code" for c in main) == 12
    assert sum(c["cell_type"] == "markdown" for c in main) == 12
    assert all(c["cell_type"] == ("markdown" if i % 2 == 0 else "code") for i, c in enumerate(main))
    assert sum(c["cell_type"] == "code" for c in historical) == 15


@pytest.mark.parametrize("destination", ["results/models/ft_transformer/strict_nested", "results/models/ft_transformer/reruns", "../outside", "results/models/ft_transformer/reruns/../canonical"])
def test_ft_runner_rejects_destinations_outside_reruns(tmp_path, destination):
    with pytest.raises(ValueError, match="beneath"):
        execution.run(tmp_path, output_dir=destination)
    assert not (tmp_path / "results").exists()


def test_ft_runner_destination_preserves_legacy_default_and_isolates_explicit_run(tmp_path):
    assert execution._resolve_output_directory(tmp_path) == tmp_path / "results/models/ft_transformer/strict_nested"
    relative = "results/models/ft_transformer/reruns/new-run"
    assert execution._resolve_output_directory(tmp_path, relative) == tmp_path / relative
    assert execution._resolve_output_directory(tmp_path, tmp_path / relative) == tmp_path / relative


def test_ft_notebook_new_training_uses_returned_evidence_in_isolated_directory(monkeypatch, tmp_path):
    import IPython.display

    source = ROOT / "results/models/ft_transformer/strict_nested"
    saved_result = execution.load_formal_ft_input(ROOT)
    data, registry = load_development_data(ROOT)
    called = []

    def stub_run(root, output_dir=None):
        assert root == tmp_path
        assert output_dir.is_relative_to(tmp_path / "results/models/ft_transformer/reruns")
        assert output_dir != source
        called.append(output_dir)
        for name in ("candidate_selection.csv", "selection_evidence.json"):
            shutil.copyfile(source / name, output_dir / name)
        return saved_result

    def no_canonical_fallback(*args, **kwargs):
        raise AssertionError("New training incorrectly loaded canonical evidence.")

    monkeypatch.setattr(execution, "run", stub_run)
    monkeypatch.setattr(execution, "load_formal_ft_input", no_canonical_fallback)
    monkeypatch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
    monkeypatch.chdir(DIRECTORY)
    cells = notebook("05_m1_ft_transformer.ipynb")["cells"]
    namespace = {}
    exec("".join(cells[1]["source"]), namespace)
    namespace.update(ROOT=tmp_path, RUN_TRAINING=True, development_data=data, split_registry=registry)
    namespace["candidate_ids"] = [f"FT{i:02}" for i in range(1, 9)]
    exec("".join(cells[13]["source"]), namespace)
    assert len(called) == 1
    assert namespace["active_results_dir"] == called[0]
    assert namespace["oof"] is saved_result[0]
    assert namespace["summary"] is saved_result[2]
    assert len(namespace["selection"]) == 40


@pytest.mark.parametrize("corruption", ["id", "target", "fold", "nan", "infinity", "negative", "above_one"])
def test_ft_review_rejects_invalid_predictions(corruption):
    import numpy as np
    from src.evaluation.oof_validation import validate_oof_predictions

    source, registry = load_development_data(ROOT, usecols=[])
    predictions, _, _ = execution.load_formal_ft_input(ROOT)
    predictions = predictions.copy()
    if corruption == "id":
        predictions.loc[0, "Employee_ID"] = registry.loc[registry.partition == "holdout", "Employee_ID"].iloc[0]
    elif corruption == "target":
        predictions.loc[0, "y_true"] = 1 - predictions.loc[0, "y_true"]
    elif corruption == "fold":
        predictions.loc[0, "cv_fold"] = predictions.loc[0, "cv_fold"] % 5 + 1
    else:
        predictions.loc[0, "y_probability"] = {"nan": np.nan, "infinity": np.inf,
                                                "negative": -0.01, "above_one": 1.01}[corruption]
    with pytest.raises(ValueError):
        validate_oof_predictions(predictions, registry, source_data=source,
                                 expected_model_name="FT-Transformer")
