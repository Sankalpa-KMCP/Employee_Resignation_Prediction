"""Protect saved review from accidental training and validate shared input boundaries."""

from pathlib import Path
import hashlib
import json

import pandas as pd
import pytest
import yaml

from src.evaluation.inputs import load_development_data, load_group_inputs, load_saved_model_evidence
from src.models.experiment_outputs import create_run_directory, refresh_artifact_inventory


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def small_checkout(tmp_path):
    (tmp_path / "configs").mkdir()
    split = pd.DataFrame({"Employee_ID": range(1, 8), "partition": ["development"] * 5 + ["holdout"] * 2,
                          "cv_fold": [1, 2, 3, 4, 5, None, None]})
    # Invalid holdout labels prove the reader skips them before converting the target.
    raw = pd.DataFrame({"Employee_ID": range(1, 8), "Resigned": [0, 1, 0, 1, 0, "locked", "locked"]})
    raw.to_csv(tmp_path / "raw.csv", index=False)
    split.to_csv(tmp_path / "split.csv", index=False)
    contract = {
        "dataset": {"path": "raw.csv", "sha256": hashlib.sha256((tmp_path / "raw.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()},
        "split_registry": {"path": "split.csv", "sha256": hashlib.sha256((tmp_path / "split.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest(), "development_rows": 5},
    }
    (tmp_path / "configs/feature_contract.yaml").write_text(yaml.safe_dump(contract))
    registry = {"models": {"CatBoost": {"model_name": "CatBoost Classifier", "oof_path": "oof.csv"}}}
    (tmp_path / "configs/analysis_registry.yaml").write_text(yaml.safe_dump(registry))
    predictions = pd.DataFrame({"Employee_ID": range(1, 6), "cv_fold": range(1, 6),
                                "y_true": [0, 1, 0, 1, 0], "y_probability": [.1, .7, .3, .6, .2],
                                "model_name": "CatBoost Classifier"})
    predictions.to_csv(tmp_path / "oof.csv", index=False)
    return tmp_path


def test_shared_loader_skips_holdout_targets_and_sorts(small_checkout):
    frame = pd.read_csv(small_checkout / "oof.csv").iloc[::-1]
    frame.to_csv(small_checkout / "oof.csv", index=False)
    _, frames = load_group_inputs(small_checkout)
    assert frames["CatBoost"].Employee_ID.tolist() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("column,value,error", [
    ("y_true", 1, "original dataset"),
    ("cv_fold", 2, "frozen split registry"),
    ("cv_fold", 1.5, "integers"),
    ("y_probability", float("inf"), "non-finite"),
    ("y_probability", -0.1, "between 0 and 1"),
    ("Employee_ID", 6, "development partition"),
    ("model_name", "wrong", "exactly one model_name"),
])
def test_shared_loader_rejects_corrupt_model_input(small_checkout, column, value, error):
    frame = pd.read_csv(small_checkout / "oof.csv")
    frame[column] = frame[column].astype(float if column == "cv_fold" else object)
    frame.loc[0, column] = value
    if column == "cv_fold" and value == 2:
        frame.loc[1, column] = 1  # Retain all five folds while violating frozen assignment.
    frame.to_csv(small_checkout / "oof.csv", index=False)
    with pytest.raises(ValueError, match=error):
        load_group_inputs(small_checkout)


def test_shared_loader_checks_frozen_source_files(small_checkout):
    path = small_checkout / "split.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_development_data(small_checkout)


def test_new_runs_cannot_overwrite_canonical_or_existing_output(tmp_path):
    canonical = tmp_path / "results/models/ebm/individual_IT25100429"
    canonical.mkdir(parents=True)
    protected = canonical / "evidence.csv"
    protected.write_text("historical")
    with pytest.raises(ValueError, match="beneath"):
        create_run_directory(tmp_path, "ebm", canonical)
    first = create_run_directory(tmp_path, "ebm")
    second = create_run_directory(tmp_path, "ebm")
    assert first != second
    assert first.parent == tmp_path / "results/models/ebm/reruns"
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "ebm", first)
    with pytest.raises(ValueError):
        create_run_directory(tmp_path, "ebm", tmp_path / "outside")
    assert protected.read_text() == "historical"


def test_inventory_reports_missing_artifacts_and_changed_metadata(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"corrected":true}')
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"artifact": ["metadata.json", "missing.zip", "manifest.csv"],
                  "exists": [True, True, True], "size_bytes": [10, 100, None],
                  "sha256": ["stale", "unavailable-old-hash", None]}).to_csv(manifest_path, index=False)
    inventory = refresh_artifact_inventory(tmp_path, manifest_path)
    assert inventory.loc[0, "sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()
    assert not inventory.loc[1, "exists"]
    assert pd.isna(inventory.loc[1, "sha256"]) and pd.isna(inventory.loc[1, "size_bytes"])
    assert inventory.loc[1, "availability_note"] == "not available in this checkout"
    assert pd.isna(inventory.loc[2, "sha256"]) and pd.isna(inventory.loc[2, "size_bytes"])


def test_legacy_catboost_inventory_refreshes_windows_relative_paths(tmp_path):
    artifact = tmp_path / "results/models/catboost/oof.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("synthetic evidence")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"file_name": ["oof.csv", "missing.cbm"],
                  "relative_path": [r"results\models\catboost\oof.csv", r"models\catboost\missing.cbm"],
                  "size_bytes": [1, 99], "exists": [True, True]}).to_csv(manifest_path, index=False)
    refreshed = refresh_artifact_inventory(tmp_path, manifest_path)
    assert refreshed.loc[0, "exists"]
    assert refreshed.loc[0, "size_bytes"] == artifact.stat().st_size
    assert refreshed.loc[0, "sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert not refreshed.loc[1, "exists"]
    assert pd.isna(refreshed.loc[1, "size_bytes"])
    assert pd.isna(refreshed.loc[1, "sha256"])


def test_inventory_rejects_paths_outside_checkout(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"artifact": ["../outside.txt"], "size_bytes": [1]}).to_csv(manifest_path, index=False)
    original = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="escapes"):
        refresh_artifact_inventory(tmp_path, manifest_path)
    assert manifest_path.read_bytes() == original


def test_inventory_uses_relative_path_instead_of_artifact_label_and_adds_sizes(tmp_path):
    # FT-Transformer's original schema has a display label, path and exists only.
    artifact = tmp_path / "results/models/ft_transformer/oof.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("synthetic evidence")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"artifact": ["oof.csv", "missing.zip"],
                  "relative_path": [r"results\models\ft_transformer\oof.csv", r"models\missing.zip"],
                  "exists": [True, True]}).to_csv(manifest_path, index=False)
    refreshed = refresh_artifact_inventory(tmp_path, manifest_path)
    assert refreshed.loc[0, "exists"]
    assert refreshed.loc[0, "size_bytes"] == artifact.stat().st_size
    assert refreshed.loc[0, "sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert not refreshed.loc[1, "exists"]
    assert pd.isna(refreshed.loc[1, "size_bytes"])
    assert pd.isna(refreshed.loc[1, "sha256"])


def test_development_loader_rejects_fractional_source_labels(small_checkout):
    raw_path = small_checkout / "raw.csv"
    raw = pd.read_csv(raw_path)
    raw.loc[0, "Resigned"] = "0.8"
    raw.to_csv(raw_path, index=False)
    contract_path = small_checkout / "configs/feature_contract.yaml"
    contract = yaml.safe_load(contract_path.read_text())
    contract["dataset"]["sha256"] = hashlib.sha256(raw_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    contract_path.write_text(yaml.safe_dump(contract))
    with pytest.raises(ValueError, match="Resigned must contain only binary"):
        load_development_data(small_checkout)


def test_oof_parity_rejects_fractional_original_source_labels(small_checkout):
    from src.evaluation.oof_validation import validate_oof_predictions

    source, split = load_development_data(small_checkout)
    source["Resigned"] = source.Resigned.astype(float)
    source.loc[0, "Resigned"] = 0.8
    predictions = pd.read_csv(small_checkout / "oof.csv")
    with pytest.raises(ValueError, match="Resigned must contain only binary"):
        validate_oof_predictions(predictions, split, source_data=source)


@pytest.mark.parametrize("member,prefix", [("member_2_catboost", "06"), ("member_3_xgboost", "07"), ("member_5_ebm", "09")])
def test_default_notebooks_never_fit_or_export(monkeypatch, member, prefix):
    import IPython.display
    from src.models import catboost_model, xgboost_model, ebm_experiments, runner, experiment_outputs

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved review attempted training or artifact mutation.")

    monkeypatch.setattr(catboost_model.CatBoostAdapter, "fit", forbidden)
    monkeypatch.setattr(xgboost_model.XGBoostAdapter, "fit", forbidden)
    monkeypatch.setattr(ebm_experiments, "fit_candidate", forbidden)
    monkeypatch.setattr(ebm_experiments, "run_experiments", forbidden)
    monkeypatch.setattr(runner, "run_5fold_cv", forbidden)
    monkeypatch.setattr(experiment_outputs, "create_run_directory", forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_csv", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
    path = next((ROOT / "notebooks/team_contributions/models" / member).glob(prefix + "*.ipynb"))
    monkeypatch.chdir(path.parent)
    namespace = {}
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)


def test_saved_review_rejects_summary_disagreement(tmp_path):
    from shutil import copyfile
    directory = tmp_path / "model"
    directory.mkdir()
    for filename in ("oof_predictions.csv", "fold_metrics.csv", "summary_metrics.json"):
        copyfile(ROOT / "results/models/xgboost" / filename, directory / filename)
    summary = json.loads((directory / "summary_metrics.json").read_text())
    summary["pooled_average_precision"] += .01
    (directory / "summary_metrics.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="disagrees"):
        load_saved_model_evidence(ROOT, "XGBoost", directory)
