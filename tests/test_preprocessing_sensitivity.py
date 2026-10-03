"""Verify exploratory sampling boundaries without training real models."""

import hashlib
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import run_preprocessing_sensitivity as sensitivity


@pytest.fixture
def sensitivity_checkout(tmp_path):
    (tmp_path / "configs").mkdir()
    raw = pd.DataFrame({"Employee_ID": range(1, 23),
                        "Resigned": [0, 1] * 10 + ["locked", "locked"],
                        "Monthly_Salary": list(range(1000, 1020)) + ["locked", "locked"]})
    split = pd.DataFrame({"Employee_ID": range(1, 23),
                          "partition": ["development"] * 20 + ["holdout"] * 2,
                          "cv_fold": list(np.repeat(range(1, 6), 4)) + [None, None]})
    raw.to_csv(tmp_path / "raw.csv", index=False)
    split.to_csv(tmp_path / "split.csv", index=False)
    contract = {"primary_model_features": ["Monthly_Salary"], "categorical_features": []}
    for name, filename in (("dataset", "raw.csv"), ("split_registry", "split.csv")):
        contract[name] = {"path": filename,
                          "sha256": hashlib.sha256((tmp_path / filename).read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
    contract["split_registry"]["development_rows"] = 20
    (tmp_path / "configs/feature_contract.yaml").write_text(yaml.safe_dump(contract))
    return tmp_path


def test_sensitivity_sampling_skips_holdout_values_and_retains_outer_boundary(sensitivity_checkout, monkeypatch):
    monkeypatch.setattr(sensitivity, "TRAIN_ROWS", 10)
    monkeypatch.setattr(sensitivity, "VALIDATION_ROWS", 2)
    train, valid, y_train, y_valid, categories = sensitivity.load_data(sensitivity_checkout)
    # Salary values identify the source rows without including IDs as predictors.
    assert set(train.Monthly_Salary).issubset(set(range(1004, 1020)))
    assert set(valid.Monthly_Salary).issubset(set(range(1000, 1004)))
    assert set(train.Monthly_Salary).isdisjoint(valid.Monthly_Salary)
    assert len(train) == len(y_train) == 10
    assert len(valid) == len(y_valid) == 2
    assert set(y_train) == set(y_valid) == {0, 1}
    assert categories == []


@pytest.mark.parametrize("filename", ["raw.csv", "split.csv"])
def test_sensitivity_rejects_changes_to_frozen_files(sensitivity_checkout, filename):
    path = sensitivity_checkout / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        sensitivity.load_data(sensitivity_checkout)


def test_sensitivity_reruns_preserve_original_and_previous_evidence(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(sensitivity, "ROOT", tmp_path)
    monkeypatch.setattr(sensitivity, "TRAIN_ROWS", 10)
    monkeypatch.setattr(sensitivity, "VALIDATION_ROWS", 2)
    columns = ["Monthly_Salary", *[f"feature_{i}" for i in range(16)]]
    train = pd.DataFrame(np.ones((10, 17)), columns=columns)
    valid = pd.DataFrame(np.ones((2, 17)), columns=columns)
    monkeypatch.setattr(sensitivity, "load_data", lambda: (train, valid, pd.Series([0, 1] * 5), pd.Series([0, 1]), []))

    class SyntheticAdapter:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.full(len(X), .25)

    monkeypatch.setattr(sensitivity, "make_adapter", lambda *args: SyntheticAdapter())
    original = tmp_path / "results/models/preprocessing_sensitivity/paired_exploratory.csv"
    original.parent.mkdir(parents=True)
    original.write_text("saved original evidence")
    sensitivity.main()
    first = next((original.parent / "reruns").iterdir())
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    sensitivity.main()
    assert original.read_text() == "saved original evidence"
    assert len(list((original.parent / "reruns").iterdir())) == 2
    assert before == {p.name: p.read_bytes() for p in first.iterdir()}
    assert len(pd.read_csv(first / "paired_exploratory.csv")) == 16
