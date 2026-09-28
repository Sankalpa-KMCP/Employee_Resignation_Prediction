"""Tests for the frozen M6 TabNet contract and fold-local preprocessing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import src.models.tabnet_model as tabnet_module
from src.models.base import ModelAdapter
from src.models.tabnet_model import (
    TabNetAdapter,
    TabNetPreprocessor,
    load_tabnet_config,
    validate_tabnet_config,
)


def _example_features() -> pd.DataFrame:
    config = load_tabnet_config()
    categorical = config["preprocessing"]["categorical_features"]
    numerical = config["preprocessing"]["numerical_features"]
    data: dict[str, list[object]] = {
        categorical[0]: ["Engineering", "HR", "Engineering", "Sales"],
        categorical[1]: ["Female", "Male", "Other", "Female"],
        categorical[2]: ["Engineer", "Manager", "Engineer", "Analyst"],
        categorical[3]: ["Bachelor", "Master", "PhD", "High School"],
    }
    for offset, column in enumerate(numerical):
        data[column] = [offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0]
    return pd.DataFrame(data)


def _training_features(rows: int = 40) -> pd.DataFrame:
    base = _example_features()
    repeated = pd.concat([base] * ((rows + len(base) - 1) // len(base)), ignore_index=True)
    return repeated.iloc[:rows].copy()


class _FakeTabNetClassifier:
    """Small deterministic stand-in for unit-testing adapter orchestration."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.best_epoch = 1
        self.best_cost = 0.5
        self.history = SimpleNamespace(
            history={"loss": [0.8, 0.7], "inner_valid_average_precision": [0.4, 0.5]}
        )
        self.positive_probability = 0.4

    def fit(self, X, y, **kwargs):
        self.positive_probability = 0.4 + 0.01 * int(kwargs.get("weights", 0))
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), self.positive_probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])

    def save_model(self, path):
        saved_path = Path(f"{path}.zip")
        saved_path.write_text(str(self.positive_probability), encoding="utf-8")
        return str(saved_path)

    def load_model(self, path):
        self.positive_probability = float(Path(path).read_text(encoding="utf-8"))


def _mock_tabnet_dependencies(monkeypatch):
    fake_torch = SimpleNamespace(optim=SimpleNamespace(Adam=object))
    monkeypatch.setattr(
        tabnet_module,
        "_tabnet_dependencies",
        lambda: (fake_torch, object, _FakeTabNetClassifier),
    )


def test_frozen_tabnet_config_is_bounded_and_holdout_safe():
    config = load_tabnet_config()
    assert config["model"]["id"] == "m6_tabnet"
    assert config["evaluation"]["primary_metric"] == "average_precision"
    assert config["evaluation"]["formal_holdout_allowed"] is False
    assert len(config["candidates"]) == 8
    assert config["training"]["max_epochs"] <= 150
    assert config["training"]["patience"] <= 15
    assert config["fixed_params"]["cat_emb_dim"] == 1
    assert config["training"]["cat_emb_dim"] == 1
    assert config["fixed_params"]["cat_emb_dim"] == config["training"]["cat_emb_dim"]


def test_tabnet_features_exactly_match_shared_feature_contract():
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "configs" / "feature_contract.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        feature_contract = yaml.safe_load(handle)
    config = load_tabnet_config()
    tabnet_features = (
        config["preprocessing"]["categorical_features"]
        + config["preprocessing"]["numerical_features"]
    )
    assert tabnet_features == feature_contract["primary_model_features"]
    assert len(tabnet_features) == 17
    assert {"Employee_ID", "Hire_Date", "Resigned"}.isdisjoint(tabnet_features)


def test_noncanonical_embedding_dimension_is_rejected():
    config = deepcopy(load_tabnet_config())
    config["training"]["cat_emb_dim"] = 2
    config["fixed_params"]["cat_emb_dim"] = 2
    with pytest.raises(ValueError, match="canonical value 1"):
        validate_tabnet_config(config)


def test_mismatched_embedding_sections_are_rejected():
    config = deepcopy(load_tabnet_config())
    config["fixed_params"]["cat_emb_dim"] = 2
    with pytest.raises(ValueError, match="must agree"):
        validate_tabnet_config(config)


def test_candidate_outside_frozen_search_space_is_rejected():
    config = deepcopy(load_tabnet_config())
    config["candidates"][0]["virtual_batch_size"] = 256
    config["candidates"][0]["batch_size"] = 1025
    with pytest.raises(ValueError, match="invalid batch_size"):
        validate_tabnet_config(config)


def test_preprocessor_uses_training_only_category_vocabulary():
    config = load_tabnet_config()
    categorical = config["preprocessing"]["categorical_features"]
    numerical = config["preprocessing"]["numerical_features"]
    train = _example_features()
    preprocessor = TabNetPreprocessor(categorical, numerical).fit(train)

    validation = train.iloc[[0]].copy()
    validation.loc[:, "Department"] = "Unseen Department"
    matrix = preprocessor.transform(validation)

    assert matrix.shape == (1, 17)
    assert matrix[0, 0] == 0.0
    assert np.isfinite(matrix).all()
    assert preprocessor.cat_idxs == [0, 1, 2, 3]
    assert len(preprocessor.cat_dims) == 4


def test_preprocessor_standardizes_training_numerical_features():
    config = load_tabnet_config()
    categorical = config["preprocessing"]["categorical_features"]
    numerical = config["preprocessing"]["numerical_features"]
    matrix = TabNetPreprocessor(categorical, numerical).fit_transform(_example_features())
    numerical_matrix = matrix[:, len(categorical) :]
    np.testing.assert_allclose(numerical_matrix.mean(axis=0), 0.0, atol=1e-6)


def test_tabnet_adapter_satisfies_shared_model_protocol_without_training():
    adapter = TabNetAdapter(candidate_id="T01", max_epochs_override=3, patience_override=2)
    assert isinstance(adapter, ModelAdapter)
    assert adapter.feature_order == (
        adapter.categorical_features + adapter.numerical_features
    )
    with pytest.raises(RuntimeError, match="fitted before prediction"):
        adapter.predict_proba(_example_features())


def test_unknown_candidate_is_rejected():
    with pytest.raises(ValueError, match="Unknown candidate_id"):
        TabNetAdapter(candidate_id="not-declared")


def test_auto_selection_metadata_and_tie_breaking(monkeypatch):
    _mock_tabnet_dependencies(monkeypatch)
    adapter = TabNetAdapter(
        candidate_id="auto",
        max_epochs_override=3,
        patience_override=2,
        outer_fold=1,
    )
    features = _training_features()
    target = pd.Series(([0, 1] * 20), dtype=int)

    adapter.fit(features, target)
    metadata = adapter.get_metadata()

    assert metadata["outer_fold"] == 1
    assert metadata["selection_mode"] == "auto"
    assert metadata["selected_candidate_id"] == "T01"
    assert metadata["selected_candidate_parameters"]["id"] == "T01"
    assert metadata["selected_epoch"] == 2
    assert metadata["best_inner_average_precision"] == pytest.approx(0.5)
    assert metadata["seed"] == 42
    assert metadata["class_balance"] is False
    assert len(metadata["candidate_selection_records"]) == 8
    assert set(metadata["training_histories"]) == {
        f"T{candidate_number:02d}" for candidate_number in range(1, 9)
    }
    assert metadata["fit_summary"]["inner_training_rows"] == 35
    assert metadata["fit_summary"]["inner_validation_rows"] == 5


def test_save_load_preserves_state_and_predictions(monkeypatch, tmp_path):
    _mock_tabnet_dependencies(monkeypatch)
    adapter = TabNetAdapter(
        candidate_id="auto",
        max_epochs_override=3,
        patience_override=2,
        outer_fold=3,
    )
    features = _training_features()
    target = pd.Series(([0, 1] * 20), dtype=int)
    adapter.fit(features, target)
    expected = adapter.predict_proba(features.iloc[:10])

    adapter.save(tmp_path)
    reloaded = TabNetAdapter.load(tmp_path)
    actual = reloaded.predict_proba(features.iloc[:10])

    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-7)
    assert reloaded.selected_epoch == adapter.selected_epoch
    assert reloaded.candidate_id == adapter.candidate_id
    assert reloaded.outer_fold == 3
    assert reloaded.preprocessor is not None
    assert reloaded.preprocessor.feature_order == adapter.preprocessor.feature_order
    assert reloaded.preprocessor.category_mappings == adapter.preprocessor.category_mappings
    assert reloaded.candidate_selection_records == adapter.candidate_selection_records
    assert reloaded.training_histories == adapter.training_histories
    assert reloaded.fit_summary == adapter.fit_summary
