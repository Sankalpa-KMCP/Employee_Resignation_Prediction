"""Tests for the frozen M6 TabNet contract and fold-local preprocessing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

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


def test_frozen_tabnet_config_is_bounded_and_holdout_safe():
    config = load_tabnet_config()
    assert config["model"]["id"] == "m6_tabnet"
    assert config["evaluation"]["primary_metric"] == "average_precision"
    assert config["evaluation"]["formal_holdout_allowed"] is False
    assert len(config["candidates"]) == 8
    assert config["training"]["max_epochs"] <= 150
    assert config["training"]["patience"] <= 15


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
