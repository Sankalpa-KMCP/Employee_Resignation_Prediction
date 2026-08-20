"""Unit tests for model adapter implementations and ModelAdapter Protocol compliance."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.models.base import ModelAdapter
from src.models.catboost_model import CatBoostAdapter
from src.models.ebm_model import EBMAdapter
from src.models.ft_transformer import FTTransformerAdapter
from src.models.lightgbm_model import LightGBMAdapter
from src.models.logistic_regression import LogisticRegressionAdapter
from src.models.tabnet_model import TabNetAdapter
from src.models.xgboost_model import XGBoostAdapter

ALL_ADAPTER_CLASSES = [
    CatBoostAdapter,
    EBMAdapter,
    FTTransformerAdapter,
    LightGBMAdapter,
    LogisticRegressionAdapter,
    TabNetAdapter,
    XGBoostAdapter,
]


@pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
def test_adapter_satisfies_model_adapter_protocol(adapter_cls):
    """Verify that every adapter class satisfies the runtime ModelAdapter Protocol."""
    adapter = adapter_cls()
    assert isinstance(adapter, ModelAdapter), (
        f"{adapter_cls.__name__} does not satisfy ModelAdapter Protocol."
    )


@pytest.mark.parametrize("adapter_cls", ALL_ADAPTER_CLASSES)
def test_adapter_initialization_and_metadata(adapter_cls):
    """Verify clean instantiation and metadata reporting without side-effects."""
    adapter = adapter_cls()
    assert hasattr(adapter, "name")
    assert hasattr(adapter, "model_id")
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict_proba")
    assert hasattr(adapter, "save")
    assert hasattr(adapter, "load")
    assert hasattr(adapter, "get_metadata")

    metadata = adapter.get_metadata()
    assert isinstance(metadata, dict)
    assert metadata["name"] == adapter.name
    assert metadata["model_id"] == adapter.model_id
    assert metadata["fitted"] is False


def test_logistic_regression_adapter_pca_configuration():
    """Verify LogisticRegressionAdapter handles both standard and PCA configurations."""
    lr_standard = LogisticRegressionAdapter(use_pca=False)
    assert lr_standard.use_pca is False

    lr_pca = LogisticRegressionAdapter(use_pca=True, pca_variance_ratio=0.95)
    assert lr_pca.use_pca is True
    assert lr_pca.pca_variance_ratio == 0.95
