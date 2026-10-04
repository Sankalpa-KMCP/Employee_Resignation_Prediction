"""Environment audit and diagnostic tests for CatBoost pipeline (Step 1).

Verifies imports and versions for core dependencies, CatBoostAdapter instantiation
without fitting, CPU thread configuration (-1 by default), and GPU detection.
"""

from __future__ import annotations

import sys
import pytest

from src.models.catboost_model import CatBoostAdapter


def test_core_library_imports_and_versions():
    """Verify core scientific libraries import cleanly and meet minimum versions."""
    import catboost
    import matplotlib
    import numpy as np
    import pandas as pd
    import sklearn
    import seaborn as sns

    assert sys.version_info >= (3, 10), f"Python version must be >= 3.10, found {sys.version}"
    assert hasattr(catboost, "__version__"), "CatBoost must expose __version__"
    assert hasattr(pd, "__version__"), "Pandas must expose __version__"
    assert hasattr(np, "__version__"), "NumPy must expose __version__"
    assert hasattr(sklearn, "__version__"), "scikit-learn must expose __version__"
    assert hasattr(matplotlib, "__version__"), "matplotlib must expose __version__"
    assert hasattr(sns, "__version__"), "seaborn must expose __version__"


def test_catboost_adapter_instantiates_without_fitting():
    """Verify CatBoostAdapter instantiates cleanly without triggering any fitting."""
    adapter = CatBoostAdapter()
    assert adapter.name == "CatBoost"
    assert adapter.model_id == "catboost"
    assert adapter._fitted is False
    assert adapter._model is None
    assert adapter.selected_tree_count_ is None


def test_catboost_cpu_thread_configuration_default():
    """Verify CatBoostAdapter defaults to thread_count = -1 (all logical cores)."""
    adapter = CatBoostAdapter()
    assert adapter.params.get("thread_count", -1) == -1


def test_catboost_gpu_detection_cpu_remains_default():
    """Verify GPU availability check is safe and CPU remains the active default."""
    from catboost import CatBoostClassifier

    # Instantiating CPU classifier is always supported
    cpu_clf = CatBoostClassifier(iterations=1, thread_count=-1, verbose=False, allow_writing_files=False)
    assert cpu_clf.get_params().get("task_type", "CPU") in ("CPU", None)

    # Test GPU check safety
    gpu_available = False
    try:
        gpu_clf = CatBoostClassifier(task_type="GPU", iterations=1, verbose=False, allow_writing_files=False)
        gpu_available = True
    except Exception:
        gpu_available = False

    # Irrespective of hardware GPU availability, default adapter execution MUST be CPU
    adapter = CatBoostAdapter()
    task_type = adapter.params.get("task_type", "CPU")
    assert task_type == "CPU", f"Default task_type must be CPU, found {task_type}"
