"""Unit tests for PCA pipeline constraints, leakage guards, feature contracts, and notebook scaffold."""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "configs" / "feature_contract.yaml"
SPLIT_REGISTRY_PATH = PROJECT_ROOT / "data" / "processed" / "split_registry.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Extended_Employee_Performance_and_Productivity_Data.csv"
RESULTS_PCA_DIR = PROJECT_ROOT / "results" / "pca"
PCA_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04a_pca_analysis.ipynb"


@pytest.fixture
def feature_contract():
    """Load the authoritative feature contract."""
    with open(FEATURE_CONTRACT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def split_registry():
    """Load split registry."""
    return pd.read_csv(SPLIT_REGISTRY_PATH)


def test_pca_features_exclude_target_and_identifiers(feature_contract):
    """Verify target, identifier, and categorical columns are excluded from PCA feature candidates."""
    target_col = feature_contract["target"]["column"]
    id_col = feature_contract["identifier"]["column"]
    categorical_features = [item["name"] for item in feature_contract["categorical_features"]]
    numerical_features = [item["name"] for item in feature_contract["numerical_features"]]

    assert target_col not in numerical_features, "Target column must not be in PCA features."
    assert id_col not in numerical_features, "Identifier column must not be in PCA features."
    assert "Hire_Date" not in numerical_features, "Hire_Date must not be in PCA features."
    for cat in categorical_features:
        assert cat not in numerical_features, f"Categorical feature {cat} must not be in PCA features."
    assert len(numerical_features) == 13, "Expected exactly 13 numerical/ordinal features for PCA."


def test_pca_isolation_to_development_partition(split_registry):
    """Verify that PCA development input data strictly excludes holdout records."""
    dev_records = split_registry[split_registry["partition"] == "development"]
    holdout_records = split_registry[split_registry["partition"] == "holdout"]

    assert len(dev_records) == 80000
    assert len(holdout_records) == 20000

    dev_ids = set(dev_records["Employee_ID"])
    holdout_ids = set(holdout_records["Employee_ID"])

    assert dev_ids.isdisjoint(holdout_ids), "Development and holdout IDs must be disjoint."


def test_pca_variance_retention_and_scaling_simulation():
    """Verify that PCA with 95% variance parameterization retains expected variance."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 13

    # Generate synthetic correlated features
    base = np.random.randn(n_samples, 3)
    weights = np.random.randn(3, n_features)
    X_synthetic = np.dot(base, weights) + 0.1 * np.random.randn(n_samples, n_features)

    # 1. Fit StandardScaler on synthetic train fold
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_synthetic)

    # 2. Fit PCA with 95% variance retention
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    total_var = float(np.sum(pca.explained_variance_ratio_))
    assert total_var >= 0.95, f"Expected cumulative variance >= 0.95, got {total_var}"
    assert X_pca.shape[0] == n_samples
    assert X_pca.shape[1] <= n_features


def test_fold_local_pca_preprocessing_prevents_leakage():
    """Verify that fold-local StandardScaler and PCA fit only on training folds."""
    np.random.seed(42)
    n_train = 800
    n_val = 200
    n_features = 13

    X_train = np.random.randn(n_train, n_features) * 5 + 10
    X_val = np.random.randn(n_val, n_features) * 2 + 50  # Shifted distribution

    # Fold-local fitting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    assert X_train_pca.shape[1] == X_val_pca.shape[1]
    assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-5)
    # Validation data should NOT have zero mean because scaler was fit on train only
    assert not np.allclose(X_val_scaled.mean(axis=0), 0.0, atol=1e-2)


def test_pca_notebook_scaffold_integrity():
    """Verify that 04a_pca_analysis.ipynb exists as an unexecuted scaffold with all 25 sections."""
    assert PCA_NOTEBOOK_PATH.exists(), f"Missing PCA notebook: {PCA_NOTEBOOK_PATH}"

    with open(PCA_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert nb_data.get("nbformat") == 4

    # Extract all markdown headings
    headings = []
    for cell in nb_data["cells"]:
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            for line in source.split("\n"):
                if line.startswith("#"):
                    headings.append(line.strip())
        elif cell.get("cell_type") == "code":
            # Must remain unexecuted in scaffold
            assert cell.get("execution_count") is None, "Scaffold code cell must not be executed."
            assert len(cell.get("outputs", [])) == 0, "Scaffold code cell must have empty outputs."

    # Verify key structural headings
    expected_headings = [
        "# 04a — Principal Component Analysis (PCA)",
        "## 1. Purpose",
        "## 2. PCA Methodology and Leakage Rules",
        "## 3. Imports and Setup",
        "## 4. Load Dataset and Split Registry",
        "## 5. Load Feature Contract",
        "## 6. Select Development Partition",
        "## 7. Select the 13 Numerical / Ordinal Predictors",
        "## 8. Validate PCA Inputs",
        "## 9. Descriptive Statistics",
        "## 10. Standardize Features",
        "## 11. Fit PCA",
        "## 12. Explained Variance",
        "## 13. Cumulative Explained Variance",
        "## 14. Components Required for 80%, 90%, and 95% Variance",
        "## 15. Scree Plot",
        "## 16. Cumulative Variance Plot",
        "## 17. Principal Component Loadings",
        "## 18. Interpret PC1, PC2, and PC3",
        "## 19. PC1 vs PC2 Visualization",
        "## 20. Logistic Regression Without PCA",
        "## 21. Logistic Regression With PCA >=95%",
        "## 22. Five-Fold Cross-Validation Comparison",
        "## 23. Compare PR-AUC, ROC-AUC, and Brier Score",
        "## 24. Save PCA Results",
        "## 25. PCA Conclusion",
    ]

    for expected in expected_headings:
        assert any(expected in h for h in headings), f"Missing expected heading: {expected}"
