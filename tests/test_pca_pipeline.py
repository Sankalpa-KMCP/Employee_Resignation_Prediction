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
    """Verify that 04a_pca_analysis.ipynb has all 25 sections and matches approved execution state.

    The first 23 code cells are executed without errors, while the 24th code cell
    (final verification assertions) remains unexecuted.
    Also verifies all 25 logical PCA sections and generated canonical PCA CSV artifacts.
    """
    assert PCA_NOTEBOOK_PATH.exists(), f"Missing PCA notebook: {PCA_NOTEBOOK_PATH}"

    with open(PCA_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert nb_data.get("nbformat") == 4

    # Extract all code cells and all cell text sources
    code_cells = [cell for cell in nb_data["cells"] if cell.get("cell_type") == "code"]
    all_cell_sources = ["".join(cell.get("source", [])) for cell in nb_data["cells"]]

    # Verify code cells count and approved execution state:
    # 24 code cells total: all 24 executed without errors
    assert len(code_cells) == 24, f"Expected exactly 24 code cells in PCA notebook, found {len(code_cells)}"

    # All 24 code cells must be executed and contain no error outputs
    for idx, cell in enumerate(code_cells):
        assert cell.get("execution_count") is not None, f"Code cell {idx + 1} must have an execution_count"
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"Code cell {idx + 1} contains an execution error: "
                f"{output.get('ename')}: {output.get('evalue')}"
            )

    # Final code cell (cell 24, index 23) is the final verification block
    final_cell = code_cells[23]
    final_cell_source = "".join(final_cell.get("source", []))
    assert "Final checks after all PCA experiments" in final_cell_source, (
        "24th code cell must be the final verification block"
    )

    # Verify all 25 logical PCA sections are present using real notebook headings and comments
    expected_sections = [
        "Purpose",
        "PCA Methodology and Leakage Rules",
        "Imports and setup",
        "Load data and split registry",
        "Select development partition only",
        "Select PCA features",
        "Validate PCA input data",
        "Standardize PCA features",
        "Fit PCA",
        "Explained variance",
        "Components needed for variance thresholds",
        "Scree plot",
        "Cumulative explained variance plot",
        "Principal component loadings",
        "Interpret PC1, PC2, and PC3",
        "PC1 vs PC2 visualization",
        "Prepare target and CV folds",
        "Logistic Regression WITHOUT PCA",
        "Logistic Regression WITH PCA",
        "Compare Logistic Regression metrics",
        "Save PCA results",
        "PCA Conclusion",
        "Final PCA notebook checks",
        "Logistic Regression WITH PCA (80% variance)",
        "Compare all three Logistic Regression versions",
    ]
    assert len(expected_sections) == 25, f"Expected exactly 25 logical sections, got {len(expected_sections)}"

    for section in expected_sections:
        assert any(section in src for src in all_cell_sources), f"Missing expected PCA section: {section!r}"

    # Verify canonical PCA artifacts generated by the executed notebook
    expected_artifacts = [
        "explained_variance.csv",
        "pca_loadings.csv",
        "logistic_regression_pca_comparison.csv",
        "pca_summary.csv",
    ]
    for filename in expected_artifacts:
        artifact_path = RESULTS_PCA_DIR / filename
        assert artifact_path.exists(), f"Missing PCA artifact: {artifact_path}"
        assert artifact_path.stat().st_size > 0, f"PCA artifact must not be empty: {artifact_path}"

