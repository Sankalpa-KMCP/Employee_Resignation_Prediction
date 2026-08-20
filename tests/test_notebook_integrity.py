"""Unit tests for notebook structure, JSON validity, and naming sequence."""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

EXPECTED_NOTEBOOKS = [
    "01_data_audit.ipynb",
    "02_create_split_registry.ipynb",
    "03_eda.ipynb",
    "04_baselines.ipynb",
    "04a_pca_analysis.ipynb",
    "05_member_1/05_m1_ft_transformer.ipynb",
    "06_member_2/06_m2_catboost.ipynb",
    "07_member_3/07_m3_xgboost.ipynb",
    "08_member_4/08_m4_lightgbm.ipynb",
    "09_member_5/09_m5_ebm.ipynb",
    "10_member_6/10_m6_tabnet.ipynb",
    "11_model_comparison.ipynb",
    "12_calibration.ipynb",
    "13_fairness_analysis.ipynb",
    "14_retention_prioritization.ipynb",
    "15_final_holdout_evaluation.ipynb",
]

EXPECTED_MEMBER_SUBDIRS = {
    "05_member_1",
    "06_member_2",
    "07_member_3",
    "08_member_4",
    "09_member_5",
    "10_member_6",
}


@pytest.mark.parametrize("nb_rel_path", EXPECTED_NOTEBOOKS)
def test_notebook_exists_and_is_valid_json(nb_rel_path: str):
    """Verify that every expected notebook exists and is a valid Jupyter notebook."""
    nb_path = NOTEBOOKS_DIR / nb_rel_path
    assert nb_path.exists(), f"Notebook {nb_rel_path} does not exist in {NOTEBOOKS_DIR}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data, f"Notebook {nb_rel_path} missing 'cells' array."
    assert "metadata" in nb_data, f"Notebook {nb_rel_path} missing 'metadata'."
    assert "nbformat" in nb_data, f"Notebook {nb_rel_path} missing 'nbformat'."
    assert len(nb_data["cells"]) > 0, f"Notebook {nb_rel_path} has 0 cells."


def test_member_subdirectories_in_notebooks():
    """Verify notebooks directory has exactly the expected member folders and no obsolete subdirectories."""
    subdirs = {p.name for p in NOTEBOOKS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")}
    assert subdirs == EXPECTED_MEMBER_SUBDIRS, f"Notebook subdirectories mismatch. Expected {EXPECTED_MEMBER_SUBDIRS}, found {subdirs}"
