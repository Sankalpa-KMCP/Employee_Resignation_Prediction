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
    "team_contributions/models/member_1_ft_transformer/05_m1_ft_transformer.ipynb",
    "team_contributions/models/member_2_catboost/06_m2_catboost.ipynb",
    "team_contributions/models/member_3_xgboost/07_m3_xgboost.ipynb",
    "team_contributions/models/member_4_lightgbm/08_m4_lightgbm.ipynb",
    "team_contributions/models/member_5_ebm/09_m5_ebm.ipynb",
    "team_contributions/models/member_6_tabnet/10_m6_tabnet.ipynb",
    "group/01_group_pipeline.ipynb",
    "group/02_model_comparison.ipynb",
    "group/03_calibration.ipynb",
    "group/04_fairness_analysis.ipynb",
    "group/05_retention_prioritization.ipynb",
    "group/06_final_holdout_evaluation.ipynb",
]

EXPECTED_TOP_LEVEL_SUBDIRS = {"team_contributions", "group"}
EXPECTED_MODEL_SUBDIRS = {
    "member_1_ft_transformer",
    "member_2_catboost",
    "member_3_xgboost",
    "member_4_lightgbm",
    "member_5_ebm",
    "member_6_tabnet",
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


def test_notebook_architecture_subdirectories():
    """Verify the refactored group/team-contribution notebook architecture."""
    top_level_subdirs = {
        path.name
        for path in NOTEBOOKS_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    assert top_level_subdirs == EXPECTED_TOP_LEVEL_SUBDIRS

    model_dir = NOTEBOOKS_DIR / "team_contributions" / "models"
    model_subdirs = {
        path.name
        for path in model_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    assert model_subdirs == EXPECTED_MODEL_SUBDIRS
