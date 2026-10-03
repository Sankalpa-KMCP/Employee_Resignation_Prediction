"""Load and query the canonical feature contract.

This module provides the Python interface for accessing the project's frozen
feature contract defined in `configs/feature_contract.yaml`.

Architectural Purpose & Why It Matters:
---------------------------------------
In a multi-model machine learning benchmark (e.g., comparing FT-Transformer,
CatBoost, XGBoost, LightGBM, EBM, and TabNet), consistency across data splits,
features, and target definitions is critical.

This feature contract guarantees:
1. Leakage Prevention: Excludes identifier fields (`Employee_ID`), ground truth
   targets (`Resigned`), and redundant/proxy features (`Hire_Date`) from predictor matrices.
2. Single Source of Truth: All 6 headline models and baseline pipelines query this
   module to determine approved numerical and categorical predictors rather than
   hardcoding feature lists.
3. Reproducibility: Validates that the YAML contract exists and conforms to the
   expected mapping schema before any pipeline attempts to train.
"""

from __future__ import annotations

from pathlib import Path
import yaml

# Resolve the absolute path to the default feature contract YAML file.
# Path(__file__).resolve() gives the absolute path to this script (src/data/feature_contract.py).
# .parents[2] navigates up two directory levels to the repository root:
#   parents[0] -> src/data/
#   parents[1] -> src/
#   parents[2] -> project root (e.g., d:/Employee_Resignation_Prediction)
# This ensures path resolution works reliably regardless of the current working directory.
DEFAULT_CONTRACT = Path(__file__).resolve().parents[2] / "configs" / "feature_contract.yaml"


def load_feature_contract(path: str | Path = DEFAULT_CONTRACT) -> dict:
    """Read the YAML feature contract as a mapping; reject missing files or non-mapping content.

    The returned contract defines approved predictors, target, record identifier,
    split registry rules, and preprocessing boundaries.

    Parameters
    ----------
    path : str | Path, optional
        Filesystem path to the feature contract YAML file.
        Defaults to `configs/feature_contract.yaml`.

    Returns
    -------
    dict
        Parsed YAML contents containing the frozen feature specifications.

    Raises
    ------
    FileNotFoundError
        If the contract file does not exist at the specified path.
    ValueError
        If the parsed YAML content is not a dictionary/mapping.
    """
    # Convert input to a Path object to ensure cross-platform path handling
    contract_path = Path(path)

    # 1. Existence check: Ensure the contract file exists on disk before reading
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)

    # 2. Parsing: Safely parse the YAML content using UTF-8 encoding
    data = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

    # 3. Schema check: Enforce that the top-level structure is a valid dictionary/mapping
    if not isinstance(data, dict):
        raise ValueError("Feature contract must be a YAML mapping.")

    return data


def numerical_feature_names(contract: dict | None = None) -> list[str]:
    """Return numerical predictor names in contract order, loading the default contract if omitted.

    Extracts the names of all approved continuous and ordinal numerical features
    (e.g., Age, Monthly_Salary, Performance_Score, Work_Hours_Per_Week) defined
    under the `numerical_features` section of the contract.

    Parameters
    ----------
    contract : dict | None, optional
        Pre-loaded feature contract dictionary. If None, the contract will be
        automatically loaded from disk via `load_feature_contract()`.

    Returns
    -------
    list[str]
        Ordered list of approved numerical feature column names.
    """
    # Reuse provided contract dictionary or load the canonical contract from disk
    active_contract = contract or load_feature_contract()

    # Extract the 'name' attribute from each entry in the 'numerical_features' list
    # Defaults to an empty list if the key is not found
    return [feature["name"] for feature in active_contract.get("numerical_features", [])]


def categorical_feature_names(contract: dict | None = None) -> list[str]:
    """Return categorical predictor names in contract order, loading the default contract if omitted.

    Extracts the names of all approved nominal categorical features
    (e.g., Department, Gender, Job_Title, Education_Level) defined
    under the `categorical_features` section of the contract.

    Parameters
    ----------
    contract : dict | None, optional
        Pre-loaded feature contract dictionary. If None, the contract will be
        automatically loaded from disk via `load_feature_contract()`.

    Returns
    -------
    list[str]
        Ordered list of approved categorical feature column names.
    """
    # Reuse provided contract dictionary or load the canonical contract from disk
    active_contract = contract or load_feature_contract()

    # Extract the 'name' attribute from each entry in the 'categorical_features' list
    # Defaults to an empty list if the key is not found
    return [feature["name"] for feature in active_contract.get("categorical_features", [])]
