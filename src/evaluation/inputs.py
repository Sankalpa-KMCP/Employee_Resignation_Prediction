"""Read the frozen development data and validate saved inputs for group notebooks."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from src.evaluation.oof_validation import validate_oof_predictions
from src.evaluation.metrics import calculate_binary_metrics, validate_binary_targets


def load_development_data(root, *, usecols=None):
    """Return development rows and the frozen split, without parsing holdout targets.

    Verify both source files against the feature contract. Identifier-only inspection
    determines which rows to skip before reading predictors and labels.
    """
    root = Path(root)
    contract = yaml.safe_load((root / "configs/feature_contract.yaml").read_text(encoding="utf-8"))
    for key in ("dataset", "split_registry"):
        path = root / contract[key]["path"]
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest != contract[key]["sha256"]:
            raise ValueError(f"Frozen {key} checksum mismatch.")
    split = pd.read_csv(root / contract["split_registry"]["path"])
    if split.Employee_ID.isna().any() or not split.Employee_ID.is_unique:
        raise ValueError("Frozen registry IDs must be present and unique.")
    development = split.loc[split.partition == "development"]
    if len(development) != contract["split_registry"]["development_rows"]:
        raise ValueError("Frozen development row count mismatch.")
    path = root / contract["dataset"]["path"]
    identifiers = pd.read_csv(path, usecols=["Employee_ID"])
    ids = set(development.Employee_ID)
    skip = [i + 1 for i, value in enumerate(identifiers.Employee_ID) if value not in ids]
    columns = None if usecols is None else list(dict.fromkeys(["Employee_ID", "Resigned", *usecols]))
    data = pd.read_csv(path, skiprows=skip, usecols=columns)
    if not data.Employee_ID.is_unique or set(data.Employee_ID) != ids:
        raise ValueError("Source development IDs do not match the frozen registry.")
    validate_binary_targets(data.Resigned.to_numpy(), name="Resigned")
    data["Resigned"] = data.Resigned.astype(int)
    print(f"Rows before/after development isolation: {len(identifiers):,} -> {len(data):,}; "
          f"Resigned prevalence={data.Resigned.mean():.5f} (imbalanced; no resampling during saved review).")
    return data, split


def load_group_inputs(root):
    """Return the analysis registry and all validated, ID-sorted model OOF frames.

    Every model must match frozen development IDs, folds, source targets and
    probability rules. EBM additionally retains its config-bound manifest check.
    Missing or invalid evidence raises an error; there is no historical fallback.
    """
    root = Path(root)
    registry = yaml.safe_load((root / "configs/analysis_registry.yaml").read_text(encoding="utf-8"))
    source, split = load_development_data(root, usecols=[])
    frames = {}
    for model, entry in registry["models"].items():
        if model == "EBM":
            from src.models.ebm_experiments import load_formal_ebm_input
            from src.models.config import load_model_config
            config = load_model_config(root / "configs/models/ebm.yaml", strict_schema=True)
            if entry["oof_path"] != config["canonical_results"]["oof_predictions"]:
                raise ValueError("Group EBM registry must point to the configured formal OOF input.")
            predictions, _, _ = load_formal_ebm_input(root)
        else:
            predictions = pd.read_csv(root / entry["oof_path"])
        validate_oof_predictions(predictions, split, source_data=source,
                                 expected_model_name=entry.get("model_name", model))
        frames[model] = predictions.sort_values("Employee_ID").reset_index(drop=True)
    return registry, frames


def load_saved_model_evidence(root, model, directory, *, summary_file="summary_metrics.json"):
    """Validate a completed model's OOF and summary for read-only notebook review.

    Recompute pooled AP, ROC-AUC and Brier from source-validated predictions.
    EBM also checks the formal manifest; this function never fits or writes files.
    """
    root, directory = Path(root), Path(directory)
    source, split = load_development_data(root, usecols=[])
    if model == "EBM":
        from src.models.ebm_experiments import CONFIG, load_formal_ebm_input
        if directory.resolve() != (root / CONFIG["canonical_results"]["directory"]).resolve():
            raise ValueError("Saved EBM review requires the configured formal directory.")
        predictions, folds, summary = load_formal_ebm_input(root)
    else:
        predictions = pd.read_csv(directory / "oof_predictions.csv")
        folds = pd.read_csv(directory / "fold_metrics.csv")
        summary = json.loads((directory / summary_file).read_text(encoding="utf-8"))
    expected_name = "CatBoost Classifier" if model == "CatBoost" else model
    validate_oof_predictions(predictions, split, source_data=source, expected_model_name=expected_name)
    actual = calculate_binary_metrics(predictions.y_true, predictions.y_probability)
    for metric in ("average_precision", "roc_auc", "brier_score"):
        key = "pooled_" + metric if "pooled_" + metric in summary else metric
        if key not in summary or not np.isclose(actual[metric], summary[key], rtol=0, atol=1e-9):
            raise ValueError(f"Saved {model} {metric} disagrees with its OOF predictions.")
    return predictions, folds, summary
