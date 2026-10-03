"""Development-only loading and formal output handling for Member 4 LightGBM.

This module orchestrates an already-fitted strict-nested run. Candidate selection,
early stopping, and outer-fold fitting remain in ``LightGBMAdapter`` and the
shared five-fold runner.

The Member 4 notebook uses this layer to load development inputs and export formal evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_binary_metrics
from src.evaluation.oof_validation import validate_oof_predictions
from src.models.base import OOFResult


FORMAL_CANDIDATE_IDS = tuple(f"LGBM_F{number:02d}" for number in range(1, 9))
FORMAL_FILE_ROLES = {
    "oof_predictions.csv": "formal_oof_predictions",
    "fold_metrics.csv": "formal_fold_metrics",
    "summary_metrics.json": "formal_summary_metrics",
    "fold_selection.csv": "formal_fold_selection",
    "candidate_selection.csv": "formal_inner_candidate_evidence",
    "runtime_efficiency.csv": "formal_runtime_evidence",
    "run_metadata.json": "formal_run_provenance",
}


def _json_safe(value: Any) -> Any:
    """Convert metadata to strict JSON values without changing finite numbers."""
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _strict_json(value: Any) -> str:
    """Serialize metadata with non-finite numbers replaced by null, rejecting non-standard JSON NaN."""
    return json.dumps(_json_safe(value), indent=2, allow_nan=False)


def build_lightgbm_output_manifest(output_dir: Path, project_root: Path) -> pd.DataFrame:
    """Inventory all eight artifacts; a manifest cannot hash or size itself."""
    output_dir = Path(output_dir)
    project_root = Path(project_root)
    rows = [
        {
            "artifact": (output_dir / name).relative_to(project_root).as_posix(),
            "exists": (output_dir / name).is_file(),
            "size_bytes": (output_dir / name).stat().st_size,
            "sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
            "role": role,
            "git_tracking": "review_before_commit",
            "checksum_note": "",
        }
        for name, role in FORMAL_FILE_ROLES.items()
    ]
    rows.append({
        "artifact": (output_dir / "output_manifest.csv").relative_to(project_root).as_posix(),
        "exists": True,
        "size_bytes": None,
        "sha256": None,
        "role": "formal_output_manifest",
        "git_tracking": "review_before_commit",
        "checksum_note": "self-reference: checksum and size intentionally omitted",
    })
    manifest = pd.DataFrame(rows)
    if not manifest["artifact"].is_unique:
        raise ValueError("Formal output manifest contains duplicate paths.")
    return manifest


def load_lightgbm_development_only(
    raw_path: Path,
    split_registry: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Read approved columns for development rows without parsing holdout values.

    The first CSV pass reads Employee_ID only. Its record positions determine
    which rows the second pass parses; ``skiprows`` excludes holdout records
    before their predictors or target can enter a DataFrame.
    """
    raw_path = Path(raw_path)
    if len(feature_cols) != 17 or len(set(feature_cols)) != 17:
        raise ValueError("The formal LightGBM workflow requires 17 distinct predictors.")
    required_registry = {"Employee_ID", "partition", "cv_fold"}
    if not required_registry.issubset(split_registry.columns):
        raise ValueError("Split registry is missing required columns.")
    if len(split_registry) != 100_000 or not split_registry["Employee_ID"].is_unique:
        raise ValueError("The frozen split registry must contain 100,000 unique IDs.")

    development = split_registry.loc[split_registry["partition"].eq("development")]
    holdout = split_registry.loc[split_registry["partition"].eq("holdout")]
    if len(development) != 80_000 or len(holdout) != 20_000:
        raise ValueError("Expected 80,000 development and 20,000 holdout registry IDs.")
    dev_ids = set(development["Employee_ID"])
    holdout_ids = set(holdout["Employee_ID"])
    if dev_ids & holdout_ids:
        raise ValueError("Development and holdout registry IDs overlap.")
    if development["cv_fold"].isna().any() or (
        development["cv_fold"].astype(int).value_counts().sort_index().to_dict()
        != {fold: 16_000 for fold in range(1, 6)}
    ):
        raise ValueError("Frozen development fold assignments are invalid.")

    raw_ids = pd.read_csv(raw_path, usecols=["Employee_ID"])["Employee_ID"]
    if len(raw_ids) != 100_000 or not raw_ids.is_unique:
        raise ValueError("Raw CSV must contain 100,000 unique Employee_ID values.")
    if set(raw_ids) != set(split_registry["Employee_ID"]):
        raise ValueError("Raw Employee_ID values do not match the frozen registry.")

    # CSV row zero is the header; data record at zero-based index i is row i+1.
    development_csv_rows = set(np.flatnonzero(raw_ids.isin(dev_ids).to_numpy()) + 1)
    if len(development_csv_rows) != 80_000:
        raise ValueError("Development row positions do not match the registry.")
    columns = ["Employee_ID", *feature_cols, "Resigned"]
    loaded = pd.read_csv(
        raw_path,
        usecols=columns,
        skiprows=lambda csv_row: csv_row > 0 and csv_row not in development_csv_rows,
    )
    if set(loaded.columns) != set(columns) or len(loaded) != 80_000:
        raise ValueError("Development-only CSV read returned unexpected columns or rows.")
    if not loaded["Employee_ID"].is_unique or set(loaded["Employee_ID"]) != dev_ids:
        raise ValueError("Loaded development IDs do not match the frozen registry.")
    if set(loaded["Employee_ID"]) & holdout_ids:
        raise ValueError("Holdout IDs entered the formal modeling data.")
    return loaded


def validate_lightgbm_formal_outputs(
    predictions: pd.DataFrame,
    candidate_selection: pd.DataFrame,
    fold_selection: pd.DataFrame,
    split_registry: pd.DataFrame,
    development: pd.DataFrame,
    *,
    expected_rows: int = 80_000,
    expected_fold_rows: int = 16_000,
) -> None:
    """Check strict-nested output boundaries before any formal files are saved."""
    validate_oof_predictions(
        predictions=predictions,
        split_registry=split_registry,
        source_data=development,
        expected_model_name="LightGBM",
    )
    if len(predictions) != expected_rows or predictions["Employee_ID"].nunique() != expected_rows:
        raise ValueError("Formal OOF row or unique-ID count is incorrect.")
    if predictions["cv_fold"].value_counts().sort_index().to_dict() != {
        fold: expected_fold_rows for fold in range(1, 6)
    }:
        raise ValueError("Formal OOF fold sizes are incorrect.")
    if len(development) != expected_rows or not development["Employee_ID"].is_unique:
        raise ValueError("Development source does not contain the expected unique IDs.")
    dev_ids = set(split_registry.loc[split_registry["partition"].eq("development"), "Employee_ID"])
    holdout_ids = set(split_registry.loc[split_registry["partition"].eq("holdout"), "Employee_ID"])
    if set(development["Employee_ID"]) != dev_ids or set(development["Employee_ID"]) & holdout_ids:
        raise ValueError("Development source violates partition isolation.")

    required_candidates = {
        "outer_fold", "candidate_id", "inner_average_precision",
        "configured_n_estimators", "best_iteration", "class_weight",
        "runtime_seconds", "success", "failure_reason",
    }
    required_folds = {
        "outer_fold", "outer_train_rows", "outer_validation_rows",
        "inner_train_rows", "inner_validation_rows", "selected_candidate_id",
        "selected_inner_average_precision", "selected_best_iteration",
        "selected_class_weight", "winner_refit_rows", "fold_runtime_seconds",
    }
    if not required_candidates.issubset(candidate_selection.columns):
        raise ValueError("Candidate selection is missing required columns.")
    if not required_folds.issubset(fold_selection.columns):
        raise ValueError("Fold selection is missing required columns.")
    if len(candidate_selection) != 40 or candidate_selection.duplicated(["outer_fold", "candidate_id"]).any():
        raise ValueError("Expected 40 unique candidate evaluation records.")
    if len(fold_selection) != 5 or set(fold_selection["outer_fold"]) != set(range(1, 6)):
        raise ValueError("Expected one fold-selection record for each outer fold.")
    for fold in range(1, 6):
        candidates = candidate_selection.loc[candidate_selection["outer_fold"].eq(fold)]
        if set(candidates["candidate_id"]) != set(FORMAL_CANDIDATE_IDS):
            raise ValueError(f"Fold {fold} does not record all eight formal candidates.")
        if candidates["success"].isna().any() or not candidates["success"].isin([True, False]).all():
            raise ValueError(f"Fold {fold} has an invalid candidate success flag.")
        failed = candidates.loc[candidates["success"].eq(False)]
        if failed["failure_reason"].isna().any() or failed["failure_reason"].astype(str).str.strip().eq("").any():
            raise ValueError(f"Fold {fold} has an unrecorded candidate failure.")
        successful = candidates.loc[candidates["success"].eq(True)]
        if successful.empty or not np.isfinite(pd.to_numeric(successful["inner_average_precision"], errors="coerce")).all():
            raise ValueError(f"Fold {fold} has no valid successful candidate AP.")
        winner = successful.sort_values(
            ["inner_average_precision", "candidate_id"], ascending=[False, True]
        ).iloc[0]
        selected = fold_selection.loc[fold_selection["outer_fold"].eq(fold)].iloc[0]
        if selected["selected_candidate_id"] != winner["candidate_id"] or not np.isclose(
            selected["selected_inner_average_precision"], winner["inner_average_precision"]
        ) or int(selected["selected_best_iteration"]) != int(winner["best_iteration"]):
            raise ValueError(f"Fold {fold} winner does not match inner AP and tie policy.")
        if int(selected["outer_validation_rows"]) != expected_fold_rows or int(selected["outer_train_rows"]) != expected_rows - expected_fold_rows:
            raise ValueError(f"Fold {fold} outer row counts are incorrect.")
        if int(selected["winner_refit_rows"]) != int(selected["outer_train_rows"]):
            raise ValueError(f"Fold {fold} winner was not refitted on all outer training rows.")
        if int(selected["inner_train_rows"]) + int(selected["inner_validation_rows"]) != int(selected["outer_train_rows"]):
            raise ValueError(f"Fold {fold} inner split does not cover outer training rows.")


def write_lightgbm_formal_artifacts(
    result: OOFResult,
    adapters: Sequence[Any],
    development: pd.DataFrame,
    split_registry: pd.DataFrame,
    config: dict[str, Any],
    project_root: Path,
    output_dir: Path,
    source_head: str,
    *,
    expected_rows: int = 80_000,
    expected_fold_rows: int = 16_000,
) -> dict[str, Any]:
    """Validate a completed run, then write the eight distinct formal outputs."""
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Formal output directory already exists: {output_dir}")
    if len(adapters) != 5:
        raise ValueError("Expected five fitted outer-fold adapters.")
    candidate_ids = [candidate["id"] for candidate in config["candidates"]]
    if candidate_ids != list(FORMAL_CANDIDATE_IDS):
        raise ValueError("Formal candidate registry differs from LGBM_F01–F08.")

    fold_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for adapter in adapters:
        metadata = adapter.get_metadata()
        fold = int(metadata["outer_fold"])
        fit = metadata["fit_summary"]
        fold_metrics = result.fold_metrics.loc[result.fold_metrics["fold"].eq(fold)]
        if len(fold_metrics) != 1:
            raise ValueError(f"Missing fold metrics for fold {fold}.")
        metrics = fold_metrics.iloc[0]
        fold_rows.append({
            "outer_fold": fold,
            "outer_train_rows": fit["outer_training_row_count"],
            "outer_validation_rows": int(result.predictions["cv_fold"].eq(fold).sum()),
            "inner_train_rows": fit["inner_training_row_count"],
            "inner_validation_rows": fit["inner_validation_row_count"],
            "selected_candidate_id": fit["selected_candidate_id"],
            "selected_inner_average_precision": fit["selected_inner_average_precision"],
            "selected_best_iteration": fit["selected_best_iteration"],
            "selected_class_weight": fit["selected_class_weight"],
            "winner_refit_rows": fit["refit_row_count"],
            "candidate_selection_time_seconds": fit["selection_time_seconds"],
            "winner_refit_time_seconds": fit["refit_time_seconds"],
            "fit_time_seconds": fit["fit_time_seconds"],
            "prediction_time_seconds": float(metrics["predict_time_seconds"]),
            "fold_runtime_seconds": float(fit["fit_time_seconds"] + metrics["predict_time_seconds"]),
        })
        for record in metadata["candidate_selection_records"]:
            candidate_rows.append({
                "outer_fold": fold,
                "candidate_id": record["candidate_id"],
                "inner_average_precision": record.get("inner_average_precision"),
                "configured_n_estimators": record["configured_n_estimators"],
                "best_iteration": record.get("selected_best_iteration"),
                "class_weight": record["class_weight"],
                "runtime_seconds": record["runtime_seconds"],
                "success": record["success"],
                "failure_reason": record.get("failure_reason"),
                "candidate_parameters": json.dumps(record["candidate_parameters"], sort_keys=True),
                "resolved_parameters": json.dumps(record["resolved_parameters"], sort_keys=True),
                "inner_train_rows": record["inner_train_row_count"],
                "inner_validation_rows": record["inner_validation_row_count"],
            })

    fold_selection = pd.DataFrame(fold_rows).sort_values("outer_fold").reset_index(drop=True)
    candidate_selection = pd.DataFrame(candidate_rows).sort_values(
        ["outer_fold", "candidate_id"]
    ).reset_index(drop=True)
    predictions = result.predictions.sort_values(["cv_fold", "Employee_ID"]).reset_index(drop=True)
    fold_metrics = result.fold_metrics.sort_values("fold").reset_index(drop=True)
    validate_lightgbm_formal_outputs(
        predictions, candidate_selection, fold_selection, split_registry, development,
        expected_rows=expected_rows, expected_fold_rows=expected_fold_rows,
    )
    if len(fold_metrics) != 5 or set(fold_metrics["fold"]) != set(range(1, 6)):
        raise ValueError("Expected exactly five outer-fold metric records.")

    summary = dict(result.summary_metrics)
    pooled = calculate_binary_metrics(
        predictions["y_true"],
        predictions["y_probability"],
        predictions.get("y_pred_0_5"),
        threshold=0.50,
    )
    for pooled_key, pooled_value in pooled.items():
        summary_key = f"pooled_{pooled_key}"
        if summary_key in summary and not np.isclose(summary[summary_key], pooled_value):
            raise ValueError(f"{summary_key} does not match the formal OOF predictions.")
        summary[summary_key] = pooled_value
    if "pooled_pr_auc" in summary and not np.isclose(
        summary["pooled_pr_auc"], pooled["average_precision"]
    ):
        raise ValueError("pooled_pr_auc does not match the formal OOF predictions.")
    summary.update({
        "result_label": "FORMAL STRICT-NESTED LIGHTGBM DEVELOPMENT OOF",
        "config_version": config["model"]["config_version"],
        "candidate_evaluations": len(candidate_selection),
        "outer_winner_refits": len(fold_selection),
        "formal_holdout_used": False,
        "formal_holdout_overlap": 0,
    })
    runtime = fold_selection[[
        "outer_fold", "selected_candidate_id", "candidate_selection_time_seconds",
        "winner_refit_time_seconds", "fit_time_seconds", "prediction_time_seconds",
        "fold_runtime_seconds",
    ]].copy()
    metadata = {
        "result_label": summary["result_label"],
        "model": "LightGBM",
        "config_version": config["model"]["config_version"],
        "candidate_registry_version": config["model"]["candidate_registry_version"],
        "candidate_ids": candidate_ids,
        "candidate_definitions": config["candidates"],
        "source_head": source_head,
        "split_registry": "data/processed/split_registry.csv",
        "development_rows": expected_rows,
        "outer_folds": 5,
        "inner_fraction": config["inner_validation"]["fraction"],
        "seed": config["inner_validation"]["random_seed"],
        "stratified": config["inner_validation"]["stratified"],
        "selection_metric": config["inner_validation"]["metric"],
        "tie_breaker": config["inner_validation"]["tie_breaker"],
        "early_stopping_rounds": config["inner_validation"]["early_stopping_rounds"],
        "refit_policy": "full outer training at selected best iteration without early stopping",
        "outer_validation_role": "prediction only after candidate selection and refit",
        "fold_winners": fold_selection[[
            "outer_fold", "selected_candidate_id", "selected_best_iteration",
            "selected_class_weight", "selected_inner_average_precision",
        ]].to_dict(orient="records"),
        "candidate_evaluations": len(candidate_selection),
        "winner_refits": len(fold_selection),
        "total_fit_time_seconds": result.fit_time_seconds_total,
        "total_predict_time_seconds": result.predict_time_seconds_total,
        "formal_holdout_used": False,
        "formal_holdout_feature_or_target_values_loaded": False,
        "formal_holdout_rows_used_for_modeling": 0,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
    }
    try:
        import lightgbm
        import sklearn
        metadata["lightgbm_version"] = lightgbm.__version__
        metadata["scikit_learn_version"] = sklearn.__version__
    except ImportError:
        pass

    output_dir.mkdir(parents=True)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "summary_metrics.json").write_text(_strict_json(summary), encoding="utf-8")
    fold_selection.to_csv(output_dir / "fold_selection.csv", index=False)
    candidate_selection.to_csv(output_dir / "candidate_selection.csv", index=False)
    runtime.to_csv(output_dir / "runtime_efficiency.csv", index=False)
    metadata = _json_safe(metadata)
    (output_dir / "run_metadata.json").write_text(_strict_json(metadata), encoding="utf-8")
    manifest = build_lightgbm_output_manifest(output_dir, project_root)
    manifest.to_csv(output_dir / "output_manifest.csv", index=False)
    return {
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "summary_metrics": summary,
        "fold_selection": fold_selection,
        "candidate_selection": candidate_selection,
        "runtime_efficiency": runtime,
        "run_metadata": metadata,
        "output_manifest": manifest,
    }
