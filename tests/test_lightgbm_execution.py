"""Execution-layer checks that never train the formal LightGBM candidates."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.base import OOFResult
from src.models.config import load_model_config
from src.models.lightgbm_execution import (
    FORMAL_CANDIDATE_IDS,
    FORMAL_FILE_ROLES,
    _strict_json,
    load_lightgbm_development_only,
    validate_lightgbm_formal_outputs,
    write_lightgbm_formal_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Extended_Employee_Performance_and_Productivity_Data.csv"
REGISTRY_PATH = ROOT / "data" / "processed" / "split_registry.csv"
NOTEBOOK_PATH = ROOT / "notebooks" / "team_contributions" / "models" / "member_4_lightgbm" / "08_m4_lightgbm.ipynb"
CONFIG_PATH = ROOT / "configs" / "models" / "lightgbm.yaml"


def test_real_loader_reads_only_development_model_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    if not RAW_PATH.exists():
        pytest.skip("The ignored raw dataset is not available in this checkout.")
    registry = pd.read_csv(REGISTRY_PATH, dtype={"cv_fold": "Int64"})
    with (ROOT / "configs" / "feature_contract.yaml").open(encoding="utf-8") as handle:
        features = yaml.safe_load(handle)["primary_model_features"]
    calls: list[dict] = []
    original_read_csv = pd.read_csv

    def observed_read_csv(*args, **kwargs):
        if args and Path(args[0]) == RAW_PATH:
            calls.append(kwargs.copy())
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", observed_read_csv)
    development = load_lightgbm_development_only(RAW_PATH, registry, features)
    dev_ids = set(registry.loc[registry["partition"].eq("development"), "Employee_ID"])
    holdout_ids = set(registry.loc[registry["partition"].eq("holdout"), "Employee_ID"])

    assert len(calls) == 2
    assert calls[0]["usecols"] == ["Employee_ID"]
    assert set(calls[1]["usecols"]) == {"Employee_ID", "Resigned", *features}
    assert callable(calls[1]["skiprows"])
    assert calls[1]["skiprows"](0) is False
    raw_ids = original_read_csv(RAW_PATH, usecols=["Employee_ID"])["Employee_ID"]
    first_dev_row = int(np.flatnonzero(raw_ids.isin(dev_ids).to_numpy())[0]) + 1
    first_holdout_row = int(np.flatnonzero(raw_ids.isin(holdout_ids).to_numpy())[0]) + 1
    assert calls[1]["skiprows"](first_dev_row) is False
    assert calls[1]["skiprows"](first_holdout_row) is True
    assert len(development) == development["Employee_ID"].nunique() == 80_000
    assert set(development["Employee_ID"]) == dev_ids
    assert set(development["Employee_ID"]).isdisjoint(holdout_ids)
    assert set(features).issubset(development.columns) and len(features) == 17
    assert "Resigned" in development


def tiny_formal_tables():
    ids = [f"D{number:02d}" for number in range(10)]
    registry = pd.DataFrame({
        "Employee_ID": [*ids, "H01", "H02"],
        "partition": ["development"] * 10 + ["holdout"] * 2,
        "cv_fold": pd.array([*(number // 2 + 1 for number in range(10)), pd.NA, pd.NA], dtype="Int64"),
    })
    development = pd.DataFrame({
        "Employee_ID": ids,
        "Resigned": [number % 2 for number in range(10)],
    })
    predictions = pd.DataFrame({
        "Employee_ID": ids,
        "cv_fold": [number // 2 + 1 for number in range(10)],
        "y_true": development["Resigned"].to_numpy(),
        "model_name": ["LightGBM"] * 10,
        "y_probability": np.linspace(0.05, 0.95, 10),
    })
    predictions["y_pred_0_5"] = predictions["y_probability"].ge(0.5).astype(int)
    candidates = pd.DataFrame([
        {
            "outer_fold": fold,
            "candidate_id": candidate_id,
            "inner_average_precision": 0.6 if candidate_id == "LGBM_F01" else 0.5,
            "configured_n_estimators": 600,
            "best_iteration": 17,
            "class_weight": None,
            "runtime_seconds": 0.01,
            "success": True,
            "failure_reason": None,
        }
        for fold in range(1, 6)
        for candidate_id in FORMAL_CANDIDATE_IDS
    ])
    folds = pd.DataFrame([
        {
            "outer_fold": fold,
            "outer_train_rows": 8,
            "outer_validation_rows": 2,
            "inner_train_rows": 7,
            "inner_validation_rows": 1,
            "selected_candidate_id": "LGBM_F01",
            "selected_inner_average_precision": 0.6,
            "selected_best_iteration": 17,
            "selected_class_weight": None,
            "winner_refit_rows": 8,
            "fold_runtime_seconds": 0.2,
        }
        for fold in range(1, 6)
    ])
    return predictions, candidates, folds, registry, development


def validate_tiny(tables) -> None:
    validate_lightgbm_formal_outputs(*tables, expected_rows=10, expected_fold_rows=2)


def test_formal_output_contract_accepts_complete_tiny_evidence() -> None:
    tables = tiny_formal_tables()
    validate_tiny(tables)
    assert len(tables[1]) == 40
    assert len(tables[2]) == 5


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "missing", "holdout", "fold_size", "probability_nan", "probability_inf", "probability_outside", "target"],
)
def test_formal_oof_validation_rejects_corruption(mutation: str) -> None:
    tables = list(tiny_formal_tables())
    predictions = tables[0]
    if mutation == "duplicate":
        predictions.loc[0, "Employee_ID"] = predictions.loc[1, "Employee_ID"]
    elif mutation == "missing":
        predictions.loc[0, "Employee_ID"] = "UNKNOWN"
    elif mutation == "holdout":
        predictions.loc[0, "Employee_ID"] = "H01"
    elif mutation == "fold_size":
        predictions.loc[0, "cv_fold"] = 2
    elif mutation == "probability_nan":
        predictions.loc[0, "y_probability"] = np.nan
    elif mutation == "probability_inf":
        predictions.loc[0, "y_probability"] = np.inf
    elif mutation == "probability_outside":
        predictions.loc[0, "y_probability"] = -0.1
    elif mutation == "target":
        predictions.loc[0, "y_true"] = 1
    with pytest.raises(ValueError):
        validate_tiny(tables)


def test_selection_contract_requires_all_candidates_and_folds() -> None:
    tables = list(tiny_formal_tables())
    tables[1] = tables[1].iloc[:-1].copy()
    with pytest.raises(ValueError, match="40"):
        validate_tiny(tables)
    tables = list(tiny_formal_tables())
    tables[2] = tables[2].iloc[:-1].copy()
    with pytest.raises(ValueError, match="fold-selection"):
        validate_tiny(tables)


def test_strict_json_sanitizes_nonfinite_and_numpy_scalars() -> None:
    value = {
        "unweighted": np.float64(np.nan),
        "nested": [np.float64(np.inf), -np.inf, np.int64(17), np.bool_(True)],
        "finite": np.float64(0.10970300378540344),
    }
    serialized = _strict_json(value)
    assert "NaN" not in serialized and "Infinity" not in serialized
    parsed = json.loads(
        serialized,
        parse_constant=lambda token: pytest.fail(f"Non-standard JSON token: {token}"),
    )
    assert parsed["unweighted"] is None
    assert parsed["nested"] == [None, None, 17, True]
    assert parsed["finite"] == 0.10970300378540344
    assert '"unweighted": null' in serialized


def test_disabled_notebook_guard_has_no_formal_side_effects(tmp_path: Path) -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    guard = next(cell for cell in notebook["cells"] if cell.get("id") == "lgbm-formal-run-guard")
    code = "".join(guard["source"])
    assert code.startswith("RUN_FORMAL_STRICT_NESTED_CV = False\n")
    ast.parse(code)
    # Rebind the output base to a temporary path. The false branch must not
    # call any loader, runner, or writer.
    namespace = {"RESULTS_DIR": tmp_path}
    exec(compile(code, str(NOTEBOOK_PATH), "exec"), namespace)
    assert not (tmp_path / "strict_nested").exists()


def test_tiny_artifact_writer_smoke(tmp_path: Path) -> None:
    predictions, candidates, folds, registry, development = tiny_formal_tables()
    config = load_model_config(CONFIG_PATH, strict_schema=True)
    from src.evaluation.metrics import calculate_binary_metrics

    fold_metric_rows = []
    for fold in range(1, 6):
        subset = predictions.loc[predictions["cv_fold"].eq(fold)]
        metrics = calculate_binary_metrics(
            subset["y_true"], subset["y_probability"], subset["y_pred_0_5"]
        )
        metrics.update({"fold": fold, "fit_time_seconds": 0.1, "predict_time_seconds": 0.01})
        fold_metric_rows.append(metrics)
    pooled = calculate_binary_metrics(
        predictions["y_true"], predictions["y_probability"], predictions["y_pred_0_5"]
    )
    result = OOFResult(
        model_name="LightGBM",
        model_id="lightgbm",
        predictions=predictions,
        fold_metrics=pd.DataFrame(fold_metric_rows),
        summary_metrics={
            "model_name": "LightGBM",
            "pooled_average_precision": pooled["average_precision"],
            "pooled_roc_auc": pooled["roc_auc"],
            "pooled_f1": pooled["f1"],
        },
        fit_time_seconds_total=0.5,
        predict_time_seconds_total=0.05,
    )

    class MetadataOnlyAdapter:
        def __init__(self, fold: int):
            self.fold = fold

        def get_metadata(self):
            selection = candidates.loc[candidates["outer_fold"].eq(self.fold)]
            return {
                "outer_fold": self.fold,
                "candidate_selection_records": [
                    {
                        "candidate_id": row.candidate_id,
                        "inner_average_precision": row.inner_average_precision,
                        "configured_n_estimators": row.configured_n_estimators,
                        "selected_best_iteration": row.best_iteration,
                        "class_weight": row.class_weight,
                        "runtime_seconds": row.runtime_seconds,
                        "success": row.success,
                        "candidate_parameters": {"n_estimators": 600},
                        "resolved_parameters": {"n_estimators": 600},
                        "inner_train_row_count": 7,
                        "inner_validation_row_count": 1,
                    }
                    for row in selection.itertuples()
                ],
                "fit_summary": {
                    "outer_training_row_count": 8,
                    "inner_training_row_count": 7,
                    "inner_validation_row_count": 1,
                    "selected_candidate_id": "LGBM_F01",
                    "selected_inner_average_precision": 0.6,
                    "selected_best_iteration": 17,
                    "selected_class_weight": None,
                    "refit_row_count": 8,
                    "selection_time_seconds": 0.08,
                    "refit_time_seconds": 0.02,
                    "fit_time_seconds": 0.10,
                },
            }

    first_root = tmp_path / "first"
    output_dir = first_root / "strict_nested"
    adapters = [MetadataOnlyAdapter(fold) for fold in range(1, 6)]
    outputs = write_lightgbm_formal_artifacts(
        result=result,
        adapters=adapters,
        development=development,
        split_registry=registry,
        config=config,
        project_root=first_root,
        output_dir=output_dir,
        source_head="a" * 40,
        expected_rows=10,
        expected_fold_rows=2,
    )
    assert {path.name for path in output_dir.iterdir()} == {*FORMAL_FILE_ROLES, "output_manifest.csv"}
    assert len(outputs["candidate_selection"]) == 40
    assert len(outputs["fold_selection"]) == 5
    assert len(outputs["output_manifest"]) == 8
    assert outputs["output_manifest"]["artifact"].is_unique
    self_rows = outputs["output_manifest"].loc[
        outputs["output_manifest"]["artifact"].str.endswith("/output_manifest.csv")
    ]
    assert len(self_rows) == 1
    assert pd.isna(self_rows.iloc[0]["sha256"])
    assert pd.isna(self_rows.iloc[0]["size_bytes"])
    assert "self-reference" in self_rows.iloc[0]["checksum_note"]
    assert set(outputs["output_manifest"]["artifact"].str.rsplit("/", n=1).str[-1]) == {
        *FORMAL_FILE_ROLES, "output_manifest.csv"
    }
    assert {"Employee_ID", "cv_fold", "y_true", "model_name", "y_probability"}.issubset(
        outputs["predictions"].columns
    )
    assert {"fold", "average_precision", "roc_auc", "f1", "brier_score"}.issubset(
        outputs["fold_metrics"].columns
    )
    assert {
        "config_version", "candidate_registry_version", "candidate_ids",
        "candidate_definitions", "source_head", "split_registry", "development_rows",
        "outer_folds", "inner_fraction", "seed", "stratified", "selection_metric",
        "tie_breaker", "early_stopping_rounds", "refit_policy", "fold_winners",
        "formal_holdout_used",
    }.issubset(outputs["run_metadata"])
    assert {"artifact", "exists", "size_bytes", "sha256", "role"}.issubset(
        outputs["output_manifest"].columns
    )
    assert outputs["run_metadata"]["formal_holdout_used"] is False
    assert outputs["summary_metrics"]["pooled_average_precision"] == pooled["average_precision"]
    metadata_text = (output_dir / "run_metadata.json").read_text(encoding="utf-8")
    assert "NaN" not in metadata_text and "Infinity" not in metadata_text
    strict_metadata = json.loads(
        metadata_text,
        parse_constant=lambda token: pytest.fail(f"Non-standard JSON token: {token}"),
    )
    assert strict_metadata["fold_winners"][0]["selected_class_weight"] is None
    assert strict_metadata["fold_winners"][1]["selected_class_weight"] is None
    second_root = tmp_path / "second"
    second_dir = second_root / "strict_nested"
    write_lightgbm_formal_artifacts(
        result=result,
        adapters=adapters,
        development=development,
        split_registry=registry,
        config=config,
        project_root=second_root,
        output_dir=second_dir,
        source_head="a" * 40,
        expected_rows=10,
        expected_fold_rows=2,
    )
    assert {
        path.name: path.read_bytes() for path in output_dir.iterdir()
    } == {
        path.name: path.read_bytes() for path in second_dir.iterdir()
    }
    result.summary_metrics["pooled_average_precision"] = 0.0
    stale_output_dir = tmp_path / "stale_summary"
    with pytest.raises(ValueError, match="pooled_average_precision"):
        write_lightgbm_formal_artifacts(
            result=result,
            adapters=[MetadataOnlyAdapter(fold) for fold in range(1, 6)],
            development=development,
            split_registry=registry,
            config=config,
            project_root=tmp_path,
            output_dir=stale_output_dir,
            source_head="a" * 40,
            expected_rows=10,
            expected_fold_rows=2,
        )
    assert not stale_output_dir.exists()
