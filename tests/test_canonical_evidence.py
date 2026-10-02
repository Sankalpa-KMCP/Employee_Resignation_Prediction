"""Check that the group's shipped evidence matches its model registry."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.oof_validation import validate_oof_predictions


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_six_model_inputs_match_frozen_development() -> None:
    registry = yaml.safe_load((ROOT / "configs/analysis_registry.yaml").read_text(encoding="utf-8"))
    split = pd.read_csv(ROOT / "data/processed/split_registry.csv", dtype={"cv_fold": "Int64"})
    raw = pd.read_csv(ROOT / "data/raw/Extended_Employee_Performance_and_Productivity_Data.csv")

    assert len(registry["models"]) == 6
    for entry in registry["models"].values():
        predictions = pd.read_csv(ROOT / entry["oof_path"])
        validate_oof_predictions(predictions, split, source_data=raw)


def test_member3_primary_result_is_the_group_xgboost_input() -> None:
    registry = yaml.safe_load((ROOT / "configs/analysis_registry.yaml").read_text(encoding="utf-8"))
    path = registry["models"]["XGBoost"]["oof_path"]
    assert path == "results/models/xgboost/oof_predictions.csv"
    predictions = pd.read_csv(ROOT / path)
    summary = json.loads((ROOT / "results/models/xgboost/summary_metrics.json").read_text(encoding="utf-8"))
    varieties = pd.read_csv(ROOT / "results/models/xgboost/preprocessing_comparison.csv")
    primary = varieties.loc[varieties["variant"] == "P1_onehot"].squeeze()

    assert len(predictions) == 80_000
    assert np.isclose(average_precision_score(predictions.y_true, predictions.y_probability), summary["pooled_average_precision"])
    assert np.isclose(roc_auc_score(predictions.y_true, predictions.y_probability), summary["pooled_roc_auc"])
    assert np.isclose(brier_score_loss(predictions.y_true, predictions.y_probability), summary["pooled_brier_score"])
    assert np.isclose(primary["pooled_ap"], summary["pooled_average_precision"])


def test_ft_weighted_historical_evidence_matches_current_candidate() -> None:
    config = yaml.safe_load((ROOT / "configs/models/ft_transformer.yaml").read_text(encoding="utf-8"))
    weighted = next(item for item in config["candidates"] if item["id"] == "FT02")
    actual = {**config["fixed_params"], **weighted["params"]}
    directory = ROOT / "results/models/ft_transformer/weighted_early_stopping_final"
    saved = json.loads((directory / "ft_transformer_final_configuration.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "ft_transformer_final_run_metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((ROOT / "notebooks/team_contributions/models/member_1_ft_transformer/05_m1_ft_transformer.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")

    assert all(actual[key] == value for key, value in saved.items() if key in actual)
    assert actual["weighting"] == "dynamic_ratio"
    assert metadata["class_weighting"] == "inner-training positive-class weight"
    assert 'c["id"] == "FT02"' in code
    assert "RUN_FT_TRAINING = False" in code
    assert 'RESULTS_DIR / "reruns"' in code
