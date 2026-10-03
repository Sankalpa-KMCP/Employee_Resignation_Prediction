"""Check that the group's shipped evidence matches its model registry."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.oof_validation import validate_oof_predictions
from src.evaluation.inputs import load_group_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_six_model_inputs_match_frozen_development() -> None:
    registry, frames = load_group_inputs(ROOT)
    assert len(registry["models"]) == 6
    assert set(frames) == set(registry["models"])
    assert all(len(predictions) == 80_000 for predictions in frames.values())


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
    assert "load_formal_ft_input" in code
    assert "RUN_TRAINING=False" in code


def test_canonical_ft_gpu_evidence_and_group_metrics_agree() -> None:
    registry = yaml.safe_load((ROOT / "configs/analysis_registry.yaml").read_text(encoding="utf-8"))
    entry = registry["models"]["FT-Transformer"]
    directory = (ROOT / entry["oof_path"]).parent
    assert directory.name == "strict_nested"
    from src.models.ft_transformer_execution import load_formal_ft_input
    predictions, folds, summary = load_formal_ft_input(ROOT)
    evidence = json.loads((directory / "selection_evidence.json").read_text())
    assert entry["candidate_search_executed"] and entry["tuning_complete"]
    assert len(evidence) == 5 and all(len(e["all_candidates"]) == 8 for e in evidence)
    assert all(e["fitted_on_all_outer_rows"] for e in evidence)
    comparison = pd.read_csv(ROOT / "results/comparison/model_comparison.csv")
    grouped = comparison.loc[comparison.model == "FT-Transformer"].squeeze()
    assert np.isclose(average_precision_score(predictions.y_true, predictions.y_probability), summary["average_precision"])
    assert np.isclose(grouped.pooled_average_precision, summary["average_precision"])
    assert np.isclose(grouped.pooled_roc_auc, summary["roc_auc"])
    assert np.isclose(grouped.brier_score, summary["brier_score"])
    assert np.isclose(grouped.cv_ap_mean, folds.average_precision.mean())
