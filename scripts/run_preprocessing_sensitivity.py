"""Small, paired preprocessing checks for members 1, 2, 4 and 6.

Each model is evaluated with its normal 17 predictors and with salary masked
at the training-only median. Both representations use the same 5,000 training
rows and 2,000 untouched rows from frozen outer fold 1. These exploratory
fits are not candidates for the six-model headline comparison.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.catboost_model import CatBoostAdapter
from src.models.ft_transformer import FTTransformerAdapter
from src.models.lightgbm_model import LightGBMAdapter
from src.models.tabnet_model import TabNetAdapter, load_tabnet_config
from src.evaluation.inputs import load_development_data
from src.models.experiment_outputs import create_run_directory


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

OUTER_FOLD = 1
TRAIN_ROWS = 5_000
VALIDATION_ROWS = 2_000


def load_data(root=ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Sample only hash-verified development rows; never parse holdout outcomes."""
    root = Path(root)
    contract = yaml.safe_load((root / "configs/feature_contract.yaml").read_text(encoding="utf-8"))
    features = list(contract["primary_model_features"])
    categories = [item["name"] for item in contract["categorical_features"]]
    raw, registry = load_development_data(root, usecols=features)
    development = registry.loc[registry["partition"] == "development", ["Employee_ID", "cv_fold"]]
    data = raw.merge(development, on="Employee_ID", how="inner", validate="one_to_one")
    assert len(data) == contract["split_registry"]["development_rows"] and data["Employee_ID"].is_unique
    assert set(data["Employee_ID"]).isdisjoint(
        registry.loc[registry["partition"] == "holdout", "Employee_ID"]
    )
    train_pool = data.loc[data["cv_fold"] != OUTER_FOLD]
    validation_pool = data.loc[data["cv_fold"] == OUTER_FOLD]
    train, _ = train_test_split(
        train_pool, train_size=TRAIN_ROWS, stratify=train_pool["Resigned"], random_state=2026
    )
    validation, _ = train_test_split(
        validation_pool, train_size=VALIDATION_ROWS,
        stratify=validation_pool["Resigned"], random_state=2027,
    )
    return (
        train[features].reset_index(drop=True),
        validation[features].reset_index(drop=True),
        train["Resigned"].astype(int).reset_index(drop=True),
        validation["Resigned"].astype(int).reset_index(drop=True),
        categories,
    )


def make_adapter(model: str, profile: str, categories: list[str]):
    if model == "FT-Transformer":
        config = yaml.safe_load((ROOT / "configs/models/ft_transformer.yaml").read_text(encoding="utf-8"))
        candidate = next(item for item in config["candidates"] if item["id"] == profile)
        params = {
            **config["fixed_params"], **candidate["params"],
            "max_epochs": 5, "patience": 2, "validation_fraction_inner": 0.15,
        }
        return FTTransformerAdapter(params=params, categorical_features=categories)
    if model == "CatBoost":
        config = yaml.safe_load((ROOT / "configs/models/catboost.yaml").read_text(encoding="utf-8"))
        candidate = next(item for item in config["candidates"] if item["id"] == profile)
        params = {**config["fixed_params"], **candidate["params"], "iterations": 120, "thread_count": 4}
        return CatBoostAdapter(params=params, categorical_features=categories)
    if model == "LightGBM":
        config = yaml.safe_load((ROOT / "configs/models/lightgbm.yaml").read_text(encoding="utf-8"))
        candidate = next(item for item in config["candidates"] if item["id"] == profile)
        params = {**config["fixed_params"], **candidate["params"], "n_estimators": 120, "n_jobs": 4}
        return LightGBMAdapter(params=params, categorical_features=categories)
    if model == "TabNet":
        config = load_tabnet_config(ROOT / "configs/models/tabnet.yaml")
        return TabNetAdapter(
            config=config, candidate_id=profile, max_epochs_override=5,
            patience_override=2, outer_fold=OUTER_FOLD,
        )
    raise ValueError(model)


def main() -> None:
    import torch

    torch.set_num_threads(4)
    X_train, X_valid, y_train, y_valid, categories = load_data()
    # Reserve a fresh destination before fitting. A repeat must not replace the
    # original paired exercise or any previous rerun's evidence.
    output_dir = create_run_directory(ROOT, "preprocessing_sensitivity")
    salary_median = float(X_train["Monthly_Salary"].median())
    assert X_train.shape == (TRAIN_ROWS, 17)
    assert X_valid.shape == (VALIDATION_ROWS, 17)

    profiles = {
        "FT-Transformer": ("FT01", "FT02"),
        "CatBoost": ("C01", "C02"),
        "LightGBM": ("LGBM_F01", "LGBM_F02"),
        "TabNet": ("T01", "T02"),
    }
    rows = []
    for model, candidate_ids in profiles.items():
        for profile in candidate_ids:
            for preprocessing in ("full_17", "salary_train_median"):
                train = X_train.copy()
                valid = X_valid.copy()
                if preprocessing == "salary_train_median":
                    train["Monthly_Salary"] = salary_median
                    valid["Monthly_Salary"] = salary_median
                adapter = make_adapter(model, profile, categories)
                started = time.perf_counter()
                adapter.fit(train, y_train)
                probability = np.asarray(adapter.predict_proba(valid), dtype=float)
                assert len(probability) == VALIDATION_ROWS
                assert np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all()
                rows.append({
                    "model": model,
                    "candidate_template": profile,
                    "preprocessing": preprocessing,
                    "train_rows": TRAIN_ROWS,
                    "outer_fold_validation_rows": VALIDATION_ROWS,
                    "train_positive_rate": float(y_train.mean()),
                    "validation_positive_rate": float(y_valid.mean()),
                    "average_precision": average_precision_score(y_valid, probability),
                    "roc_auc": roc_auc_score(y_valid, probability),
                    "brier_score": brier_score_loss(y_valid, probability),
                    "fit_and_predict_seconds": time.perf_counter() - started,
                })
                print(model, profile, preprocessing, f"AP={rows[-1]['average_precision']:.6f}", flush=True)

    pd.DataFrame(rows).to_csv(output_dir / "paired_exploratory.csv", index=False)
    details = {
        "role": "Exploratory preprocessing sensitivity; never select the headline experiment from these validation scores",
        "outer_fold": OUTER_FOLD,
        "training_sample_rows": TRAIN_ROWS,
        "validation_sample_rows": VALIDATION_ROWS,
        "training_sample_seed": 2026,
        "validation_sample_seed": 2027,
        "salary_mask_value_from_training_only": salary_median,
        "candidate_templates": profiles,
        "training_cap": "FT/TabNet 5 epochs; CatBoost/LightGBM 120 trees; all shorter than formal evaluations",
        "split": "training sample from folds 2-5; untouched validation sample from outer fold 1; no formal holdout IDs",
    }
    (output_dir / "protocol.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print("Saved", output_dir / "paired_exploratory.csv")


if __name__ == "__main__":
    main()
