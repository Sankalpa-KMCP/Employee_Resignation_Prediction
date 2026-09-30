"""Fast tests for the formal strict-nested LightGBM workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.models.lightgbm_model as lightgbm_module
from src.models.config import load_model_config
from src.models.lightgbm_model import LightGBMAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "models" / "lightgbm.yaml"
EXPECTED_IDS = [f"LGBM_F{number:02d}" for number in range(1, 9)]


class FakeLGBMClassifier:
    """Small deterministic stand-in that records every fit boundary."""

    fit_calls: list[dict[str, Any]] = []

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.best_iteration_ = min(17, int(self.params.get("n_estimators", 17)))
        self.last_predict_frame: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> FakeLGBMClassifier:
        eval_set = kwargs.get("eval_set")
        if eval_set is None and kwargs.get("eval_X") is not None:
            eval_set = [(kwargs["eval_X"], kwargs["eval_y"])]
        self.fit_calls.append(
            {
                "model": self,
                "params": self.params.copy(),
                "train_indices": tuple(X.index.tolist()),
                "train_y": np.asarray(y, dtype=int).copy(),
                "eval_set": eval_set,
                "eval_metric": kwargs.get("eval_metric"),
                "callbacks": kwargs.get("callbacks"),
            }
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.last_predict_frame = X.copy()
        age = pd.to_numeric(X["Age"], errors="raise").to_numpy(dtype=float)
        if len(age) == 0:
            score = np.asarray([], dtype=float)
        elif float(self.params.get("learning_rate", 0.05)) == 0.10:
            scale = max(float(age.max() - age.min()), 1.0)
            score = 0.05 + 0.90 * (age - age.min()) / scale
        elif float(self.params.get("learning_rate", 0.05)) == 0.03:
            scale = max(float(age.max() - age.min()), 1.0)
            score = 0.95 - 0.90 * (age - age.min()) / scale
        else:
            score = np.full(len(X), 0.10, dtype=float)
        return np.column_stack([1.0 - score, score])


def fake_early_stopping(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def synthetic_training_data(rows: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    age = np.arange(rows, dtype=float)
    data = pd.DataFrame(
        {
            "Department": np.where(np.arange(rows) % 2 == 0, "Sales", "Engineering"),
            "Gender": np.where(np.arange(rows) % 2 == 0, "Female", "Male"),
            "Job_Title": np.where(np.arange(rows) % 3 == 0, "Analyst", "Engineer"),
            "Education_Level": np.where(
                np.arange(rows) % 2 == 0, "Bachelor", "Master"
            ),
            "Age": age,
            "Monthly_Salary": 3000.0 + age * 10.0,
        }
    )
    target = pd.Series((age >= rows * 0.80).astype(int), name="Resigned")
    return data, target


def formal_config() -> dict[str, Any]:
    return load_model_config(CONFIG_PATH, strict_schema=True)


def test_formal_registry_is_exact_bounded_and_weight_balanced() -> None:
    config = formal_config()
    candidates = config["candidates"]
    assert len(candidates) == 8
    assert len(candidates) <= 12
    assert [candidate["id"] for candidate in candidates] == EXPECTED_IDS
    assert config["candidate_registry"]["formal_candidate_ids"] == EXPECTED_IDS

    allowed = {
        "n_estimators": {300, 600, 1000},
        "learning_rate": {0.03, 0.05, 0.10},
        "num_leaves": {15, 31, 63},
        "max_depth": {-1, 6, 10},
        "min_child_samples": {20, 50, 100},
        "subsample": {0.8, 1.0},
        "subsample_freq": {1},
        "colsample_bytree": {0.8, 1.0},
        "reg_lambda": {0, 1, 5},
        "class_weight": {None, "balanced"},
    }
    for candidate in candidates:
        for parameter, values in allowed.items():
            assert candidate["params"][parameter] in values

    weights = [candidate["params"]["class_weight"] for candidate in candidates]
    assert weights.count(None) == 4
    assert weights.count("balanced") == 4

    for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
        first_params = candidates[first]["params"].copy()
        second_params = candidates[second]["params"].copy()
        assert first_params.pop("class_weight") is None
        assert second_params.pop("class_weight") == "balanced"
        assert first_params == second_params


def test_inner_split_is_deterministic_stratified_and_complete() -> None:
    _, target = synthetic_training_data()
    adapter = LightGBMAdapter(config=formal_config(), candidate_id="auto")
    train_one, validation_one = adapter._create_inner_split(target)
    train_two, validation_two = adapter._create_inner_split(target)

    np.testing.assert_array_equal(train_one, train_two)
    np.testing.assert_array_equal(validation_one, validation_two)
    assert set(train_one).isdisjoint(set(validation_one))
    assert set(train_one).union(set(validation_one)) == set(range(len(target)))
    assert len(validation_one) == 30
    assert np.isclose(target.iloc[train_one].mean(), target.mean(), atol=0.01)
    assert np.isclose(target.iloc[validation_one].mean(), target.mean(), atol=0.01)


def test_formal_selection_uses_shared_inner_split_ap_tie_break_and_clean_refit(
    monkeypatch,
) -> None:
    FakeLGBMClassifier.fit_calls = []
    monkeypatch.setattr(
        lightgbm_module,
        "_lightgbm_dependencies",
        lambda: (FakeLGBMClassifier, fake_early_stopping),
    )
    data, target = synthetic_training_data()
    adapter = LightGBMAdapter(config=formal_config(), candidate_id="auto", outer_fold=1)
    adapter.fit(data, target)

    assert len(FakeLGBMClassifier.fit_calls) == 9
    candidate_calls = FakeLGBMClassifier.fit_calls[:8]
    refit_call = FakeLGBMClassifier.fit_calls[-1]

    expected_train_indices = candidate_calls[0]["train_indices"]
    expected_validation_indices = tuple(candidate_calls[0]["eval_set"][0][0].index)
    for call in candidate_calls:
        assert call["train_indices"] == expected_train_indices
        assert tuple(call["eval_set"][0][0].index) == expected_validation_indices
        assert call["eval_metric"] == "average_precision"
        assert call["callbacks"] and call["callbacks"][0]["stopping_rounds"] == 50
        assert set(call["train_indices"]).isdisjoint(expected_validation_indices)

    # F07 and F08 have the same best AP under the fake estimator; ID order selects F07.
    assert adapter.selected_candidate_id == "LGBM_F07"
    assert adapter.selected_best_iteration == 17
    assert len(adapter.candidate_selection_records) == 8
    assert all(record["success"] for record in adapter.candidate_selection_records)
    assert adapter.selected_inner_average_precision == max(
        record["inner_average_precision"]
        for record in adapter.candidate_selection_records
    )

    assert refit_call["train_indices"] == tuple(data.index)
    assert refit_call["eval_set"] is None
    assert refit_call["callbacks"] is None
    assert refit_call["params"]["n_estimators"] == 17
    assert adapter.fit_summary["refit_row_count"] == len(data)
    assert adapter.fit_summary["outer_training_row_count"] == len(data)
    assert adapter.fit_summary["all_candidates_shared_inner_split"] is True

    balanced_records = [
        record
        for record in adapter.candidate_selection_records
        if record["class_weight"] == "balanced"
    ]
    assert len(balanced_records) == 4
    assert all(
        record["inner_training_weight_evidence"]["row_count"]
        == len(expected_train_indices)
        for record in balanced_records
    )


def test_selection_mappings_exclude_inner_validation_then_refit_all_rows(
    monkeypatch,
) -> None:
    FakeLGBMClassifier.fit_calls = []
    monkeypatch.setattr(
        lightgbm_module,
        "_lightgbm_dependencies",
        lambda: (FakeLGBMClassifier, fake_early_stopping),
    )
    data, target = synthetic_training_data()
    probe = LightGBMAdapter(config=formal_config(), candidate_id="auto")
    _, validation_positions = probe._create_inner_split(target)
    special_position = int(validation_positions[0])
    data.loc[special_position, "Department"] = "INNER_VALIDATION_ONLY"

    adapter = LightGBMAdapter(config=formal_config(), candidate_id="auto")
    adapter.fit(data, target)
    first_inner_validation = FakeLGBMClassifier.fit_calls[0]["eval_set"][0][0]
    assert -1 in first_inner_validation["Department"].astype("int32").to_numpy()

    metadata = adapter.get_metadata()
    final_mapping = metadata["categorical_mapping_metadata"]["category_mappings"]
    assert "INNER_VALIDATION_ONLY" in final_mapping["Department"]

    outer_validation = data.iloc[[0]].copy()
    outer_validation.index = [10_000]
    outer_validation.loc[10_000, "Department"] = "OUTER_VALIDATION_ONLY"
    probabilities = adapter.predict_proba(outer_validation)
    assert probabilities.shape == (1,)
    assert 0.0 <= probabilities[0] <= 1.0
    transformed = adapter._model.last_predict_frame
    assert transformed is not None
    assert int(transformed["Department"].astype("int32").iloc[0]) == -1
    assert 10_000 not in set(
        index
        for call in FakeLGBMClassifier.fit_calls
        for index in call["train_indices"]
    )


def test_best_iteration_falls_back_to_ceiling() -> None:
    class NoEarlyStopModel:
        best_iteration_ = 0

    assert LightGBMAdapter._resolve_best_iteration(NoEarlyStopModel(), 600) == 600


def test_formal_adapter_save_load_prediction_parity(monkeypatch, tmp_path: Path) -> None:
    FakeLGBMClassifier.fit_calls = []
    monkeypatch.setattr(
        lightgbm_module,
        "_lightgbm_dependencies",
        lambda: (FakeLGBMClassifier, fake_early_stopping),
    )
    data, target = synthetic_training_data()
    adapter = LightGBMAdapter(config=formal_config(), candidate_id="auto")
    adapter.fit(data, target)
    expected = adapter.predict_proba(data.iloc[:12])

    metadata = adapter.save(tmp_path)
    reloaded = LightGBMAdapter.load(tmp_path)
    observed = reloaded.predict_proba(data.iloc[:12])
    np.testing.assert_allclose(observed, expected)
    assert metadata["selected_candidate_id"] == "LGBM_F07"
    assert reloaded.get_metadata()["selected_best_iteration"] == 17


def test_real_lightgbm_single_candidate_nested_smoke() -> None:
    pytest.importorskip("lightgbm")
    config = formal_config()
    balanced_candidate = next(
        candidate
        for candidate in config["candidates"]
        if candidate["id"] == "LGBM_F08"
    )
    config["candidates"] = [balanced_candidate]
    config["candidate_registry"]["formal_candidate_ids"] = ["LGBM_F08"]

    data, target = synthetic_training_data(rows=300)
    adapter = LightGBMAdapter(
        config=config,
        candidate_id="LGBM_F08",
        outer_fold=1,
    )
    adapter.fit(data, target)
    probabilities = adapter.predict_proba(data.iloc[:20])

    assert adapter.selected_candidate_id == "LGBM_F08"
    assert adapter.selected_candidate_parameters["class_weight"] == "balanced"
    assert 1 <= adapter.selected_best_iteration <= 300
    assert adapter.fit_summary["inner_training_row_count"] == 255
    assert adapter.fit_summary["inner_validation_row_count"] == 45
    assert adapter.fit_summary["refit_row_count"] == 300
    assert probabilities.shape == (20,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
