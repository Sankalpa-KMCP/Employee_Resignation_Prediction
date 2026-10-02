"""Unit tests for the XGBoost configuration and XGBoostAdapter (Member 3).

Covers the strict nested outer-CV / inner-selection protocol from Project
Context v4.6.2 (Sections 8.17, 8.19.2, 8.21.5) and QA tests:
    T-008  encoder fitted on outer-training rows only; unknown categories work
    T-013  inner validation contains only outer-training rows
    T-014  search count does not exceed the frozen cap
    T-016  probability range / NaN / row checks
    T-023  reload plus encoder reproduces predictions

Also covers the review fixes: inner selection is never skipped (zero/None
patience), one parameter-precedence rule for raw and resolved configs, the
candidate cap, the preprocessing variants and the tree-count diagnostic.

Leakage tests do not trust the adapter's own bookkeeping: they wrap
``xgboost.XGBClassifier.fit`` and record every row XGBoost actually trains on
or early-stops on.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xgboost

from src.models.base import ModelAdapter
from src.models.config import (
    _detect_unresolved_placeholders,
    load_model_config,
    resolve_candidate,
    validate_model_config,
)
from src.models.runner import run_5fold_cv
from src.models.xgboost_model import (
    ARTIFACT_FILENAME,
    METADATA_FILENAME,
    XGBoostAdapter,
)

CATEGORICALS = ["Department", "Gender", "Job_Title", "Education_Level"]

# Canonical XGBoost grid (Project Context v4.6.2, Section 8.21.5)
CANONICAL_GRID = {
    "max_depth": {3, 5, 7},
    "learning_rate": {0.03, 0.05, 0.10},
    "min_child_weight": {1, 5, 10},
    "subsample": {0.8, 1.0},
    "colsample_bytree": {0.8, 1.0},
    "reg_lambda": {1, 5, 10},
    "weighting": {"none", "dynamic_ratio"},
}
CANONICAL_MAX_CANDIDATES = 12


# =============================================================================
# Fixtures and helpers
# =============================================================================

@pytest.fixture(scope="module")
def xgb_config() -> dict:
    """Authoritative XGBoost config loaded through the central strict loader."""
    return load_model_config("xgboost", strict_schema=True)


@pytest.fixture
def fast_config(xgb_config: dict) -> dict:
    """Same config with a lower round ceiling and patience to keep tests fast."""
    cfg = copy.deepcopy(xgb_config)
    cfg["inner_validation"]["max_boosting_rounds"] = 60
    cfg["inner_validation"]["early_stopping_rounds"] = 10
    return cfg


def _make_data(n: int = 600, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic data with contract-style columns, ~15% positives and weak signal.

    ``Row_Tag`` is a unique numeric passthrough column so tests can identify
    exactly which rows reached XGBoost after one-hot encoding.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "Row_Tag": np.arange(n, dtype=float),
            "Age": rng.integers(22, 60, size=n),
            "Monthly_Salary": rng.uniform(3000, 15000, size=n),
            "Satisfaction": rng.uniform(1, 5, size=n),
            "Department": rng.choice(["IT", "HR", "Sales", "Finance"], size=n),
            "Gender": rng.choice(["Male", "Female", "Other"], size=n),
            "Job_Title": rng.choice(["Analyst", "Engineer", "Manager"], size=n),
            "Education_Level": rng.choice(["Bachelor", "Master", "PhD"], size=n),
        }
    )
    logit = -2.0 + 0.8 * (X["Satisfaction"] < 2).astype(float) + 0.5 * (X["Department"] == "Sales")
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="Resigned")
    return X, y


class FitSpy:
    """Records the Row_Tag values of every XGBClassifier.fit call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "FitSpy":
        original_fit = xgboost.XGBClassifier.fit
        spy = self

        def recording_fit(model, X, y, *args, **kwargs):
            eval_set = kwargs.get("eval_set") or []
            spy.calls.append(
                {
                    "train_tags": set(np.asarray(X["Row_Tag"]).astype(int).tolist()),
                    "train_y": np.asarray(y).astype(int),
                    "eval_tags": set().union(
                        *[set(np.asarray(ev[0]["Row_Tag"]).astype(int).tolist()) for ev in eval_set]
                    ) if eval_set else set(),
                    "has_eval_set": bool(eval_set),
                    "n_estimators": model.get_params()["n_estimators"],
                    "scale_pos_weight": model.get_params()["scale_pos_weight"],
                    "early_stopping_rounds": model.get_params().get("early_stopping_rounds"),
                    "columns": list(X.columns),
                    "extra_kwargs": dict(getattr(model, "kwargs", None) or {}),
                }
            )
            return original_fit(model, X, y, *args, **kwargs)

        monkeypatch.setattr(xgboost.XGBClassifier, "fit", recording_fit)
        return self


def _ratio(y: np.ndarray) -> float:
    return float((y == 0).sum() / (y == 1).sum())


# =============================================================================
# 1. Configuration integrity
# =============================================================================

def test_xgboost_yaml_passes_strict_schema(xgb_config: dict):
    validate_model_config(xgb_config, strict_schema=True, model_id="xgboost")
    assert xgb_config["model"]["model_id"] == "xgboost"
    assert xgb_config["model"]["name"] == "XGBoost"
    assert xgb_config["model"]["implementation"] == "xgboost.XGBClassifier"
    assert xgb_config["model"]["config_version"] == "2.1.0"
    assert _detect_unresolved_placeholders(xgb_config) == []


def test_inner_validation_policy_values(xgb_config: dict):
    inner = xgb_config["inner_validation"]
    assert inner["fraction"] == 0.15
    assert inner["stratified"] is True
    assert inner["metric"] == "average_precision"
    assert inner["early_stopping_rounds"] == 50
    assert inner["refit_outer_train"] is True
    assert inner["max_boosting_rounds"] == xgb_config["fixed_params"]["n_estimators"]
    assert xgb_config["class_imbalance"]["strategy"] == "dynamic_ratio"
    assert xgb_config["evaluation"]["formal_holdout_allowed"] is False


def test_candidate_registry_within_canonical_grid_and_cap(xgb_config: dict):
    """T-014 plus Section 8.21.5: bounded, unique, in-grid candidates."""
    candidates = xgb_config["candidates"]
    ids = [c["id"] for c in candidates]
    assert ids == [f"X{i:02d}" for i in range(1, 9)]
    assert len(ids) <= CANONICAL_MAX_CANDIDATES
    assert xgb_config["search_budget"]["max_candidates"] <= CANONICAL_MAX_CANDIDATES
    assert xgb_config["search_budget"]["declared_candidates"] == len(ids)

    signatures = set()
    for cand in candidates:
        params = cand["params"]
        assert set(params) == set(CANONICAL_GRID), f"{cand['id']} keys differ from canonical grid"
        for key, value in params.items():
            assert value in CANONICAL_GRID[key], f"{cand['id']}.{key}={value!r} outside canonical grid"
        signatures.add(tuple(sorted(params.items())))
    assert len(signatures) == len(candidates), "Duplicate candidate parameter sets."

    weightings = {c["params"]["weighting"] for c in candidates}
    assert weightings == {"none", "dynamic_ratio"}


def test_every_candidate_resolves_with_fixed_params(xgb_config: dict):
    for cand in xgb_config["candidates"]:
        resolved = resolve_candidate(xgb_config, candidate_id=cand["id"])
        assert resolved["candidate_id"] == cand["id"]
        assert resolved["params"]["objective"] == "binary:logistic"
        assert resolved["params"]["tree_method"] == "hist"
        assert resolved["params"]["max_depth"] == cand["params"]["max_depth"]


# =============================================================================
# 2. Construction and protocol compliance
# =============================================================================

def test_zero_arg_adapter_is_protocol_compliant_and_unfitted():
    adapter = XGBoostAdapter()
    assert isinstance(adapter, ModelAdapter)
    meta = adapter.get_metadata()
    assert meta["fitted"] is False
    assert meta["name"] == "XGBoost" and meta["model_id"] == "xgboost"
    with pytest.raises(RuntimeError):
        adapter.predict_proba(_make_data(10)[0])


def test_config_mode_defaults(xgb_config: dict):
    adapter = XGBoostAdapter(config=xgb_config)
    assert adapter.mode == "nested"
    assert adapter.candidate_id == "auto"
    assert adapter.name == "XGBoost"
    assert adapter.inner_val_fraction == 0.15
    assert adapter.early_stopping_rounds == 50
    assert adapter.refit_outer_train is True
    assert adapter.random_seed == 42
    assert adapter.get_metadata()["candidate_pool_ids"] == [f"X{i:02d}" for i in range(1, 9)]


def test_specific_candidate_restricts_pool(xgb_config: dict):
    adapter = XGBoostAdapter(config=xgb_config, candidate_id="X04")
    assert adapter.get_metadata()["candidate_pool_ids"] == ["X04"]


def test_unknown_candidate_raises(xgb_config: dict):
    with pytest.raises(ValueError, match="Unknown candidate_id"):
        XGBoostAdapter(config=xgb_config, candidate_id="X99")


def test_invalid_weighting_raises(fast_config: dict):
    cfg = copy.deepcopy(fast_config)
    cfg["candidates"][0]["params"]["weighting"] = "balanced_magic"
    X, y = _make_data()
    with pytest.raises(ValueError, match="Invalid weighting"):
        XGBoostAdapter(config=cfg, candidate_id="X01").fit(X, y)


def test_fit_rejects_single_class_target(fast_config: dict):
    X, _ = _make_data(100)
    with pytest.raises(ValueError, match="both classes"):
        XGBoostAdapter(config=fast_config).fit(X, np.zeros(len(X), dtype=int))


# =============================================================================
# 3. Inner split isolation (T-013) and encoder locality (T-008)
# =============================================================================

def test_inner_split_uses_only_rows_passed_to_fit(fast_config: dict, monkeypatch):
    X, y = _make_data()
    spy = FitSpy().install(monkeypatch)
    adapter = XGBoostAdapter(config=fast_config, categorical_features=CATEGORICALS).fit(X, y)

    all_tags = set(X["Row_Tag"].astype(int))
    selection_calls = [c for c in spy.calls if c["has_eval_set"]]
    assert len(selection_calls) == 8

    inner_train = selection_calls[0]["train_tags"]
    inner_val = selection_calls[0]["eval_tags"]
    for call in selection_calls:
        assert call["train_tags"] == inner_train, "All candidates must share one inner split."
        assert call["eval_tags"] == inner_val
    assert inner_train | inner_val == all_tags
    assert not (inner_train & inner_val)
    assert abs(len(inner_val) - 0.15 * len(X)) <= 1

    # Stratified: inner-val positive rate close to the outer-train rate.
    val_mask = X["Row_Tag"].astype(int).isin(inner_val).to_numpy()
    assert abs(y.to_numpy()[val_mask].mean() - y.mean()) < 0.02

    # Adapter bookkeeping agrees with what XGBoost actually saw.
    assert set(X.loc[adapter.inner_validation_index_, "Row_Tag"].astype(int)) == inner_val
    assert set(X.loc[adapter.inner_training_index_, "Row_Tag"].astype(int)) == inner_train


def test_inner_split_is_seeded_and_reproducible(fast_config: dict):
    X, y = _make_data()
    a = XGBoostAdapter(config=fast_config).fit(X, y)
    b = XGBoostAdapter(config=fast_config).fit(X, y)
    assert a.inner_validation_index_.equals(b.inner_validation_index_)
    c = XGBoostAdapter(config=fast_config, random_seed=123).fit(X, y)
    assert not a.inner_validation_index_.equals(c.inner_validation_index_)


def test_final_encoder_learns_only_fit_rows_and_ignores_unseen_categories(fast_config: dict):
    """T-008: categories come only from fit rows; unseen values encode to all zeros."""
    X, y = _make_data()
    X_train = X[X["Department"] != "Finance"]
    y_train = y.loc[X_train.index]
    adapter = XGBoostAdapter(config=fast_config, categorical_features=CATEGORICALS).fit(X_train, y_train)

    dept_categories = set(adapter._encoder.categories_[CATEGORICALS.index("Department")])
    assert dept_categories == {"IT", "HR", "Sales"}
    assert "Department_Finance" not in adapter.get_metadata()["encoded_feature_names"]

    X_new = X.head(20).copy()
    X_new["Department"] = "Finance"
    X_new.loc[X_new.index[:5], "Job_Title"] = "Never_Seen_Title"
    probs = adapter.predict_proba(X_new)
    assert probs.shape == (20,) and np.isfinite(probs).all()


# =============================================================================
# 4. Candidate selection logic
# =============================================================================

def test_selected_candidate_is_argmax_of_inner_average_precision(fast_config: dict):
    X, y = _make_data()
    adapter = XGBoostAdapter(config=fast_config).fit(X, y)
    evidence = pd.DataFrame(adapter.candidate_selection_evidence_)

    assert list(evidence["candidate_id"]) == [f"X{i:02d}" for i in range(1, 9)]
    assert evidence["selected"].sum() == 1
    best_ap = evidence["inner_average_precision"].max()
    first_best = evidence.loc[evidence["inner_average_precision"] == best_ap, "candidate_id"].iloc[0]
    assert adapter.selected_candidate_id_ == first_best
    for col in ("inner_roc_auc", "inner_brier_score", "best_iteration", "fit_time_seconds"):
        assert evidence[col].notna().all()


def test_selection_follows_inner_scores_deterministically(fast_config: dict, monkeypatch):
    """Controlled scores: the 6th evaluated candidate (X06) must win."""
    import sklearn.metrics

    scores = iter([0.10, 0.12, 0.11, 0.09, 0.13, 0.30, 0.20, 0.29])
    monkeypatch.setattr(sklearn.metrics, "average_precision_score", lambda *a, **k: next(scores))
    X, y = _make_data()
    adapter = XGBoostAdapter(config=fast_config).fit(X, y)
    assert adapter.selected_candidate_id_ == "X06"
    assert adapter.get_metadata()["selected_candidate_id"] == "X06"


def test_selection_tie_keeps_earliest_declared_candidate(fast_config: dict, monkeypatch):
    import sklearn.metrics

    monkeypatch.setattr(sklearn.metrics, "average_precision_score", lambda *a, **k: 0.15)
    X, y = _make_data()
    adapter = XGBoostAdapter(config=fast_config).fit(X, y)
    assert adapter.selected_candidate_id_ == "X01"


def test_dynamic_ratio_uses_inner_training_labels_only(fast_config: dict, monkeypatch):
    X, y = _make_data()
    spy = FitSpy().install(monkeypatch)
    adapter = XGBoostAdapter(config=fast_config).fit(X, y)

    weighting = {c["id"]: c["params"]["weighting"] for c in fast_config["candidates"]}
    for record, call in zip(adapter.candidate_selection_evidence_, [c for c in spy.calls if c["has_eval_set"]]):
        expected = _ratio(call["train_y"]) if weighting[record["candidate_id"]] == "dynamic_ratio" else 1.0
        assert call["scale_pos_weight"] == pytest.approx(expected)
        assert record["scale_pos_weight"] == pytest.approx(expected)
    # Guard that the test can tell the two apart: the inner-train ratio differs
    # from the full outer-train ratio, so using the wrong labels would fail above.
    inner_ratio = _ratio(spy.calls[0]["train_y"])
    assert inner_ratio != pytest.approx(_ratio(y.to_numpy()), abs=1e-9)
    # The semantic 'weighting' directive must never be forwarded to XGBoost.
    for call in spy.calls:
        assert "weighting" not in call["extra_kwargs"]


# =============================================================================
# 5. Outer refit semantics
# =============================================================================

@pytest.mark.parametrize("candidate_id", ["X01", "X02"])
def test_refit_uses_all_outer_rows_and_selected_tree_count(fast_config: dict, monkeypatch, candidate_id):
    X, y = _make_data()
    spy = FitSpy().install(monkeypatch)
    adapter = XGBoostAdapter(config=fast_config, candidate_id=candidate_id).fit(X, y)

    assert len(spy.calls) == 2, "One inner selection fit + one refit expected."
    selection, refit = spy.calls
    assert selection["has_eval_set"] and selection["early_stopping_rounds"] == 10

    assert not refit["has_eval_set"]
    assert refit["early_stopping_rounds"] is None
    assert refit["train_tags"] == set(X["Row_Tag"].astype(int))
    best_iteration = adapter.candidate_selection_evidence_[0]["best_iteration"]
    assert refit["n_estimators"] == best_iteration + 1 == adapter.selected_n_estimators_
    assert adapter._model.get_booster().num_boosted_rounds() == adapter.selected_n_estimators_

    expected_spw = _ratio(y.to_numpy()) if candidate_id == "X02" else 1.0
    assert refit["scale_pos_weight"] == pytest.approx(expected_spw)
    assert adapter.final_scale_pos_weight_ == pytest.approx(expected_spw)

    meta = adapter.get_metadata()
    assert meta["fitted_on_all_outer_rows"] is True
    assert meta["n_outer_train_rows"] == len(X)
    assert meta["params"]["n_estimators"] == adapter.selected_n_estimators_
    assert "weighting" not in meta["params"]


def test_no_refit_option_keeps_inner_winner(fast_config: dict, monkeypatch):
    X, y = _make_data()
    spy = FitSpy().install(monkeypatch)
    adapter = XGBoostAdapter(config=fast_config, candidate_id="X01", refit_outer_train=False).fit(X, y)
    assert len(spy.calls) == 1
    assert adapter.get_metadata()["fitted_on_all_outer_rows"] is False
    assert np.isfinite(adapter.predict_proba(X)).all()


# =============================================================================
# 6. Output validity
# =============================================================================

def test_predict_proba_is_bounded_1d_and_column_order_invariant(fast_config: dict):
    """T-016."""
    X, y = _make_data()
    adapter = XGBoostAdapter(config=fast_config).fit(X.iloc[:500], y.iloc[:500])
    probs = adapter.predict_proba(X.iloc[500:])
    assert probs.shape == (100,) and probs.dtype == float
    assert np.isfinite(probs).all()
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    shuffled = X.iloc[500:][list(reversed(X.columns))]
    np.testing.assert_array_equal(probs, adapter.predict_proba(shuffled))


def test_predict_proba_rejects_missing_columns(fast_config: dict):
    X, y = _make_data()
    adapter = XGBoostAdapter(config=fast_config).fit(X, y)
    with pytest.raises(ValueError, match="missing fitted feature columns"):
        adapter.predict_proba(X.drop(columns=["Age"]))


def test_full_fit_is_deterministic(fast_config: dict):
    X, y = _make_data()
    a = XGBoostAdapter(config=fast_config).fit(X, y)
    b = XGBoostAdapter(config=fast_config).fit(X, y)
    assert a.selected_candidate_id_ == b.selected_candidate_id_
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def test_feature_importances_map_back_to_original_features(fast_config: dict):
    X, y = _make_data()
    adapter = XGBoostAdapter(config=fast_config, categorical_features=CATEGORICALS).fit(X, y)
    fi = adapter.feature_importances("gain")
    assert set(fi["original_feature"]) == set(X.columns)
    assert list(fi["encoded_feature"]) == adapter.get_metadata()["encoded_feature_names"]
    assert (fi["importance"] >= 0).all()


# =============================================================================
# 7. Serialization (T-023)
# =============================================================================

def test_save_load_round_trip_reproduces_predictions_and_metadata(fast_config: dict, tmp_path: Path):
    X, y = _make_data()
    adapter = XGBoostAdapter(config=fast_config, categorical_features=CATEGORICALS).fit(X.iloc[:500], y.iloc[:500])
    X_new = X.iloc[500:].copy()
    X_new.loc[X_new.index[:3], "Department"] = "Unseen_Dept"
    expected = adapter.predict_proba(X_new)

    saved_meta = adapter.save(tmp_path)
    assert (tmp_path / ARTIFACT_FILENAME).exists()
    meta_json = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert meta_json["selected_candidate_id"] == adapter.selected_candidate_id_
    assert len(meta_json["candidate_selection_evidence"]) == 8
    assert saved_meta["artifact_file"] == ARTIFACT_FILENAME

    restored = XGBoostAdapter.load(tmp_path)
    assert isinstance(restored, ModelAdapter)
    np.testing.assert_array_equal(expected, restored.predict_proba(X_new))

    original_meta, restored_meta = adapter.get_metadata(), restored.get_metadata()
    for key in (
        "name", "model_id", "fitted", "mode", "selected_candidate_id", "selected_n_estimators",
        "final_scale_pos_weight", "params", "feature_names", "encoded_feature_names",
        "candidate_selection_evidence", "fitted_on_all_outer_rows", "random_seed",
    ):
        assert restored_meta[key] == original_meta[key], key
    assert list(restored._encoder.categories_[0]) == list(adapter._encoder.categories_[0])


def test_save_unfitted_and_load_missing_raise(tmp_path: Path):
    with pytest.raises(RuntimeError):
        XGBoostAdapter().save(tmp_path)
    with pytest.raises(FileNotFoundError):
        XGBoostAdapter.load(tmp_path / "does_not_exist")


# =============================================================================
# 8. Direct mode (backward compatibility with the fixed-parameter notebook)
# =============================================================================

def test_direct_mode_single_fit_no_inner_split(monkeypatch):
    X, y = _make_data()
    params = {
        "n_estimators": 25, "learning_rate": 0.05, "max_depth": 3,
        "objective": "binary:logistic", "tree_method": "hist",
        "random_state": 42, "verbosity": 0, "n_jobs": 1, "scale_pos_weight": 4.0,
    }
    spy = FitSpy().install(monkeypatch)
    adapter = XGBoostAdapter(name="XGBoost", model_id="xgboost", params=params,
                             categorical_features=CATEGORICALS).fit(X, y)
    assert len(spy.calls) == 1 and not spy.calls[0]["has_eval_set"]
    assert spy.calls[0]["scale_pos_weight"] == 4.0
    assert adapter.mode == "direct"
    assert adapter.candidate_selection_evidence_ == []
    assert adapter._model.get_booster().num_boosted_rounds() == 25
    assert adapter._fitted_columns and adapter._model is not None  # used by the current notebook


# =============================================================================
# 9. Integration with the shared runner: outer-validation and holdout isolation
# =============================================================================

@pytest.mark.parametrize("variant", ["P1_onehot", "P2_native_categorical", "P3_onehot_no_salary"])
def test_runner_never_exposes_outer_validation_or_holdout_rows(fast_config: dict, monkeypatch, variant):
    """Every row XGBoost trains or early-stops on belongs to that fold's outer-training set, for every variant."""
    n_dev, n_holdout = 800, 200
    X_all, y_all = _make_data(n_dev + n_holdout, seed=11)
    ids = [f"EMP_{i:04d}" for i in range(n_dev + n_holdout)]
    registry = pd.DataFrame(
        {
            "Employee_ID": ids,
            "partition": ["development"] * n_dev + ["holdout"] * n_holdout,
            "cv_fold": [1, 2, 3, 4, 5] * (n_dev // 5) + [np.nan] * n_holdout,
        }
    )
    data = X_all.iloc[:n_dev].copy()
    data.insert(0, "Employee_ID", ids[:n_dev])
    data["Resigned"] = y_all.iloc[:n_dev].to_numpy()
    feature_cols = [c for c in X_all.columns]

    tag_to_fold = dict(zip(data["Row_Tag"].astype(int), registry["cv_fold"].iloc[:n_dev].astype(int)))
    holdout_tags = set(X_all["Row_Tag"].iloc[n_dev:].astype(int))

    spy = FitSpy().install(monkeypatch)
    adapters: list[XGBoostAdapter] = []

    def factory() -> XGBoostAdapter:
        adapter = XGBoostAdapter(config=fast_config, categorical_features=CATEGORICALS, preprocessing=variant)
        adapters.append(adapter)
        return adapter

    result = run_5fold_cv(factory, data, registry, feature_cols, expected_model_name="XGBoost")

    assert len(adapters) == 5 and len({id(a) for a in adapters}) == 5
    assert len(spy.calls) == 5 * 9  # 8 inner candidate fits + 1 refit per fold
    for fold_idx in range(5):
        fold = fold_idx + 1
        fold_calls = spy.calls[fold_idx * 9:(fold_idx + 1) * 9]
        outer_val_tags = {t for t, f in tag_to_fold.items() if f == fold}
        outer_train_tags = {t for t, f in tag_to_fold.items() if f != fold}
        for call in fold_calls:
            seen = call["train_tags"] | call["eval_tags"]
            assert not (seen & outer_val_tags), f"Fold {fold}: outer-validation rows reached XGBoost"
            assert not (seen & holdout_tags), f"Fold {fold}: holdout rows reached XGBoost"
            assert seen <= outer_train_tags
        assert fold_calls[-1]["train_tags"] == outer_train_tags  # refit on 100% outer-train

    preds = result.predictions
    assert len(preds) == n_dev and preds["Employee_ID"].is_unique
    assert preds["y_probability"].between(0, 1).all()
    assert result.model_name == "XGBoost"



# =============================================================================
# 10. Review fixes: patience, parameter precedence, seed, candidate cap
# =============================================================================

@pytest.mark.parametrize("patience", [0, None])
def test_disabled_early_stopping_still_runs_inner_selection(fast_config: dict, monkeypatch, patience):
    """Zero/None patience disables early stopping but never skips candidate selection."""
    cfg = copy.deepcopy(fast_config)
    cfg["inner_validation"]["early_stopping_rounds"] = patience
    X, y = _make_data()
    spy = FitSpy().install(monkeypatch)
    adapter = XGBoostAdapter(config=cfg).fit(X, y)

    assert len(adapter.candidate_selection_evidence_) == 8
    assert adapter.selected_candidate_id_ in {f"X{i:02d}" for i in range(1, 9)}
    assert len(spy.calls) == 9
    selection_calls, refit = spy.calls[:8], spy.calls[8]
    assert all(not c["has_eval_set"] and c["early_stopping_rounds"] is None for c in selection_calls)
    assert all(c["n_estimators"] == 60 for c in selection_calls)  # full ceiling, no early stop
    assert adapter.selected_n_estimators_ == 60 and refit["n_estimators"] == 60
    meta = adapter.get_metadata()
    assert meta["early_stopping_enabled"] is False and meta["mode"] == "nested"


@pytest.mark.parametrize("bad", [-1, 2.5, True])
def test_invalid_patience_raises(fast_config: dict, bad):
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        XGBoostAdapter(config=fast_config, early_stopping_rounds=bad)


def test_params_overrides_apply_identically_to_raw_and_resolved_configs(fast_config: dict, monkeypatch):
    """Precedence fixed -> candidate -> overrides holds for both config forms."""
    X, y = _make_data()
    overrides = {"max_depth": 2, "n_estimators": 25}
    spy = FitSpy().install(monkeypatch)
    raw = XGBoostAdapter(config=fast_config, candidate_id="X05", params=overrides).fit(X, y)
    raw_calls = list(spy.calls)
    spy.calls.clear()
    resolved = XGBoostAdapter(config=resolve_candidate(fast_config, "X05"), params=overrides).fit(X, y)

    for calls in (raw_calls, spy.calls):
        assert calls[0]["n_estimators"] == 25  # override beats the 60-round ceiling
    for adapter in (raw, resolved):
        assert adapter.candidate_selection_evidence_[0]["max_boosting_rounds"] == 25
        est = adapter._model.get_params()
        assert est["max_depth"] == 2  # override beats candidate X05's depth 7
        assert est["learning_rate"] == pytest.approx(0.05)  # untouched candidate value kept
    assert raw.get_metadata()["candidate_pool_ids"] == resolved.get_metadata()["candidate_pool_ids"] == ["X05"]
    np.testing.assert_array_equal(raw.predict_proba(X), resolved.predict_proba(X))


def test_override_scale_pos_weight_replaces_candidate_weighting(fast_config: dict, monkeypatch):
    X, y = _make_data()
    spy = FitSpy().install(monkeypatch)
    XGBoostAdapter(config=fast_config, candidate_id="X02", params={"scale_pos_weight": 3.0}).fit(X, y)
    assert all(c["scale_pos_weight"] == 3.0 for c in spy.calls)
    spy.calls.clear()
    XGBoostAdapter(config=fast_config, candidate_id="X02", params={"weighting": "none"}).fit(X, y)
    assert all(c["scale_pos_weight"] == 1.0 for c in spy.calls)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"weighting": "dynamic_ratio", "scale_pos_weight": 2.0}, "not both"),
        ({"weighting": "balanced"}, "Invalid weighting"),
        ({"early_stopping_rounds": 5}, "cannot be passed in params"),
        ({"enable_categorical": True}, "cannot be passed in params"),
    ],
)
def test_ambiguous_or_reserved_overrides_raise(fast_config: dict, params, message):
    with pytest.raises(ValueError, match=message):
        XGBoostAdapter(config=fast_config, params=params)


def test_seed_precedence_and_conflict(fast_config: dict):
    assert XGBoostAdapter(config=fast_config).random_seed == 42
    assert XGBoostAdapter(config=fast_config, params={"random_state": 7}).random_seed == 7
    assert XGBoostAdapter(config=fast_config, random_seed=9).params["random_state"] == 9
    with pytest.raises(ValueError, match="conflicts"):
        XGBoostAdapter(config=fast_config, random_seed=9, params={"random_state": 7})


def test_resolved_config_with_conflicting_candidate_id_raises(fast_config: dict):
    with pytest.raises(ValueError, match="conflicts with resolved candidate"):
        XGBoostAdapter(config=resolve_candidate(fast_config, "X01"), candidate_id="X02")


def test_candidate_cap_is_enforced(xgb_config: dict):
    """T-014: the adapter refuses registries above the frozen cap."""
    too_many = copy.deepcopy(xgb_config)
    too_many["candidates"] = [dict(c, id=f"Z{i:02d}") for i, c in enumerate(xgb_config["candidates"] * 2)]
    too_many["search_budget"]["declared_candidates"] = 16
    with pytest.raises(ValueError, match="exceeding the frozen cap of 12"):
        XGBoostAdapter(config=too_many)

    with pytest.raises(ValueError, match="exceeding the frozen cap of 4"):
        XGBoostAdapter(config=xgb_config, max_candidates=4)

    with pytest.raises(ValueError, match="canonical 12"):
        XGBoostAdapter(config=xgb_config, max_candidates=13)

    raised_cap = copy.deepcopy(xgb_config)
    raised_cap["search_budget"]["max_candidates"] = 20
    with pytest.raises(ValueError, match="canonical 12"):
        XGBoostAdapter(config=raised_cap)

    miscounted = copy.deepcopy(xgb_config)
    miscounted["search_budget"]["declared_candidates"] = 7
    with pytest.raises(ValueError, match="declared_candidates"):
        XGBoostAdapter(config=miscounted)


# =============================================================================
# 11. Preprocessing variants
# =============================================================================

def test_config_declares_primary_and_sensitivity_variants(xgb_config: dict):
    registry = xgb_config["preprocessing_variants"]
    variants = {v["id"]: v for v in registry["variants"]}
    assert registry["primary"] == "P1_onehot"
    assert variants["P1_onehot"]["role"] == "primary" and variants["P1_onehot"]["encoding"] == "onehot"
    assert variants["P2_native_categorical"]["encoding"] == "native"
    assert variants["P3_onehot_no_salary"]["drop_features"] == ["Monthly_Salary"]
    assert all(v["rationale"] for v in variants.values())
    assert sum(v["role"] == "primary" for v in variants.values()) == 1
    assert XGBoostAdapter(config=xgb_config).preprocessing_spec["id"] == "P1_onehot"


def test_native_variant_learns_vocabulary_from_fit_rows_only(fast_config: dict, monkeypatch):
    X, y = _make_data()
    X_train = X[X["Department"] != "Finance"]
    y_train = y.loc[X_train.index]
    seen_dtypes: list[dict] = []
    original_fit = xgboost.XGBClassifier.fit

    def recording_fit(model, X_, y_, *a, **k):
        seen_dtypes.append({c: str(t) for c, t in X_.dtypes.items()})
        assert model.get_params().get("enable_categorical") is True
        return original_fit(model, X_, y_, *a, **k)

    monkeypatch.setattr(xgboost.XGBClassifier, "fit", recording_fit)
    adapter = XGBoostAdapter(config=fast_config, preprocessing="P2_native_categorical",
                             categorical_features=CATEGORICALS).fit(X_train, y_train)

    assert all(d["Department"] == "category" and d["Age"] == "float64" for d in seen_dtypes)
    assert adapter._encoder["Department"] == ["HR", "IT", "Sales"]
    assert set(adapter.get_metadata()["encoded_feature_names"]) == set(X.columns)  # no one-hot columns
    X_new = X.head(10).copy()
    X_new["Department"] = "Finance"  # unseen -> missing
    assert np.isfinite(adapter.predict_proba(X_new)).all()


def test_drop_feature_variant_never_shows_dropped_column_to_xgboost(fast_config: dict, monkeypatch):
    X, y = _make_data()
    spy = FitSpy().install(monkeypatch)
    adapter = XGBoostAdapter(config=fast_config, preprocessing="P3_onehot_no_salary").fit(X, y)
    assert all("Monthly_Salary" not in c["columns"] for c in spy.calls)
    meta = adapter.get_metadata()
    assert "Monthly_Salary" in meta["input_feature_names"] and "Monthly_Salary" not in meta["feature_names"]
    assert adapter.predict_proba(X).shape == (len(X),)  # full contract input still accepted


def test_all_variants_share_the_same_inner_split(fast_config: dict):
    """Variants differ only in representation, so they are compared on identical inner rows."""
    X, y = _make_data()
    splits = [
        XGBoostAdapter(config=fast_config, preprocessing=v).fit(X, y).inner_validation_index_
        for v in ("P1_onehot", "P2_native_categorical", "P3_onehot_no_salary")
    ]
    assert splits[0].equals(splits[1]) and splits[0].equals(splits[2])


def test_invalid_preprocessing_requests_raise(fast_config: dict):
    with pytest.raises(ValueError, match="Unknown preprocessing variant"):
        XGBoostAdapter(config=fast_config, preprocessing="P9_missing")
    with pytest.raises(ValueError, match="Invalid encoding"):
        XGBoostAdapter(config=fast_config, preprocessing={"id": "bad", "encoding": "target"})
    X, y = _make_data()
    with pytest.raises(ValueError, match="not present in X"):
        XGBoostAdapter(config=fast_config, preprocessing={"id": "d", "encoding": "onehot",
                                                           "drop_features": ["Not_A_Column"]}).fit(X, y)


def test_native_variant_save_load_round_trip(fast_config: dict, tmp_path: Path):
    X, y = _make_data()
    adapter = XGBoostAdapter(config=fast_config, preprocessing="P2_native_categorical").fit(X, y)
    adapter.save(tmp_path)
    restored = XGBoostAdapter.load(tmp_path)
    np.testing.assert_array_equal(adapter.predict_proba(X), restored.predict_proba(X))
    assert restored.get_metadata()["preprocessing"]["id"] == "P2_native_categorical"


def test_load_rejects_unknown_artifact_format(tmp_path: Path):
    import pickle

    (tmp_path / ARTIFACT_FILENAME).write_bytes(pickle.dumps({"init": {}, "fit": {}}))
    with pytest.raises(ValueError, match="Unsupported XGBoost artifact format"):
        XGBoostAdapter.load(tmp_path)


# =============================================================================
# 12. Tree-count diagnostic (300/600/1000 versus early stopping)
# =============================================================================

def test_tree_count_profile_uses_inner_rows_only(fast_config: dict, monkeypatch):
    X, y = _make_data()
    adapter = XGBoostAdapter(config=fast_config).fit(X, y)
    spy = FitSpy().install(monkeypatch)
    profile = adapter.tree_count_profile(X, y, checkpoints=(10, 20, 30))

    assert len(profile) == 8 * 3
    assert set(profile["n_estimators"]) == {10, 20, 30}
    assert profile["inner_average_precision"].between(0, 1).all()
    assert adapter.profile_inner_validation_index_.equals(adapter.inner_validation_index_)
    inner_val_tags = set(X.loc[adapter.inner_validation_index_, "Row_Tag"].astype(int))
    for call in spy.calls:
        assert not call["has_eval_set"] and call["n_estimators"] == 30
        assert not (call["train_tags"] & inner_val_tags)


def test_tree_count_profile_requires_nested_mode():
    X, y = _make_data(100)
    with pytest.raises(RuntimeError, match="nested"):
        XGBoostAdapter().tree_count_profile(X, y)
