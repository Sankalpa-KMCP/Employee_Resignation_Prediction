"""XGBoost Model Adapter for Employee Resignation Prediction (Member 3).

Conforms to the shared ``ModelAdapter`` protocol and implements the strict
nested outer-CV / inner-selection methodology (Project Context v4.6.2,
Sections 8.16, 8.17 step 11, 8.19.2 and 8.21.5).

Everything below happens inside ``fit(X, y)``, where ``X``/``y`` are the
outer-training rows handed over by ``src.models.runner.run_5fold_cv``. The
outer-validation fold and the formal holdout are never passed to this class
during fitting.

Nested procedure (config mode)
------------------------------
1. Stratified inner split of the outer-training rows
   (``inner_validation.fraction``, default 15%) seeded by the adapter seed.
2. The preprocessing variant (one-hot by default) is fitted on the
   inner-training split only.
3. Every candidate in the pool is trained on the inner-training split. With
   early stopping enabled (``early_stopping_rounds >= 1``) training stops on
   the inner-validation split; with it disabled (``None`` or ``0``) each
   candidate trains to its full ``n_estimators``. Inner selection is
   performed in both cases - it is never skipped.
   ``weighting: dynamic_ratio`` sets ``scale_pos_weight = n_neg / n_pos``
   from the labels of the rows being fitted.
4. The candidate with the highest inner-validation Average Precision wins
   (ties go to the earlier-declared candidate).
5. Refit: a fresh preprocessor and the winning candidate are fitted on 100%
   of the outer-training rows with the selected tree count and no evaluation
   set.

Parameter precedence (identical for raw and resolved configs)
-------------------------------------------------------------
``fixed_params`` (with ``n_estimators`` set to the boosting-round ceiling)
-> candidate ``params`` -> constructor ``params`` overrides -> seed.
The estimator ``random_state`` always equals the adapter seed, which is
taken from ``random_seed`` if given, else ``params["random_state"]``, else the
config seed. Conflicting or ambiguous requests raise ``ValueError`` instead of
being silently resolved.

Direct mode (``params=...`` without ``config``) trains one fixed parameter set
with no inner selection; it is kept for backward compatibility only.
"""

from __future__ import annotations

import copy
import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_CATEGORICAL_FEATURES: list[str] = [
    "Department",
    "Gender",
    "Job_Title",
    "Education_Level",
]

DEFAULT_DIRECT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}

# Canonical search cap for XGBoost (Project Context v4.6.2, Section 8.21.5).
CANONICAL_MAX_CANDIDATES = 12
VALID_WEIGHTING = frozenset({"none", "dynamic_ratio"})
VALID_ENCODINGS = frozenset({"onehot", "native"})
DEFAULT_PREPROCESSING: dict[str, Any] = {
    "id": "P1_onehot",
    "encoding": "onehot",
    "drop_features": [],
}
# Keys that must be set through dedicated constructor arguments, not ``params``.
RESERVED_OVERRIDE_KEYS = frozenset({"early_stopping_rounds", "enable_categorical"})

ARTIFACT_FILENAME = "xgboost_adapter.pkl"
METADATA_FILENAME = "xgboost_adapter_metadata.json"
ARTIFACT_FORMAT_VERSION = 2


def _to_json_safe(obj: Any) -> Any:
    """Convert numpy / pandas scalars and containers into JSON-serializable types."""
    if isinstance(obj, Mapping):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class XGBoostAdapter:
    """XGBoost adapter with fold-local preprocessing and nested inner selection.

    Parameters
    ----------
    config : Mapping or None
        The validated config from ``load_model_config("xgboost")`` (contains
        ``candidates``), or a dictionary from ``resolve_candidate`` (treated as
        that config restricted to the resolved candidate).
    candidate_id : str or None
        ``"auto"``/``"tune"``/``None`` selects among all declared candidates.
        A specific id trains only that candidate (still inner-evaluated and
        refitted).
    params : dict or None
        Config mode: estimator overrides applied *after* candidate params, for
        every candidate. Direct mode (no config): the full estimator params.
    name, model_id : str or None
        Identity labels; default to the config values, else XGBoost/xgboost.
    categorical_features : sequence of str or None
        Nominal columns (default: the four contract categoricals).
    inner_val_fraction, early_stopping_rounds, refit_outer_train, random_seed
        Overrides for the config's inner-validation policy and seed.
        ``early_stopping_rounds`` of ``None`` or ``0`` disables early stopping
        (candidates train to ``n_estimators``); selection still happens.
    preprocessing : str, Mapping or None
        Preprocessing variant: an id from ``config["preprocessing_variants"]``,
        an explicit mapping ``{"id", "encoding", "drop_features"}``, or
        ``None`` for the config's primary variant (one-hot if undeclared).
    max_candidates : int or None
        Search cap; defaults to ``search_budget.max_candidates`` and may never
        exceed the canonical cap of 12.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        candidate_id: str | None = None,
        params: dict[str, Any] | None = None,
        name: str | None = None,
        model_id: str | None = None,
        categorical_features: Sequence[str] | None = None,
        inner_val_fraction: float | None = None,
        early_stopping_rounds: int | None = None,
        refit_outer_train: bool | None = None,
        random_seed: int | None = None,
        preprocessing: str | Mapping[str, Any] | None = None,
        max_candidates: int | None = None,
    ) -> None:
        self.categorical_features: list[str] = (
            list(categorical_features)
            if categorical_features is not None
            else list(DEFAULT_CATEGORICAL_FEATURES)
        )
        overrides = copy.deepcopy(dict(params)) if params is not None else {}

        if config is not None:
            self._init_config_mode(
                config, candidate_id, overrides, name, model_id,
                inner_val_fraction, early_stopping_rounds, refit_outer_train,
                random_seed, preprocessing, max_candidates,
            )
        else:
            self._init_direct_mode(
                candidate_id, params, overrides, name, model_id,
                inner_val_fraction, early_stopping_rounds, refit_outer_train,
                random_seed, preprocessing,
            )

        self.params: dict[str, Any] = copy.deepcopy(self._fixed_params)
        self.params.update(copy.deepcopy(self._overrides))
        self.params["random_state"] = self.random_seed
        self._reset_fit_state()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _init_config_mode(
        self, config, candidate_id, overrides, name, model_id,
        inner_val_fraction, early_stopping_rounds, refit_outer_train,
        random_seed, preprocessing, max_candidates,
    ) -> None:
        self.mode = "nested"

        # A resolved candidate dict is normalised back to its raw config so that
        # raw and resolved inputs follow exactly the same precedence rules.
        if "params" in config and isinstance(config["params"], Mapping) and "candidates" not in config:
            raw = config.get("raw_config")
            resolved_id = str(config.get("candidate_id", candidate_id or "default"))
            if isinstance(raw, Mapping) and "candidates" in raw:
                if candidate_id not in (None, resolved_id):
                    raise ValueError(
                        f"candidate_id {candidate_id!r} conflicts with resolved candidate {resolved_id!r}."
                    )
                config, candidate_id = raw, resolved_id
            else:
                config = {
                    "model": config.get("model", {}),
                    "reproducibility": config.get("reproducibility", {}),
                    "inner_validation": config.get("inner_validation", {}),
                    "fixed_params": dict(config["params"]),
                    "candidates": [{"id": resolved_id, "params": {}}],
                }
                candidate_id = resolved_id

        if "candidates" not in config or not isinstance(config["candidates"], Sequence):
            raise ValueError("XGBoostAdapter config must declare a 'candidates' list.")

        all_candidates = [copy.deepcopy(dict(c)) for c in config["candidates"] if isinstance(c, Mapping)]
        ids = [str(c.get("id")) for c in all_candidates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate candidate ids in config: {ids}")

        # ---- Search cap (T-014) ----
        budget = config.get("search_budget", {}) or {}
        cap = max_candidates if max_candidates is not None else budget.get("max_candidates", CANONICAL_MAX_CANDIDATES)
        cap = int(cap)
        if not 1 <= cap <= CANONICAL_MAX_CANDIDATES:
            raise ValueError(
                f"Candidate cap must be between 1 and the canonical {CANONICAL_MAX_CANDIDATES}, got {cap}."
            )
        if len(all_candidates) > cap:
            raise ValueError(
                f"Config declares {len(all_candidates)} candidates, exceeding the frozen cap of {cap}."
            )
        declared = budget.get("declared_candidates")
        if declared is not None and int(declared) != len(all_candidates):
            raise ValueError(
                f"search_budget.declared_candidates={declared} but {len(all_candidates)} candidates are listed."
            )
        self.max_candidates = cap

        if candidate_id in (None, "auto", "tune"):
            self.candidate_id = "auto"
            self._candidate_pool = all_candidates
        else:
            matches = [c for c in all_candidates if str(c.get("id")) == str(candidate_id)]
            if not matches:
                raise ValueError(f"Unknown candidate_id {candidate_id!r}. Available: {ids}")
            self.candidate_id = str(candidate_id)
            self._candidate_pool = matches

        model_meta = config.get("model", {}) or {}
        repro = config.get("reproducibility", {}) or {}
        inner_val = config.get("inner_validation", {}) or {}

        self.name = name or str(model_meta.get("name", "XGBoost"))
        self.model_id = model_id or str(model_meta.get("model_id", "xgboost"))

        self._overrides = self._validate_overrides(overrides)
        self.random_seed = self._resolve_seed(
            random_seed, self._overrides, repro.get("random_seed", repro.get("seed", 42))
        )

        self.inner_val_fraction = float(
            inner_val_fraction if inner_val_fraction is not None else inner_val.get("fraction", 0.15)
        )
        if not 0.0 < self.inner_val_fraction < 0.5:
            raise ValueError(
                f"inner_val_fraction must be in (0, 0.5) for nested mode, got {self.inner_val_fraction}."
            )
        es = early_stopping_rounds if early_stopping_rounds is not None else inner_val.get("early_stopping_rounds")
        self.early_stopping_rounds = self._validate_patience(es)
        self.refit_outer_train = bool(
            refit_outer_train if refit_outer_train is not None else inner_val.get("refit_outer_train", True)
        )

        self._fixed_params = copy.deepcopy(dict(config.get("fixed_params", {}) or {}))
        ceiling = inner_val.get("max_boosting_rounds")
        if ceiling is not None:
            self._fixed_params["n_estimators"] = int(ceiling)
        self.max_boosting_rounds = int(self._fixed_params.get("n_estimators", 100))

        self.preprocessing_spec = self._resolve_preprocessing(preprocessing, config)

    def _init_direct_mode(
        self, candidate_id, params, overrides, name, model_id,
        inner_val_fraction, early_stopping_rounds, refit_outer_train,
        random_seed, preprocessing,
    ) -> None:
        self.mode = "direct"
        self.name = name or "XGBoost"
        self.model_id = model_id or "xgboost"
        reserved = RESERVED_OVERRIDE_KEYS & set(overrides)
        if reserved:
            raise ValueError(f"Keys {sorted(reserved)} cannot be passed in params; use constructor arguments.")
        self._fixed_params = copy.deepcopy(params) if params is not None else copy.deepcopy(DEFAULT_DIRECT_PARAMS)
        self._overrides = {}
        self.random_seed = self._resolve_seed(random_seed, self._fixed_params, 42)
        self.candidate_id = candidate_id or ("direct" if params is not None else "default")
        self._candidate_pool = [{"id": self.candidate_id, "params": {}}]
        self.max_candidates = 1
        self.inner_val_fraction = float(inner_val_fraction) if inner_val_fraction is not None else 0.0
        self.early_stopping_rounds = self._validate_patience(early_stopping_rounds)
        self.refit_outer_train = bool(refit_outer_train) if refit_outer_train is not None else False
        self.max_boosting_rounds = int(self._fixed_params.get("n_estimators", 100))
        self.preprocessing_spec = self._resolve_preprocessing(preprocessing, None)

    @staticmethod
    def _validate_patience(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"early_stopping_rounds must be None or a non-negative integer, got {value!r}.")
        return int(value) if int(value) > 0 else None

    @staticmethod
    def _validate_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
        reserved = RESERVED_OVERRIDE_KEYS & set(overrides)
        if reserved:
            raise ValueError(
                f"Keys {sorted(reserved)} cannot be passed in params; use the dedicated constructor arguments."
            )
        if "scale_pos_weight" in overrides and "weighting" in overrides:
            raise ValueError("Pass either 'weighting' or 'scale_pos_weight' in params, not both.")
        if "weighting" in overrides and overrides["weighting"] not in VALID_WEIGHTING:
            raise ValueError(f"Invalid weighting {overrides['weighting']!r}; allowed: {sorted(VALID_WEIGHTING)}.")
        return overrides

    @staticmethod
    def _resolve_seed(random_seed: Any, params: Mapping[str, Any], fallback: Any) -> int:
        param_seed = params.get("random_state")
        if random_seed is not None and param_seed is not None and int(param_seed) != int(random_seed):
            raise ValueError(
                f"random_seed={random_seed} conflicts with params['random_state']={param_seed}."
            )
        if random_seed is not None:
            return int(random_seed)
        if param_seed is not None:
            return int(param_seed)
        return int(fallback)

    def _resolve_preprocessing(self, preprocessing: Any, config: Mapping[str, Any] | None) -> dict[str, Any]:
        registry = (config or {}).get("preprocessing_variants", {}) or {}
        variants = {str(v["id"]): dict(v) for v in registry.get("variants", []) if isinstance(v, Mapping)}
        if preprocessing is None:
            primary = registry.get("primary")
            spec = variants[str(primary)] if primary is not None else dict(DEFAULT_PREPROCESSING)
        elif isinstance(preprocessing, str):
            if preprocessing not in variants:
                raise ValueError(
                    f"Unknown preprocessing variant {preprocessing!r}. Available: {sorted(variants)}"
                )
            spec = variants[preprocessing]
        elif isinstance(preprocessing, Mapping):
            spec = dict(preprocessing)
        else:
            raise TypeError("preprocessing must be None, a variant id, or a mapping.")

        spec = {
            "id": str(spec.get("id", "custom")),
            "encoding": str(spec.get("encoding", "onehot")),
            "drop_features": [str(f) for f in (spec.get("drop_features") or [])],
            **{k: v for k, v in spec.items() if k not in ("id", "encoding", "drop_features")},
        }
        if spec["encoding"] not in VALID_ENCODINGS:
            raise ValueError(f"Invalid encoding {spec['encoding']!r}; allowed: {sorted(VALID_ENCODINGS)}.")
        return spec

    # ------------------------------------------------------------------
    # Fit-state helpers
    # ------------------------------------------------------------------
    def _reset_fit_state(self) -> None:
        self._model: Any = None
        self._encoder: Any = None
        self._input_feature_names: list[str] = []
        self._feature_names: list[str] = []
        self._fitted_columns: list[str] = []
        self._encoded_to_original: dict[str, str] = {}
        self._fitted = False
        self._fitted_on_all_outer_rows = False
        self.selected_candidate_id_: str | None = None
        self.selected_n_estimators_: int | None = None
        self.final_params_: dict[str, Any] = {}
        self.final_scale_pos_weight_: float | None = None
        self.candidate_selection_evidence_: list[dict[str, Any]] = []
        self.inner_validation_index_: pd.Index | None = None
        self.inner_training_index_: pd.Index | None = None
        self.n_outer_train_rows_: int = 0
        self.outer_train_positive_rate_: float | None = None
        self.candidate_selection_time_seconds_: float = 0.0
        self.refit_time_seconds_: float = 0.0
        self.total_fit_time_seconds_: float = 0.0
        self.profile_inner_validation_index_: pd.Index | None = None

    def _present_categoricals(self, X: pd.DataFrame) -> list[str]:
        return [c for c in self.categorical_features if c in X.columns]

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        drop = self.preprocessing_spec["drop_features"]
        missing = [c for c in drop if c not in X.columns]
        if missing:
            raise ValueError(f"Preprocessing variant drops features not present in X: {missing}")
        return X.drop(columns=drop)

    # ------------------------------------------------------------------
    # Preprocessing (fold-local)
    # ------------------------------------------------------------------
    def _fit_encoder(self, X: pd.DataFrame) -> Any:
        """Fit the variant's categorical preprocessor on the given rows only."""
        cat_cols = self._present_categoricals(X)
        if not cat_cols:
            return None
        if self.preprocessing_spec["encoding"] == "native":
            # Category vocabularies learned from these rows only; unseen -> missing.
            return {col: sorted(X[col].astype(str).unique().tolist()) for col in cat_cols}
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(X[cat_cols].astype(str))
        return encoder

    def _transform(self, X: pd.DataFrame, encoder: Any) -> pd.DataFrame:
        """Apply a fitted preprocessor; numerical columns are passed through unscaled."""
        cat_cols = self._present_categoricals(X)
        X_num = X.drop(columns=cat_cols).astype(float)
        if not cat_cols:
            return X_num.copy()
        if encoder is None:
            raise RuntimeError("Encoder is not fitted.")
        if self.preprocessing_spec["encoding"] == "native":
            columns = {}
            for col in cat_cols:
                values = X[col].astype(str)
                known = values.where(values.isin(encoder[col]))  # unseen -> missing (NaN)
                columns[col] = pd.Categorical(known, categories=encoder[col])
            cat_df = pd.DataFrame(columns, index=X.index)
        else:
            cat_df = pd.DataFrame(
                encoder.transform(X[cat_cols].astype(str)),
                columns=list(encoder.get_feature_names_out(cat_cols)),
                index=X.index,
            )
        return pd.concat([X_num, cat_df], axis=1)

    def _build_encoded_mapping(self, X: pd.DataFrame, encoder: Any) -> dict[str, str]:
        """Map each model input column back to its original contract feature."""
        cat_cols = self._present_categoricals(X)
        mapping = {col: col for col in X.columns if col not in cat_cols}
        if encoder is None:
            return mapping
        if self.preprocessing_spec["encoding"] == "native":
            mapping.update({col: col for col in cat_cols})
        else:
            for col, cats in zip(cat_cols, encoder.categories_):
                for cat in cats:
                    mapping[f"{col}_{cat}"] = col
        return mapping

    # ------------------------------------------------------------------
    # Estimator construction
    # ------------------------------------------------------------------
    @staticmethod
    def _positive_ratio(y: np.ndarray) -> float:
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0:
            raise ValueError("Cannot compute dynamic scale_pos_weight: no positive rows.")
        return float(n_neg / n_pos)

    def _estimator_params(
        self,
        candidate_params: Mapping[str, Any],
        y_train: np.ndarray,
    ) -> tuple[dict[str, Any], str, float]:
        """fixed -> candidate -> overrides -> seed; then translate ``weighting``.

        ``dynamic_ratio`` uses ``y_train``: always the labels of the rows being fitted.
        """
        merged = copy.deepcopy(self._fixed_params)
        merged.update(copy.deepcopy(dict(candidate_params)))
        if "scale_pos_weight" in self._overrides:
            merged.pop("weighting", None)  # explicit override replaces the candidate's weighting
        if "weighting" in self._overrides:
            merged.pop("scale_pos_weight", None)
        merged.update(copy.deepcopy(self._overrides))
        merged["random_state"] = self.random_seed
        if self.preprocessing_spec["encoding"] == "native":
            merged["enable_categorical"] = True

        weighting = merged.pop("weighting", None)
        if weighting is None:
            spw_raw = merged.get("scale_pos_weight", 1.0)
            if isinstance(spw_raw, str):
                if spw_raw.lower() not in ("auto", "dynamic_ratio"):
                    raise ValueError(f"Unsupported scale_pos_weight value {spw_raw!r}.")
                weighting = "dynamic_ratio"
            else:
                spw = float(spw_raw)
                merged["scale_pos_weight"] = spw
                return merged, ("none" if spw == 1.0 else "fixed"), spw

        if weighting not in VALID_WEIGHTING:
            raise ValueError(f"Invalid weighting {weighting!r}; allowed: {sorted(VALID_WEIGHTING)}.")
        spw = self._positive_ratio(y_train) if weighting == "dynamic_ratio" else 1.0
        merged["scale_pos_weight"] = spw
        return merged, weighting, spw

    @staticmethod
    def _new_classifier(params: Mapping[str, Any], early_stopping_rounds: int | None = None) -> Any:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "xgboost is required for XGBoostAdapter. Install with pip install xgboost."
            ) from exc
        kwargs = dict(params)
        kwargs.pop("early_stopping_rounds", None)
        if early_stopping_rounds:
            kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        return XGBClassifier(**kwargs)

    def _inner_split(self, X: pd.DataFrame, y_arr: np.ndarray):
        from sklearn.model_selection import train_test_split

        positions = np.arange(len(X))
        pos_tr, pos_val = train_test_split(
            positions,
            test_size=self.inner_val_fraction,
            stratify=y_arr,
            random_state=self.random_seed,
        )
        return np.sort(pos_tr), np.sort(pos_val)

    @staticmethod
    def _validate_xy(X: Any, y: Any) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with contract column names.")
        y_arr = np.asarray(y).astype(int)
        if len(y_arr) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y_arr)} labels.")
        if not set(np.unique(y_arr)).issubset({0, 1}) or len(np.unique(y_arr)) != 2:
            raise ValueError("y must be binary and contain both classes.")
        return y_arr

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> XGBoostAdapter:
        """Fit on outer-training rows only, with nested inner selection in config mode."""
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        y_arr = self._validate_xy(X, y)
        self._reset_fit_state()
        t_start = time.perf_counter()
        self._input_feature_names = list(X.columns)
        X = self._select_features(X)
        self._feature_names = list(X.columns)
        self.n_outer_train_rows_ = int(len(X))
        self.outer_train_positive_rate_ = float(y_arr.mean())

        if self.mode == "nested":
            # ---------- 1. Inner split carved from the outer-training rows ----------
            pos_tr, pos_val = self._inner_split(X, y_arr)
            X_in_tr, X_in_val = X.iloc[pos_tr], X.iloc[pos_val]
            y_in_tr, y_in_val = y_arr[pos_tr], y_arr[pos_val]
            self.inner_training_index_ = X_in_tr.index.copy()
            self.inner_validation_index_ = X_in_val.index.copy()

            # ---------- 2. Preprocessor fitted on inner-training rows only ----------
            inner_encoder = self._fit_encoder(X_in_tr)
            Xe_in_tr = self._transform(X_in_tr, inner_encoder)
            Xe_in_val = self._transform(X_in_val, inner_encoder)

            # ---------- 3. Candidate evaluation (always performed) ----------
            sel_start = time.perf_counter()
            best: dict[str, Any] | None = None
            best_model: Any = None
            for cand in self._candidate_pool:
                cid = str(cand.get("id"))
                c_start = time.perf_counter()
                est_params, weighting, spw = self._estimator_params(cand.get("params", {}), y_in_tr)
                max_rounds = int(est_params.get("n_estimators", 100))
                model = self._new_classifier(est_params, self.early_stopping_rounds)
                if self.early_stopping_rounds:
                    model.fit(Xe_in_tr, y_in_tr, eval_set=[(Xe_in_val, y_in_val)], verbose=False)
                    best_iter = int(model.best_iteration)
                else:
                    model.fit(Xe_in_tr, y_in_tr, verbose=False)
                    best_iter = max_rounds - 1
                prob = model.predict_proba(Xe_in_val, iteration_range=(0, best_iter + 1))[:, 1]
                record = {
                    "candidate_id": cid,
                    "description": cand.get("description", ""),
                    "preprocessing_variant": self.preprocessing_spec["id"],
                    "weighting": weighting,
                    "scale_pos_weight": float(spw),
                    "params": _to_json_safe(dict(cand.get("params", {}))),
                    "early_stopping": bool(self.early_stopping_rounds),
                    "best_iteration": best_iter,
                    "selected_n_estimators": best_iter + 1,
                    "max_boosting_rounds": max_rounds,
                    "hit_round_ceiling": bool(best_iter + 1 >= max_rounds),
                    "inner_average_precision": float(average_precision_score(y_in_val, prob)),
                    "inner_roc_auc": float(roc_auc_score(y_in_val, prob)),
                    "inner_brier_score": float(brier_score_loss(y_in_val, prob)),
                    "inner_train_rows": int(len(y_in_tr)),
                    "inner_val_rows": int(len(y_in_val)),
                    "fit_time_seconds": float(time.perf_counter() - c_start),
                    "selected": False,
                }
                self.candidate_selection_evidence_.append(record)
                # Strictly greater: ties keep the earlier-declared candidate.
                if best is None or record["inner_average_precision"] > best["inner_average_precision"]:
                    best = record
                    best_model = model
            self.candidate_selection_time_seconds_ = float(time.perf_counter() - sel_start)

            assert best is not None
            best["selected"] = True
            self.selected_candidate_id_ = best["candidate_id"]
            self.selected_n_estimators_ = int(best["selected_n_estimators"])
            winner = next(c for c in self._candidate_pool if str(c.get("id")) == best["candidate_id"])

            # ---------- 4. Refit winner on 100% of outer-training rows ----------
            if self.refit_outer_train:
                refit_start = time.perf_counter()
                self._encoder = self._fit_encoder(X)
                X_enc = self._transform(X, self._encoder)
                final_params, _, final_spw = self._estimator_params(winner.get("params", {}), y_arr)
                final_params["n_estimators"] = self.selected_n_estimators_
                self._model = self._new_classifier(final_params, early_stopping_rounds=None)
                self._model.fit(X_enc, y_arr, verbose=False)
                self.refit_time_seconds_ = float(time.perf_counter() - refit_start)
                self._fitted_on_all_outer_rows = True
                self._fitted_columns = list(X_enc.columns)
                self._encoded_to_original = self._build_encoded_mapping(X, self._encoder)
            else:
                self._encoder = inner_encoder
                self._model = best_model
                final_params, _, final_spw = self._estimator_params(winner.get("params", {}), y_in_tr)
                final_params["n_estimators"] = self.selected_n_estimators_
                self._fitted_on_all_outer_rows = False
                self._fitted_columns = list(Xe_in_tr.columns)
                self._encoded_to_original = self._build_encoded_mapping(X_in_tr, inner_encoder)

        else:
            # ---------- Direct mode: single fixed configuration, no inner split ----------
            refit_start = time.perf_counter()
            self._encoder = self._fit_encoder(X)
            X_enc = self._transform(X, self._encoder)
            final_params, _, final_spw = self._estimator_params({}, y_arr)
            self._model = self._new_classifier(final_params, early_stopping_rounds=None)
            self._model.fit(X_enc, y_arr, verbose=False)
            self.refit_time_seconds_ = float(time.perf_counter() - refit_start)
            self.selected_candidate_id_ = self.candidate_id
            self.selected_n_estimators_ = int(final_params.get("n_estimators", 100))
            self._fitted_on_all_outer_rows = True
            self._fitted_columns = list(X_enc.columns)
            self._encoded_to_original = self._build_encoded_mapping(X, self._encoder)

        self.final_params_ = _to_json_safe(final_params)
        self.final_scale_pos_weight_ = float(final_spw)
        self.params = copy.deepcopy(self.final_params_)
        self.total_fit_time_seconds_ = float(time.perf_counter() - t_start)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class (``Resigned=True``) probabilities, shape ``(N,)``."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Adapter is not fitted. Call fit() first.")
        missing = [c for c in self._feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing fitted feature columns: {missing}")
        X_enc = self._transform(X[self._feature_names], self._encoder)[self._fitted_columns]
        n_trees = self.selected_n_estimators_
        if n_trees is not None and not self._fitted_on_all_outer_rows:
            probs = self._model.predict_proba(X_enc, iteration_range=(0, n_trees))[:, 1]
        else:
            probs = self._model.predict_proba(X_enc)[:, 1]
        return np.asarray(probs, dtype=float)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def tree_count_profile(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        checkpoints: Sequence[int] = (300, 600, 1000),
    ) -> pd.DataFrame:
        """Inner-validation AP at fixed tree counts, for comparison with early stopping.

        Re-creates the same seeded inner split from the rows passed (outer-training
        rows only), trains every pool candidate WITHOUT early stopping up to
        ``max(checkpoints)`` rounds, and scores the inner-validation split at each
        checkpoint. Diagnostic only: it never changes the fitted model. The inner
        validation rows used are recorded in ``profile_inner_validation_index_``.
        """
        from sklearn.metrics import average_precision_score

        if self.mode != "nested":
            raise RuntimeError("tree_count_profile requires config (nested) mode.")
        y_arr = self._validate_xy(X, y)
        X = self._select_features(X)
        pos_tr, pos_val = self._inner_split(X, y_arr)
        X_in_tr, X_in_val = X.iloc[pos_tr], X.iloc[pos_val]
        y_in_tr, y_in_val = y_arr[pos_tr], y_arr[pos_val]
        encoder = self._fit_encoder(X_in_tr)
        Xe_tr, Xe_val = self._transform(X_in_tr, encoder), self._transform(X_in_val, encoder)

        rounds = int(max(checkpoints))
        rows = []
        for cand in self._candidate_pool:
            params, weighting, _ = self._estimator_params(cand.get("params", {}), y_in_tr)
            params["n_estimators"] = rounds
            model = self._new_classifier(params, early_stopping_rounds=None)
            model.fit(Xe_tr, y_in_tr, verbose=False)
            for cp in checkpoints:
                prob = model.predict_proba(Xe_val, iteration_range=(0, int(cp)))[:, 1]
                rows.append(
                    {
                        "candidate_id": str(cand.get("id")),
                        "weighting": weighting,
                        "n_estimators": int(cp),
                        "inner_average_precision": float(average_precision_score(y_in_val, prob)),
                    }
                )
        # Kept on the adapter (not in DataFrame.attrs, which pandas compares on concat).
        self.profile_inner_validation_index_ = X_in_val.index.copy()
        return pd.DataFrame(rows)

    def feature_importances(self, importance_type: str = "gain") -> pd.DataFrame:
        """Booster importances per model input column, mapped back to contract features.

        Columns never used in a split get 0. Model-behaviour diagnostics only.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Adapter is not fitted. Call fit() first.")
        scores = self._model.get_booster().get_score(importance_type=importance_type)
        return pd.DataFrame(
            {
                "encoded_feature": self._fitted_columns,
                "original_feature": [self._encoded_to_original.get(c, c) for c in self._fitted_columns],
                "importance": [float(scores.get(c, 0.0)) for c in self._fitted_columns],
            }
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, directory: Path) -> dict[str, Any]:
        """Serialize the fitted adapter (model, preprocessor, evidence) and metadata."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted adapter.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        artifact_file = directory / ARTIFACT_FILENAME
        with open(artifact_file, "wb") as f:
            pickle.dump({"format_version": ARTIFACT_FORMAT_VERSION, "state": self.__dict__}, f)

        metadata = self.get_metadata()
        metadata["artifact_file"] = artifact_file.name
        metadata["artifact_format_version"] = ARTIFACT_FORMAT_VERSION
        with open(directory / METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(metadata), f, indent=2)
        return metadata

    @classmethod
    def load(cls, directory: Path) -> XGBoostAdapter:
        """Restore a fitted adapter saved by :meth:`save`."""
        artifact_file = Path(directory) / ARTIFACT_FILENAME
        if not artifact_file.exists():
            raise FileNotFoundError(f"No saved XGBoost adapter at {artifact_file}")
        with open(artifact_file, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported XGBoost artifact format in {artifact_file}; re-run the notebook to regenerate it."
            )
        adapter = cls.__new__(cls)
        adapter.__dict__.update(payload["state"])
        return adapter

    def get_metadata(self) -> dict[str, Any]:
        """Return identity, protocol settings, selection evidence and fitted state."""
        try:
            import xgboost

            xgb_version = xgboost.__version__
        except ImportError:  # pragma: no cover
            xgb_version = None

        return {
            "name": self.name,
            "model_id": self.model_id,
            "fitted": self._fitted,
            "mode": self.mode,
            "candidate_id": self.candidate_id,
            "candidate_pool_ids": [str(c.get("id")) for c in self._candidate_pool],
            "max_candidates": self.max_candidates,
            "preprocessing": copy.deepcopy(self.preprocessing_spec),
            "parameter_precedence": "fixed_params(+ceiling) -> candidate params -> params overrides -> seed",
            "param_overrides": copy.deepcopy(self._overrides),
            "selected_candidate_id": self.selected_candidate_id_,
            "selected_n_estimators": self.selected_n_estimators_,
            "max_boosting_rounds": self.max_boosting_rounds,
            "final_scale_pos_weight": self.final_scale_pos_weight_,
            "params": copy.deepcopy(self.params),
            "random_seed": self.random_seed,
            "inner_val_fraction": self.inner_val_fraction,
            "early_stopping_rounds": self.early_stopping_rounds,
            "early_stopping_enabled": bool(self.early_stopping_rounds),
            "refit_outer_train": self.refit_outer_train,
            "fitted_on_all_outer_rows": self._fitted_on_all_outer_rows,
            "n_outer_train_rows": self.n_outer_train_rows_,
            "outer_train_positive_rate": self.outer_train_positive_rate_,
            "candidate_selection_evidence": copy.deepcopy(self.candidate_selection_evidence_),
            "candidate_selection_time_seconds": self.candidate_selection_time_seconds_,
            "refit_time_seconds": self.refit_time_seconds_,
            "total_fit_time_seconds": self.total_fit_time_seconds_,
            "categorical_features": list(self.categorical_features),
            "input_feature_names": list(self._input_feature_names),
            "feature_names": list(self._feature_names),
            "encoded_feature_names": list(self._fitted_columns),
            "encoded_to_original": dict(self._encoded_to_original),
            "num_fitted_features": len(self._fitted_columns),
            "xgboost_version": xgb_version,
        }
