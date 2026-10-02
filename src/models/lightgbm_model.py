"""LightGBM adapter with strict-nested, fold-local candidate selection."""

from __future__ import annotations

import copy
import pickle
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src.models.config import validate_model_config


DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "random_state": 42,
    "verbosity": -1,
    "force_col_wise": True,
    "n_jobs": -1,
}


def _lightgbm_dependencies() -> tuple[Any, Any]:
    """Import LightGBM lazily so adapter metadata remains dependency-light."""
    try:
        from lightgbm import LGBMClassifier, early_stopping
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for LightGBMAdapter. Install with pip install lightgbm."
        ) from exc
    return LGBMClassifier, early_stopping


class LightGBMAdapter:
    """LightGBM model adapter with optional formal inner candidate selection.

    When a validated formal configuration is supplied, ``fit`` receives only an
    outer-training partition from the shared runner. It creates one stratified
    inner split, evaluates every configured candidate on that shared split,
    resolves exact Average Precision ties by candidate ID, and refits the winner
    on every supplied outer-training row. The outer-validation fold is never
    passed to this method.

    The legacy ``params`` construction path remains available for historical
    fixed-configuration evidence and basic protocol compatibility.
    """

    def __init__(
        self,
        name: str = "LightGBM",
        model_id: str = "lightgbm",
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
        config: Mapping[str, Any] | None = None,
        candidate_id: str | None = None,
        outer_fold: int | None = None,
    ) -> None:
        if config is not None and params is not None:
            raise ValueError("Supply either formal config or direct params, not both.")

        self.name = name
        self.model_id = model_id
        self.categorical_features = categorical_features or [
            "Department",
            "Gender",
            "Job_Title",
            "Education_Level",
        ]
        self.outer_fold = outer_fold
        self.config = copy.deepcopy(dict(config)) if config is not None else None
        self.requested_candidate_id = candidate_id
        self.selection_mode = "direct"
        self._candidate_pool: list[dict[str, Any]] = []

        if self.config is not None:
            validate_model_config(self.config, strict_schema=True, model_id="lightgbm")
            self._candidate_pool = [
                copy.deepcopy(candidate) for candidate in self.config["candidates"]
            ]
            maximum_candidates = int(
                self.config.get("candidate_registry", {}).get(
                    "maximum_candidates", len(self._candidate_pool)
                )
            )
            if len(self._candidate_pool) > maximum_candidates:
                raise ValueError(
                    "Formal LightGBM candidate count exceeds configured maximum: "
                    f"{len(self._candidate_pool)} > {maximum_candidates}."
                )

            inner = self.config["inner_validation"]
            reproducibility = self.config["reproducibility"]
            self.inner_validation_fraction = float(inner["fraction"])
            self.inner_validation_stratified = bool(inner.get("stratified", True))
            self.random_seed = int(
                inner.get(
                    "random_seed",
                    reproducibility.get("random_seed", reproducibility.get("seed", 42)),
                )
            )
            self.selection_metric = str(inner.get("metric", "average_precision"))
            self.early_stopping_rounds = int(inner.get("early_stopping_rounds", 50))
            self.tie_breaker = str(inner.get("tie_breaker", "candidate_id_ascending"))
            self.refit_outer_train = bool(inner.get("refit_outer_train", True))

            if self.selection_metric != "average_precision":
                raise ValueError("Formal LightGBM selection metric must be average_precision.")
            if self.tie_breaker != "candidate_id_ascending":
                raise ValueError("Formal LightGBM tie breaker must be candidate_id_ascending.")
            if not self.refit_outer_train:
                raise ValueError("Formal LightGBM evaluation requires full outer-training refit.")
            if self.early_stopping_rounds <= 0:
                raise ValueError("Formal LightGBM early stopping rounds must be positive.")

            self.selection_mode = "auto" if candidate_id in (None, "auto", "tune") else "single"
            if self.selection_mode == "single":
                candidate = self._find_candidate(str(candidate_id))
                self._candidate_pool = [candidate]
            self.params = copy.deepcopy(self.config.get("fixed_params", {}))
        else:
            if candidate_id is not None:
                raise ValueError("candidate_id requires a formal LightGBM config.")
            self.params = copy.deepcopy(params) if params is not None else copy.deepcopy(DEFAULT_PARAMS)
            self.inner_validation_fraction = 0.0
            self.inner_validation_stratified = False
            self.random_seed = int(self.params.get("random_state", 42))
            self.selection_metric = "average_precision"
            self.early_stopping_rounds = 0
            self.tie_breaker = "candidate_id_ascending"
            self.refit_outer_train = False

        self._model: Any = None
        self._category_maps: dict[str, dict[Any, int]] = {}
        self._feature_names: list[str] = []
        self._fitted = False
        self.selected_candidate_id: str | None = None
        self.selected_candidate_parameters: dict[str, Any] = {}
        self.selected_inner_average_precision: float | None = None
        self.selected_best_iteration: int | None = None
        self.candidate_selection_records: list[dict[str, Any]] = []
        self.fit_summary: dict[str, Any] = {}
        self.refit_parameters: dict[str, Any] = {}
        self._inner_train_positions = np.asarray([], dtype=int)
        self._inner_validation_positions = np.asarray([], dtype=int)

    def _find_candidate(self, candidate_id: str) -> dict[str, Any]:
        matches = [candidate for candidate in self._candidate_pool if candidate["id"] == candidate_id]
        if not matches:
            available = [candidate["id"] for candidate in self._candidate_pool]
            raise ValueError(
                f"Unknown LightGBM candidate_id {candidate_id!r}; choose from {available}."
            )
        return copy.deepcopy(matches[0])

    def _fit_category_maps(self, X: pd.DataFrame) -> dict[str, dict[Any, int]]:
        """Learn categorical mappings exclusively from the supplied rows."""
        mappings: dict[str, dict[Any, int]] = {}
        for column in self.categorical_features:
            if column in X.columns:
                values = sorted(X[column].dropna().unique().tolist())
                mappings[column] = {value: index for index, value in enumerate(values)}
        return mappings

    def _transform_with_maps(
        self,
        X: pd.DataFrame,
        mappings: Mapping[str, Mapping[Any, int]],
    ) -> pd.DataFrame:
        """Apply fixed mappings; unseen or missing categories become ``-1``."""
        transformed = X.copy()
        for column in self.categorical_features:
            if column in transformed.columns:
                mapping = mappings.get(column, {})
                transformed[column] = (
                    transformed[column].map(mapping).fillna(-1).astype("int32").astype("category")
                )
        return transformed

    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Backward-compatible fold-local categorical preprocessing."""
        if fit:
            self._category_maps = self._fit_category_maps(X)
        return self._transform_with_maps(X, self._category_maps)

    def _create_inner_split(
        self,
        y: pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create and validate one deterministic split within outer training."""
        y_array = np.asarray(y, dtype=int).reshape(-1)
        positions = np.arange(len(y_array), dtype=int)
        stratify = y_array if self.inner_validation_stratified else None
        inner_train, inner_validation = train_test_split(
            positions,
            test_size=self.inner_validation_fraction,
            random_state=self.random_seed,
            stratify=stratify,
        )
        inner_train = np.sort(np.asarray(inner_train, dtype=int))
        inner_validation = np.sort(np.asarray(inner_validation, dtype=int))

        train_set = set(inner_train.tolist())
        validation_set = set(inner_validation.tolist())
        if not train_set.isdisjoint(validation_set):
            raise RuntimeError("Inner training and validation positions overlap.")
        if train_set.union(validation_set) != set(positions.tolist()):
            raise RuntimeError("Inner split does not exactly cover supplied outer-training rows.")
        return inner_train, inner_validation

    @staticmethod
    def _weight_evidence(
        y: np.ndarray,
        class_weight: str | None,
    ) -> dict[str, Any]:
        """Record label-local weighting evidence without consulting other rows."""
        labels, counts = np.unique(np.asarray(y, dtype=int), return_counts=True)
        class_counts = {str(int(label)): int(count) for label, count in zip(labels, counts)}
        weights: dict[str, float] | None = None
        if class_weight == "balanced":
            total = float(len(y))
            class_total = float(len(labels))
            weights = {
                str(int(label)): total / (class_total * float(count))
                for label, count in zip(labels, counts)
            }
        return {
            "class_weight": class_weight,
            "row_count": int(len(y)),
            "class_counts": class_counts,
            "derived_balanced_weights": weights,
        }

    @staticmethod
    def _resolve_best_iteration(model: Any, configured_ceiling: int) -> int:
        """Use LightGBM's selected iteration or safely fall back to the ceiling."""
        best_iteration = getattr(model, "best_iteration_", None)
        if best_iteration is None:
            best_iteration = getattr(model, "best_iteration", None)
        try:
            resolved = int(best_iteration)
        except (TypeError, ValueError):
            resolved = 0
        if resolved <= 0:
            resolved = int(configured_ceiling)
        return min(resolved, int(configured_ceiling))

    def _fit_formal(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> LightGBMAdapter:
        classifier, early_stopping = _lightgbm_dependencies()
        y_array = np.asarray(y, dtype=int).reshape(-1)
        inner_train, inner_validation = self._create_inner_split(y_array)
        self._inner_train_positions = inner_train.copy()
        self._inner_validation_positions = inner_validation.copy()
        self.candidate_selection_records = []
        started = time.perf_counter()

        fixed_params = copy.deepcopy(self.config.get("fixed_params", {})) if self.config else {}
        for candidate in self._candidate_pool:
            candidate_id = str(candidate["id"])
            candidate_params = copy.deepcopy(candidate["params"])
            resolved_params = copy.deepcopy(fixed_params)
            resolved_params.update(candidate_params)
            configured_ceiling = int(resolved_params["n_estimators"])
            candidate_started = time.perf_counter()
            record: dict[str, Any] = {
                "candidate_id": candidate_id,
                "candidate_parameters": copy.deepcopy(candidate_params),
                "resolved_parameters": copy.deepcopy(resolved_params),
                "class_weight": candidate_params.get("class_weight"),
                "configured_n_estimators": configured_ceiling,
                "inner_train_row_count": int(len(inner_train)),
                "inner_validation_row_count": int(len(inner_validation)),
                "success": False,
            }
            try:
                candidate_maps = self._fit_category_maps(X.iloc[inner_train])
                X_inner_train = self._transform_with_maps(X.iloc[inner_train], candidate_maps)
                X_inner_validation = self._transform_with_maps(
                    X.iloc[inner_validation], candidate_maps
                )
                y_inner_train = y_array[inner_train]
                y_inner_validation = y_array[inner_validation]
                model = classifier(**resolved_params)
                callback = early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                )
                model.fit(
                    X_inner_train,
                    y_inner_train,
                    eval_set=[(X_inner_validation, y_inner_validation)],
                    eval_metric=self.selection_metric,
                    callbacks=[callback],
                )
                probability = np.asarray(
                    model.predict_proba(X_inner_validation)[:, 1], dtype=float
                )
                inner_ap = float(
                    average_precision_score(y_inner_validation, probability)
                )
                best_iteration = self._resolve_best_iteration(model, configured_ceiling)
                record.update(
                    {
                        "success": True,
                        "inner_average_precision": inner_ap,
                        "selected_best_iteration": best_iteration,
                        "category_cardinalities": {
                            column: len(mapping)
                            for column, mapping in candidate_maps.items()
                        },
                        "inner_training_weight_evidence": self._weight_evidence(
                            y_inner_train, candidate_params.get("class_weight")
                        ),
                    }
                )
            except Exception as exc:  # Preserve failure evidence and try remaining candidates.
                record.update(
                    {
                        "failure_type": type(exc).__name__,
                        "failure_reason": str(exc),
                    }
                )
            record["runtime_seconds"] = float(time.perf_counter() - candidate_started)
            self.candidate_selection_records.append(record)

        successful = [record for record in self.candidate_selection_records if record["success"]]
        if not successful:
            failures = "; ".join(
                f"{record['candidate_id']}: {record.get('failure_reason', 'unknown failure')}"
                for record in self.candidate_selection_records
            )
            raise RuntimeError(f"All formal LightGBM candidates failed. {failures}")

        winner = min(
            successful,
            key=lambda record: (
                -float(record["inner_average_precision"]),
                str(record["candidate_id"]),
            ),
        )
        self.selected_candidate_id = str(winner["candidate_id"])
        selected_candidate = self._find_candidate(self.selected_candidate_id)
        self.selected_candidate_parameters = copy.deepcopy(selected_candidate["params"])
        self.selected_inner_average_precision = float(winner["inner_average_precision"])
        self.selected_best_iteration = int(winner["selected_best_iteration"])

        refit_started = time.perf_counter()
        self._category_maps = self._fit_category_maps(X)
        X_outer_training = self._transform_with_maps(X, self._category_maps)
        refit_params = copy.deepcopy(fixed_params)
        refit_params.update(self.selected_candidate_parameters)
        refit_params["n_estimators"] = self.selected_best_iteration
        self.refit_parameters = copy.deepcopy(refit_params)
        self.params = copy.deepcopy(refit_params)
        self._model = classifier(**refit_params)
        self._model.fit(X_outer_training, y_array)
        refit_seconds = float(time.perf_counter() - refit_started)

        self._fitted = True
        self.fit_summary = {
            "outer_fold": self.outer_fold,
            "outer_training_row_count": int(len(X)),
            "inner_training_row_count": int(len(inner_train)),
            "inner_validation_row_count": int(len(inner_validation)),
            "inner_validation_fraction": self.inner_validation_fraction,
            "inner_validation_stratified": self.inner_validation_stratified,
            "all_candidates_shared_inner_split": True,
            "candidate_count": int(len(self._candidate_pool)),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_inner_average_precision": self.selected_inner_average_precision,
            "selected_best_iteration": self.selected_best_iteration,
            "selected_class_weight": self.selected_candidate_parameters.get("class_weight"),
            "selection_time_seconds": float(refit_started - started),
            "refit_time_seconds": refit_seconds,
            "fit_time_seconds": float(time.perf_counter() - started),
            "refit_row_count": int(len(X_outer_training)),
            "refit_weight_evidence": self._weight_evidence(
                y_array, self.selected_candidate_parameters.get("class_weight")
            ),
        }
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> LightGBMAdapter:
        """Fit either a direct historical model or a formal nested-selection model."""
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        if len(X) == 0:
            raise ValueError("LightGBM training data cannot be empty.")
        self._feature_names = list(X.columns)

        if self.config is not None:
            return self._fit_formal(X, y)

        classifier, _ = _lightgbm_dependencies()
        X_processed = self._preprocess(X, fit=True)
        y_array = np.asarray(y, dtype=int)
        self._model = classifier(**self.params)
        self._model.fit(X_processed, y_array)
        self._fitted = True
        self.fit_summary = {
            "outer_training_row_count": int(len(X)),
            "selection_mode": "direct",
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive-class probabilities using refit category mappings."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Adapter is not fitted. Call fit() first.")
        X_processed = self._preprocess(X, fit=False)
        probabilities = np.asarray(self._model.predict_proba(X_processed)[:, 1], dtype=float)
        if probabilities.ndim != 1 or len(probabilities) != len(X):
            raise RuntimeError("LightGBM returned an unexpected probability shape.")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("LightGBM returned non-finite probabilities.")
        if (probabilities < 0.0).any() or (probabilities > 1.0).any():
            raise RuntimeError("LightGBM returned probabilities outside [0, 1].")
        return probabilities

    def save(self, directory: Path) -> dict[str, Any]:
        """Serialize the fitted model, preprocessing state, and selection evidence."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        artifact_file = directory / "lightgbm_adapter.pkl"
        with artifact_file.open("wb") as handle:
            pickle.dump(
                {
                    "name": self.name,
                    "model_id": self.model_id,
                    "model": self._model,
                    "category_maps": self._category_maps,
                    "params": self.params,
                    "categorical_features": self.categorical_features,
                    "feature_names": self._feature_names,
                    "config": self.config,
                    "requested_candidate_id": self.requested_candidate_id,
                    "selection_mode": self.selection_mode,
                    "outer_fold": self.outer_fold,
                    "selected_candidate_id": self.selected_candidate_id,
                    "selected_candidate_parameters": self.selected_candidate_parameters,
                    "selected_inner_average_precision": self.selected_inner_average_precision,
                    "selected_best_iteration": self.selected_best_iteration,
                    "candidate_selection_records": self.candidate_selection_records,
                    "fit_summary": self.fit_summary,
                    "refit_parameters": self.refit_parameters,
                },
                handle,
            )
        metadata = self.get_metadata()
        metadata["artifact_file"] = artifact_file.name
        return metadata

    @classmethod
    def load(cls, directory: Path) -> LightGBMAdapter:
        """Load a serialized adapter and its fold-local preprocessing state."""
        artifact_file = Path(directory) / "lightgbm_adapter.pkl"
        adapter = cls()
        if not artifact_file.exists():
            return adapter
        with artifact_file.open("rb") as handle:
            state = pickle.load(handle)
        adapter.name = state.get("name", adapter.name)
        adapter.model_id = state.get("model_id", adapter.model_id)
        adapter._model = state["model"]
        adapter._category_maps = state["category_maps"]
        adapter.params = state["params"]
        adapter.categorical_features = state["categorical_features"]
        adapter._feature_names = state["feature_names"]
        adapter.config = state.get("config")
        adapter.requested_candidate_id = state.get("requested_candidate_id")
        adapter.selection_mode = state.get("selection_mode", "direct")
        adapter.outer_fold = state.get("outer_fold")
        adapter.selected_candidate_id = state.get("selected_candidate_id")
        adapter.selected_candidate_parameters = state.get(
            "selected_candidate_parameters", {}
        )
        adapter.selected_inner_average_precision = state.get(
            "selected_inner_average_precision"
        )
        adapter.selected_best_iteration = state.get("selected_best_iteration")
        adapter.candidate_selection_records = state.get(
            "candidate_selection_records", []
        )
        adapter.fit_summary = state.get("fit_summary", {})
        adapter.refit_parameters = state.get("refit_parameters", {})
        adapter._fitted = True
        return adapter

    def get_metadata(self) -> dict[str, Any]:
        """Return auditable selection, refit, and categorical-mapping metadata."""
        model_metadata = self.config.get("model", {}) if self.config else {}
        return {
            "name": self.name,
            "model_id": self.model_id,
            "fitted": self._fitted,
            "selection_mode": self.selection_mode,
            "outer_fold": self.outer_fold,
            "params": copy.deepcopy(self.params),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_parameters": copy.deepcopy(
                self.selected_candidate_parameters
            ),
            "selected_inner_average_precision": self.selected_inner_average_precision,
            "selected_best_iteration": self.selected_best_iteration,
            "selected_class_weight": self.selected_candidate_parameters.get(
                "class_weight"
            ),
            "candidate_selection_records": copy.deepcopy(
                self.candidate_selection_records
            ),
            "fit_summary": copy.deepcopy(self.fit_summary),
            "refit_parameters": copy.deepcopy(self.refit_parameters),
            "categorical_features": self.categorical_features.copy(),
            "feature_names": self._feature_names.copy(),
            "categorical_mapping_metadata": {
                "unknown_category_code": -1,
                "category_cardinalities": {
                    column: len(mapping)
                    for column, mapping in self._category_maps.items()
                },
                "category_mappings": copy.deepcopy(self._category_maps),
            },
            "seed": self.random_seed,
            "config_version": model_metadata.get("config_version"),
            "candidate_registry_version": model_metadata.get(
                "candidate_registry_version"
            ),
        }
