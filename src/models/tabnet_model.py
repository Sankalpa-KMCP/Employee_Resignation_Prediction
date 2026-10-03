"""Leakage-safe TabNet adapter for employee resignation prediction.

PyTorch and pytorch-tabnet are imported only when training or loading a fitted
model. Configuration and preprocessing checks therefore remain available in the
project's lightweight core environment.

The Member 6 notebook uses this adapter for fold-specific preprocessing, selection and refitting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "models" / "tabnet.yaml"
)


def load_tabnet_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the versioned TabNet experiment configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_tabnet_config(config)
    return config


def validate_tabnet_config(config: Mapping[str, Any]) -> None:
    """Fail fast when the bounded TabNet search contract is inconsistent."""
    required_sections = {"model", "evaluation", "preprocessing", "training", "candidates"}
    missing_sections = required_sections - set(config)
    if missing_sections:
        raise ValueError(f"TabNet config is missing sections: {sorted(missing_sections)}")

    preprocessing = config["preprocessing"]
    categorical = list(preprocessing.get("categorical_features", []))
    numerical = list(preprocessing.get("numerical_features", []))
    if len(categorical) != 4 or len(numerical) != 13:
        raise ValueError("TabNet must use exactly 4 categorical and 13 numerical features.")
    if len(set(categorical + numerical)) != 17:
        raise ValueError("TabNet feature names must be unique across all 17 predictors.")

    training = config["training"]
    fixed_params = config.get("fixed_params", {})
    training_cat_emb_dim = int(training.get("cat_emb_dim", -1))
    fixed_cat_emb_dim = int(fixed_params.get("cat_emb_dim", -1))
    if training_cat_emb_dim != fixed_cat_emb_dim:
        raise ValueError("TabNet fixed_params and training cat_emb_dim values must agree.")
    if training_cat_emb_dim != 1:
        raise ValueError("TabNet cat_emb_dim must equal the canonical value 1.")
    fraction = float(training.get("inner_validation_fraction", 0.0))
    if not 0.0 < fraction < 0.5:
        raise ValueError("inner_validation_fraction must be greater than 0 and less than 0.5.")
    max_epochs = int(training.get("max_epochs", 0))
    patience = int(training.get("patience", -1))
    if not 1 <= max_epochs <= 150:
        raise ValueError("TabNet max_epochs must be between 1 and 150.")
    if not 0 <= patience <= 15:
        raise ValueError("TabNet patience must be between 0 and 15.")

    candidates = list(config["candidates"])
    if not 1 <= len(candidates) <= 10:
        raise ValueError("TabNet must declare between 1 and 10 candidate configurations.")
    candidate_ids = [candidate.get("id") for candidate in candidates]
    if None in candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Every TabNet candidate must have a unique id.")

    allowed_values = {
        "n_d": {8, 16, 32},
        "n_a": {8, 16, 32},
        "n_steps": {3, 5, 7},
        "gamma": {1.2, 1.5},
        "lambda_sparse": {0.00001, 0.0001, 0.001},
        "learning_rate": {0.01, 0.02},
        "batch_size": {1024, 2048},
        "virtual_batch_size": {128, 256},
        "class_balance": {False, True},
    }
    for candidate in candidates:
        for parameter, allowed in allowed_values.items():
            if candidate.get(parameter) not in allowed:
                raise ValueError(
                    f"Candidate {candidate.get('id')} has invalid {parameter}: "
                    f"{candidate.get(parameter)!r}."
                )
        if candidate["batch_size"] % candidate["virtual_batch_size"] != 0:
            raise ValueError(
                f"Candidate {candidate['id']} batch_size must be divisible by "
                "virtual_batch_size."
            )


@dataclass
class TabNetPreprocessor:
    """Fold-fitted integer encoding, median imputation, and standardization."""

    categorical_features: list[str]
    numerical_features: list[str]
    category_mappings: dict[str, dict[str, int]] = field(default_factory=dict)
    imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median"))
    scaler: StandardScaler = field(default_factory=StandardScaler)
    fitted: bool = False

    @property
    def feature_order(self) -> list[str]:
        """Return categorical columns first, followed by numerical columns, matching the model matrix."""
        return self.categorical_features + self.numerical_features

    @property
    def cat_idxs(self) -> list[int]:
        """Return embedding column positions in the categorical-first input matrix."""
        return list(range(len(self.categorical_features)))

    @property
    def cat_dims(self) -> list[int]:
        """Return fitted embedding sizes, including an extra slot for unknown or missing categories."""
        if not self.fitted:
            raise RuntimeError("TabNet preprocessor has not been fitted.")
        return [len(self.category_mappings[column]) + 1 for column in self.categorical_features]

    def fit(self, X: pd.DataFrame) -> TabNetPreprocessor:
        """Learn category codes, numerical medians and scaling from the supplied training dataframe.

        Return this preprocessor; callers control which training partition it receives.
        """
        self._validate_columns(X)
        self.category_mappings = {}
        for column in self.categorical_features:
            values = sorted(X[column].dropna().astype(str).unique().tolist())
            # Reserve zero for absent/unseen labels; training categories start at one.
            self.category_mappings[column] = {
                value: code for code, value in enumerate(values, start=1)
            }

        numerical = X[self.numerical_features].apply(pd.to_numeric, errors="raise")
        imputed = self.imputer.fit_transform(numerical)
        self.scaler.fit(imputed)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted mappings, imputation and scaling to X without learning new statistics.

        Return a finite float32 matrix in feature_order; unseen or missing categories use code zero.
        """
        if not self.fitted:
            raise RuntimeError("TabNet preprocessor has not been fitted.")
        self._validate_columns(X)

        categorical_arrays: list[np.ndarray] = []
        for column in self.categorical_features:
            encoded = X[column].astype("string").map(self.category_mappings[column]).fillna(0)
            categorical_arrays.append(encoded.to_numpy(dtype=np.float32).reshape(-1, 1))

        numerical = X[self.numerical_features].apply(pd.to_numeric, errors="raise")
        imputed = self.imputer.transform(numerical)
        scaled = self.scaler.transform(imputed).astype(np.float32)
        matrix = np.hstack(categorical_arrays + [scaled]).astype(np.float32, copy=False)
        if matrix.shape != (len(X), len(self.feature_order)):
            raise RuntimeError("Unexpected TabNet model-matrix shape after preprocessing.")
        if not np.isfinite(matrix).all():
            raise ValueError("TabNet preprocessing produced non-finite values.")
        return matrix

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit on the supplied training rows and return their transformed model matrix."""
        return self.fit(X).transform(X)

    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Reject input missing any configured numerical or categorical feature."""
        missing = set(self.feature_order) - set(X.columns)
        if missing:
            raise ValueError(f"TabNet input is missing approved features: {sorted(missing)}")


def _tabnet_dependencies() -> tuple[Any, Any, Any]:
    """Import heavyweight optional dependencies with an actionable error."""
    try:
        import torch
        from pytorch_tabnet.metrics import Metric
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TabNet training requires the pinned 'torch==2.13.0' and "
            "'pytorch-tabnet==4.1.0' packages. Install them only after approval."
        ) from exc
    return torch, Metric, TabNetClassifier


def _average_precision_metric(metric_base: Any) -> type:
    """Build pytorch-tabnet's custom average-precision metric class."""

    class AveragePrecisionMetric(metric_base):
        """Expose positive-class Average Precision to TabNet's early-stopping metric interface."""
        def __init__(self) -> None:
            """Register an Average Precision metric that TabNet maximizes during early stopping."""
            self._name = "average_precision"
            self._maximize = True

        def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
            """Return Average Precision from labels and the positive-class column of TabNet probabilities."""
            return float(average_precision_score(y_true, y_score[:, 1]))

    return AveragePrecisionMetric


class TabNetAdapter:
    """ModelAdapter-compatible TabNet classifier with nested early stopping."""

    name = "TabNet"
    model_id = "tabnet"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        candidate_id: str = "T01",
        max_epochs_override: int | None = None,
        patience_override: int | None = None,
        outer_fold: int | None = None,
    ) -> None:
        """Resolve fixed or automatic candidate selection and initialize unfitted adapter state."""
        self.config = dict(config) if config is not None else load_tabnet_config()
        validate_tabnet_config(self.config)
        self.requested_candidate_id = candidate_id
        self.selection_mode = "auto" if candidate_id in {"auto", "tune"} else "fixed"
        self.candidate_id = candidate_id
        self.candidate = (
            None if self.selection_mode == "auto" else self._find_candidate(candidate_id)
        )
        self.outer_fold = outer_fold
        training = self.config["training"]
        self.max_epochs = int(
            training["max_epochs"] if max_epochs_override is None else max_epochs_override
        )
        self.patience = int(
            training["patience"] if patience_override is None else patience_override
        )
        if not 1 <= self.max_epochs <= 150:
            raise ValueError("max_epochs must be between 1 and 150.")
        if not 0 <= self.patience <= 15:
            raise ValueError("patience must be between 0 and 15.")

        preprocessing = self.config["preprocessing"]
        self.categorical_features = list(preprocessing["categorical_features"])
        self.numerical_features = list(preprocessing["numerical_features"])
        self.feature_order = self.categorical_features + self.numerical_features
        self.preprocessor: TabNetPreprocessor | None = None
        self.model: Any = None
        self.selected_epoch: int | None = None
        self.best_inner_ap: float | None = None
        self.candidate_selection_records: list[dict[str, Any]] = []
        self.training_histories: dict[str, dict[str, list[float]]] = {}
        self.selection_history: dict[str, Any] = {}
        self.fit_summary: dict[str, Any] = {}
        self._fitted = False

    def _find_candidate(self, candidate_id: str) -> dict[str, Any]:
        """Return a copy of the requested configured candidate; reject unknown IDs."""
        matches = [item for item in self.config["candidates"] if item["id"] == candidate_id]
        if not matches:
            available = [item["id"] for item in self.config["candidates"]]
            raise ValueError(f"Unknown candidate_id {candidate_id!r}; choose from {available}.")
        return dict(matches[0])

    def _new_preprocessor(self) -> TabNetPreprocessor:
        """Create fresh, unfitted preprocessing state so separate fits cannot share learned statistics."""
        return TabNetPreprocessor(
            categorical_features=self.categorical_features.copy(),
            numerical_features=self.numerical_features.copy(),
        )

    def _build_model(
        self,
        preprocessor: TabNetPreprocessor,
        candidate: Mapping[str, Any] | None = None,
    ) -> Any:
        """Construct a TabNet classifier using the candidate and fitted embedding dimensions.

        No fitting occurs here; dependency loading is deferred until a model is needed.
        """
        torch, _, classifier = _tabnet_dependencies()
        training = self.config["training"]
        resolved_candidate = candidate if candidate is not None else self.candidate
        if resolved_candidate is None:
            raise RuntimeError("A concrete TabNet candidate is required to build the model.")
        return classifier(
            n_d=int(resolved_candidate["n_d"]),
            n_a=int(resolved_candidate["n_a"]),
            n_steps=int(resolved_candidate["n_steps"]),
            gamma=float(resolved_candidate["gamma"]),
            lambda_sparse=float(resolved_candidate["lambda_sparse"]),
            cat_idxs=preprocessor.cat_idxs,
            cat_dims=preprocessor.cat_dims,
            cat_emb_dim=int(training["cat_emb_dim"]),
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": float(resolved_candidate["learning_rate"])},
            mask_type=str(training["mask_type"]),
            seed=int(training["random_seed"]),
            device_name=str(training["device_name"]),
            verbose=1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> TabNetAdapter:
        """Select a candidate and epoch internally, then refit outer-training rows."""
        self._validate_training_input(X, y)
        y_array = np.asarray(y, dtype=np.int64).reshape(-1)
        training = self.config["training"]
        indices = np.arange(len(X))
        inner_train_idx, inner_valid_idx = train_test_split(
            indices,
            test_size=float(training["inner_validation_fraction"]),
            random_state=int(training["random_seed"]),
            stratify=y_array,
        )

        started = time.perf_counter()
        _, metric_base, _ = _tabnet_dependencies()
        metric = _average_precision_metric(metric_base)
        candidate_pool = (
            [dict(candidate) for candidate in self.config["candidates"]]
            if self.selection_mode == "auto"
            else [dict(self.candidate or {})]
        )
        self.candidate_selection_records = []
        self.training_histories = {}

        for candidate in candidate_pool:
            candidate_id = str(candidate["id"])
            candidate_started = time.perf_counter()
            record: dict[str, Any] = {
                "candidate_id": candidate_id,
                "candidate_parameters": dict(candidate),
                "seed": int(training["random_seed"]),
                "class_balance": bool(candidate["class_balance"]),
                "success": False,
            }
            try:
                selection_preprocessor = self._new_preprocessor()
                X_inner_train = selection_preprocessor.fit_transform(X.iloc[inner_train_idx])
                X_inner_valid = selection_preprocessor.transform(X.iloc[inner_valid_idx])
                selection_model = self._build_model(selection_preprocessor, candidate)
                selection_model.fit(
                    X_inner_train,
                    y_array[inner_train_idx],
                    eval_set=[(X_inner_valid, y_array[inner_valid_idx])],
                    eval_name=["inner_valid"],
                    eval_metric=[metric],
                    max_epochs=self.max_epochs,
                    patience=self.patience,
                    **self._fit_kwargs(candidate),
                )

                # TabNet reports zero-based epochs; a fresh refit needs an epoch count.
                best_epoch_zero_based = int(
                    getattr(selection_model, "best_epoch", self.max_epochs - 1)
                )
                selected_epoch = max(
                    1, min(best_epoch_zero_based + 1, self.max_epochs)
                )
                history = self._normalise_history(selection_model)
                inner_ap = self._extract_best_inner_ap(selection_model, history)
                self.training_histories[candidate_id] = history
                record.update(
                    {
                        "success": True,
                        "inner_average_precision": inner_ap,
                        "best_epoch": best_epoch_zero_based,
                        "selected_epoch": selected_epoch,
                    }
                )
            except Exception as exc:  # Preserve failed candidate evidence and continue.
                self.training_histories[candidate_id] = {}
                record.update(
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            record["runtime_seconds"] = float(time.perf_counter() - candidate_started)
            self.candidate_selection_records.append(record)

        successful_records = [
            record for record in self.candidate_selection_records if record["success"]
        ]
        if not successful_records:
            failures = "; ".join(
                f"{record['candidate_id']}: {record.get('error_message', 'unknown error')}"
                for record in self.candidate_selection_records
            )
            raise RuntimeError(f"All TabNet candidate fits failed. {failures}")

        # Rank successful candidates by inner AP; candidate ID breaks exact ties deterministically.
        winner = min(
            successful_records,
            key=lambda record: (
                -float(record["inner_average_precision"]),
                str(record["candidate_id"]),
            ),
        )
        self.candidate_id = str(winner["candidate_id"])
        self.candidate = self._find_candidate(self.candidate_id)
        self.selected_epoch = int(winner["selected_epoch"])
        self.best_inner_ap = float(winner["inner_average_precision"])
        self.selection_history = self.training_histories[self.candidate_id]

        # Refit learned preprocessing and model parameters on every outer-training row.
        refit_started = time.perf_counter()
        self.preprocessor = self._new_preprocessor()
        X_outer_train = self.preprocessor.fit_transform(X.reset_index(drop=True))
        self.model = self._build_model(self.preprocessor, self.candidate)
        self.model.fit(
            X_outer_train,
            y_array,
            max_epochs=self.selected_epoch,
            patience=0,
            **self._fit_kwargs(self.candidate),
        )
        refit_time = float(time.perf_counter() - refit_started)
        self._fitted = True
        self.fit_summary = {
            "outer_fold": self.outer_fold,
            "outer_training_rows_before": int(len(X)),
            "outer_training_rows_after": int(len(X_outer_train)),
            "inner_training_rows": int(len(inner_train_idx)),
            "inner_validation_rows": int(len(inner_valid_idx)),
            "positive_rate_outer_training": float(y_array.mean()),
            "positive_rate_inner_training": float(y_array[inner_train_idx].mean()),
            "positive_rate_inner_validation": float(y_array[inner_valid_idx].mean()),
            "selected_candidate_id": self.candidate_id,
            "selected_epoch": self.selected_epoch,
            "best_inner_average_precision": self.best_inner_ap,
            "candidate_count": int(len(candidate_pool)),
            "candidate_selection_time_seconds": float(refit_started - started),
            "refit_time_seconds": refit_time,
            "fit_time_seconds": float(time.perf_counter() - started),
        }
        return self

    @staticmethod
    def _normalise_history(model: Any) -> dict[str, list[float]]:
        """Convert supported TabNet history containers to lists of floats, or return an empty mapping."""
        history_object = getattr(model, "history", {})
        history_data = getattr(history_object, "history", history_object)
        try:
            return {
                str(key): [float(value) for value in values]
                for key, values in dict(history_data).items()
            }
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _extract_best_inner_ap(
        model: Any, history: Mapping[str, list[float]]
    ) -> float:
        """Recover the best finite inner AP from best_cost or training history; fail if neither is available."""
        best_cost = getattr(model, "best_cost", None)
        if best_cost is not None and np.isfinite(float(best_cost)):
            return float(best_cost)
        metric_values = [
            float(value)
            for key, values in history.items()
            if "average_precision" in key.lower()
            for value in values
            if np.isfinite(float(value))
        ]
        if not metric_values:
            raise RuntimeError("TabNet did not expose an inner-validation AP score.")
        return float(max(metric_values))

    def _fit_kwargs(
        self, candidate: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Resolve candidate-specific class balancing and batch options for selection and refitting."""
        training = self.config["training"]
        resolved_candidate = candidate if candidate is not None else self.candidate
        if resolved_candidate is None:
            raise RuntimeError("A concrete TabNet candidate is required for fitting.")
        return {
            "weights": 1 if resolved_candidate["class_balance"] else 0,
            "batch_size": int(resolved_candidate["batch_size"]),
            "virtual_batch_size": int(resolved_candidate["virtual_batch_size"]),
            "num_workers": int(training["num_workers"]),
            "drop_last": bool(training["drop_last"]),
            "compute_importance": True,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return one finite positive-class probability per row using the saved preprocessor.

        Require a fitted adapter and a two-column classifier output; clip scores to [0, 1].
        """
        if not self._fitted or self.preprocessor is None or self.model is None:
            raise RuntimeError("TabNetAdapter must be fitted before prediction.")
        matrix = self.preprocessor.transform(X)
        probabilities = np.asarray(self.model.predict_proba(matrix), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape != (len(X), 2):
            raise RuntimeError("TabNet returned an unexpected probability matrix shape.")
        positive_probability = probabilities[:, 1]
        if not np.isfinite(positive_probability).all():
            raise ValueError("TabNet returned non-finite positive-class probabilities.")
        return np.clip(positive_probability, 0.0, 1.0)

    def save(self, directory: Path) -> dict[str, Any]:
        """Save the fitted network and adapter state, including the preprocessor, to a directory.

        Return the paths needed to reload prediction state.
        """
        if not self._fitted or self.preprocessor is None or self.model is None:
            raise RuntimeError("TabNetAdapter must be fitted before it can be saved.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        saved_model_path = Path(self.model.save_model(str(directory / "tabnet_model")))
        state = {
            "config": self.config,
            "requested_candidate_id": self.requested_candidate_id,
            "selection_mode": self.selection_mode,
            "candidate_id": self.candidate_id,
            "candidate": self.candidate,
            "outer_fold": self.outer_fold,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "selected_epoch": self.selected_epoch,
            "best_inner_ap": self.best_inner_ap,
            "candidate_selection_records": self.candidate_selection_records,
            "training_histories": self.training_histories,
            "selection_history": self.selection_history,
            "fit_summary": self.fit_summary,
            "preprocessor": self.preprocessor,
            "model_filename": saved_model_path.name,
        }
        joblib.dump(state, directory / "adapter_state.joblib")
        metadata = self.get_metadata()
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return {
            "directory": str(directory),
            "model_path": str(saved_model_path),
            "state_path": str(directory / "adapter_state.joblib"),
            "metadata_path": str(directory / "metadata.json"),
        }

    @classmethod
    def load(cls, directory: Path) -> TabNetAdapter:
        """Restore a fitted adapter and its preprocessor from trusted local model artifacts."""
        directory = Path(directory)
        state = joblib.load(directory / "adapter_state.joblib")
        adapter = cls(
            config=state["config"],
            candidate_id=state.get("requested_candidate_id", state["candidate_id"]),
            max_epochs_override=state["max_epochs"],
            patience_override=state["patience"],
            outer_fold=state.get("outer_fold"),
        )
        _, _, classifier = _tabnet_dependencies()
        adapter.model = classifier()
        adapter.model.load_model(str(directory / state["model_filename"]))
        adapter.preprocessor = state["preprocessor"]
        adapter.selection_mode = state.get("selection_mode", adapter.selection_mode)
        adapter.candidate_id = state["candidate_id"]
        adapter.candidate = state.get("candidate", adapter._find_candidate(adapter.candidate_id))
        adapter.selected_epoch = state["selected_epoch"]
        adapter.best_inner_ap = state.get("best_inner_ap")
        adapter.candidate_selection_records = state.get(
            "candidate_selection_records", []
        )
        adapter.training_histories = state.get("training_histories", {})
        adapter.selection_history = state["selection_history"]
        adapter.fit_summary = state["fit_summary"]
        adapter._fitted = True
        return adapter

    def get_metadata(self) -> dict[str, Any]:
        """Return configuration, preprocessing dimensions and recorded selection/training evidence."""
        preprocessor_metadata: dict[str, Any] = {}
        if self.preprocessor is not None and self.preprocessor.fitted:
            preprocessor_metadata = {
                "feature_order": self.preprocessor.feature_order,
                "cat_idxs": self.preprocessor.cat_idxs,
                "cat_dims": self.preprocessor.cat_dims,
                "category_cardinalities": {
                    column: len(mapping)
                    for column, mapping in self.preprocessor.category_mappings.items()
                },
            }
        return {
            "name": self.name,
            "model_name": self.name,
            "model_id": self.model_id,
            "outer_fold": self.outer_fold,
            "selection_mode": self.selection_mode,
            "requested_candidate_id": self.requested_candidate_id,
            "candidate_id": self.candidate_id,
            "selected_candidate_id": self.candidate_id if self._fitted else None,
            "candidate": self.candidate,
            "selected_candidate_parameters": self.candidate,
            "best_inner_average_precision": self.best_inner_ap,
            "seed": int(self.config["training"]["random_seed"]),
            "class_balance": (
                bool(self.candidate["class_balance"])
                if self.candidate is not None
                else None
            ),
            "config_version": self.config["model"]["config_version"],
            "project_context_version": self.config["model"]["project_context_version"],
            "fitted": self._fitted,
            "selected_epoch": self.selected_epoch,
            "candidate_selection_records": self.candidate_selection_records,
            "training_histories": self.training_histories,
            "selection_history": self.selection_history,
            "fit_summary": self.fit_summary,
            "preprocessing": preprocessor_metadata,
        }

    def _validate_training_input(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> None:
        """Require approved features, matching row counts, at least ten rows and both binary target classes."""
        missing = set(self.feature_order) - set(X.columns)
        if missing:
            raise ValueError(f"TabNet input is missing approved features: {sorted(missing)}")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        if len(X) < 10:
            raise ValueError("TabNet training requires at least 10 rows.")
        y_array = np.asarray(y).reshape(-1)
        if pd.isna(y_array).any():
            raise ValueError("TabNet target contains missing values.")
        observed = set(np.unique(y_array).tolist())
        if observed != {0, 1} and observed != {False, True}:
            raise ValueError(f"TabNet target must contain both binary classes; found {observed}.")
