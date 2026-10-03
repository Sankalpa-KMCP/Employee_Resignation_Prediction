"""CatBoost Model Adapter for Employee Resignation Prediction (Member 2).

Conforms to the ModelAdapter protocol with fold-local preprocessing,
strict nested inner-validation early stopping, and native categorical handling.

The Member 2 notebook uses this adapter with native categorical inputs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.base import ModelAdapter


class CatBoostAdapter:
    """CatBoost model adapter with native categorical handling and inner-validation tuning.

    This adapter adheres to the strict nested outer-CV / inner-selection methodology:
    - Receives an already loaded or resolved configuration mapping from `src.models.config`.
    - Partitions outer-training rows internally for inner early-stopping and/or candidate selection.
    - Captures the selected optimal tree count from inner validation.
    - Refits a final estimator on ALL outer-training rows using the selected fixed tree count.
    - Preserves native categorical features without forced one-hot encoding or pre-scaling.
    - Disallows any access to or reliance upon outer-validation rows during fitting.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        candidate_id: str | None = None,
        params: dict[str, Any] | None = None,
        name: str = "CatBoost",
        model_id: str = "catboost",
        categorical_features: list[str] | None = None,
        inner_val_fraction: float | None = None,
        early_stopping_rounds: int | None = None,
        refit_outer_train: bool | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Resolve fixed or automatically selected candidates and training options without fitting.

        Keep nominal predictors available for CatBoost's native categorical handling.
        """
        self.name = name
        self.model_id = model_id
        self.candidate_id = candidate_id or "default"
        self._fitted = False
        self._fitted_on_all_outer_rows = False
        self._model: Any = None
        self._feature_names: list[str] = []
        self.selected_tree_count_: int | None = None
        self._inner_eval_results: dict[str, Any] = {}
        self._candidate_pool: list[dict[str, Any]] = []
        self.candidate_selection_time_seconds_: float = 0.0
        self.refit_time_seconds_: float = 0.0

        # Default categorical predictors from canonical feature contract
        self.categorical_features = (
            list(categorical_features)
            if categorical_features is not None
            else [
                "Department",
                "Gender",
                "Job_Title",
                "Education_Level",
            ]
        )

        # ---------------------------------------------------------------------
        # Case A: Initialized from Central Config Mapping
        # ---------------------------------------------------------------------
        if config is not None:
            # If config is already a resolved dictionary (produced by resolve_candidate)
            if "params" in config and isinstance(config["params"], Mapping):
                resolved = dict(config)
                self.candidate_id = str(resolved.get("candidate_id", self.candidate_id))
            # If raw validated config containing 'candidates' list
            elif "candidates" in config and isinstance(config["candidates"], Sequence):
                from src.models.config import resolve_candidate

                self._candidate_pool = [
                    copy.deepcopy(dict(c))
                    for c in config["candidates"]
                    if isinstance(c, Mapping)
                ]

                # If candidate_id is 'auto' or 'tune', candidate selection will run inside fit()
                if candidate_id in ("auto", "tune"):
                    self.candidate_id = "auto"
                    resolved = resolve_candidate(config, candidate_id=None)
                else:
                    resolved = resolve_candidate(config, candidate_id=candidate_id)
                    self.candidate_id = str(resolved.get("candidate_id", "default"))
            else:
                resolved = dict(config)

            # Extract model metadata
            model_meta = resolved.get("model", {})
            self.name = str(model_meta.get("name", self.name))
            self.model_id = str(model_meta.get("model_id", self.model_id))

            # Extract reproducibility
            repro = resolved.get("reproducibility", {})
            self.random_seed = int(
                random_seed
                if random_seed is not None
                else repro.get("random_seed", repro.get("seed", 42))
            )

            # Extract inner-validation policy
            inner_val = resolved.get("inner_validation", {})
            self.inner_val_fraction = float(
                inner_val_fraction
                if inner_val_fraction is not None
                else inner_val.get("fraction", 0.15)
            )
            self.early_stopping_rounds = (
                early_stopping_rounds
                if early_stopping_rounds is not None
                else inner_val.get("early_stopping_rounds", 50)
            )
            self.refit_outer_train = (
                refit_outer_train
                if refit_outer_train is not None
                else inner_val.get("refit_outer_train", True)
            )

            # Hyperparameters from resolved config
            self.params = copy.deepcopy(dict(resolved.get("params", {})))
            if params is not None:
                self.params.update(params)

        # ---------------------------------------------------------------------
        # Case B: Initialized via Direct Parameters (Unit Test / Direct Mode)
        # ---------------------------------------------------------------------
        elif params is not None:
            self.candidate_id = candidate_id or "direct"
            self.params = copy.deepcopy(params)
            self.random_seed = int(
                random_seed
                if random_seed is not None
                else self.params.get("random_seed", 42)
            )
            self.inner_val_fraction = (
                float(inner_val_fraction) if inner_val_fraction is not None else 0.0
            )
            self.early_stopping_rounds = early_stopping_rounds
            self.refit_outer_train = (
                refit_outer_train if refit_outer_train is not None else False
            )

        # ---------------------------------------------------------------------
        # Case C: Zero-Argument Fallback (Protocol Compliance & Inspection)
        # ---------------------------------------------------------------------
        else:
            self.params = {
                "iterations": 100,
                "learning_rate": 0.05,
                "depth": 6,
                "loss_function": "Logloss",
                "eval_metric": "PRAUC",
                "random_seed": 42,
                "verbose": False,
                "allow_writing_files": False,
            }
            self.random_seed = 42
            self.inner_val_fraction = 0.0
            self.early_stopping_rounds = None
            self.refit_outer_train = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> CatBoostAdapter:
        """Fit CatBoost model on training fold data with strict nested inner tuning.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor columns for outer-training rows.
        y : pd.Series or np.ndarray
            Binary target values for outer-training rows.

        Returns
        -------
        CatBoostAdapter
            Fitted adapter instance.
        """
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "catboost is required for CatBoostAdapter. Install with pip install catboost."
            ) from exc

        self._feature_names = list(X.columns)
        cat_indices = [
            i
            for i, col in enumerate(self._feature_names)
            if col in self.categorical_features
        ]

        # Ensure categoricals are treated as string objects
        X_proc = X.copy()
        for col in self.categorical_features:
            if col in X_proc.columns:
                X_proc[col] = X_proc[col].astype(str)

        y_arr = np.asarray(y, dtype=int)

        # Check if inner validation for early stopping or candidate tuning is active
        use_inner_val = (
            self.inner_val_fraction > 0.0
            and self.early_stopping_rounds is not None
            and self.early_stopping_rounds > 0
            and len(X_proc) >= 20
        )

        if use_inner_val:
            from sklearn.metrics import average_precision_score
            from sklearn.model_selection import train_test_split

            # Stratified inner split derived strictly from rows supplied to fit()
            X_in_tr, X_in_val, y_in_tr, y_in_val = train_test_split(
                X_proc,
                y_arr,
                test_size=self.inner_val_fraction,
                stratify=y_arr,
                random_state=self.random_seed,
            )

            sel_start = time.perf_counter()
            # -------------------------------------------------------------
            # Sub-case: Inner Candidate Selection across Candidate Pool
            # -------------------------------------------------------------
            if self.candidate_id in ("auto", "tune") and self._candidate_pool:
                best_ap = -1.0
                best_cid = None
                best_trees = None
                candidate_evals = {}

                for cand in self._candidate_pool:
                    cid = str(cand.get("id"))
                    cand_params = copy.deepcopy(cand.get("params", {}))
                    cand_full = copy.deepcopy(self.params)
                    cand_full.update(cand_params)
                    cand_full.setdefault("random_seed", self.random_seed)
                    cand_full.setdefault("thread_count", -1)
                    cand_full["early_stopping_rounds"] = self.early_stopping_rounds
                    cand_full["verbose"] = False
                    cand_full["allow_writing_files"] = False

                    cand_clf = CatBoostClassifier(**cand_full, cat_features=cat_indices)
                    cand_clf.fit(
                        X_in_tr,
                        y_in_tr,
                        eval_set=(X_in_val, y_in_val),
                        verbose=False,
                    )

                    c_iter = cand_clf.get_best_iteration()
                    c_trees = (
                        int(c_iter + 1)
                        if c_iter is not None
                        else int(cand_clf.tree_count_)
                    )
                    c_probs = cand_clf.predict_proba(X_in_val)[:, 1]
                    c_ap = float(average_precision_score(y_in_val, c_probs))

                    candidate_evals[cid] = {
                        "ap_score": c_ap,
                        "inner_ap": c_ap,
                        "selected_trees": c_trees,
                        "selected_tree_count": c_trees,
                    }

                    if c_ap > best_ap:
                        best_ap = c_ap
                        best_cid = cid
                        best_trees = c_trees

                self.candidate_id = best_cid
                self.selected_tree_count_ = best_trees
                # Update params with the winning candidate's overrides
                for cand in self._candidate_pool:
                    if str(cand.get("id")) == best_cid:
                        self.params.update(cand.get("params", {}))
                        break

                self.candidate_selection_time_seconds_ = float(time.perf_counter() - sel_start)
                self._inner_eval_results = {
                    "metric": "average_precision",
                    "best_candidate_id": best_cid,
                    "best_inner_val_ap": best_ap,
                    "selected_tree_count": best_trees,
                    "all_candidates": candidate_evals,
                    "inner_train_size": len(X_in_tr),
                    "inner_val_size": len(X_in_val),
                    "candidate_selection_time_seconds": self.candidate_selection_time_seconds_,
                }

            # -------------------------------------------------------------
            # Sub-case: Single Candidate Early-Stopping Tuning
            # -------------------------------------------------------------
            else:
                inner_params = copy.deepcopy(self.params)
                inner_params.setdefault("random_seed", self.random_seed)
                inner_params["early_stopping_rounds"] = self.early_stopping_rounds
                inner_params["verbose"] = False
                inner_params["allow_writing_files"] = False

                inner_model = CatBoostClassifier(
                    **inner_params, cat_features=cat_indices
                )
                inner_model.fit(
                    X_in_tr,
                    y_in_tr,
                    eval_set=(X_in_val, y_in_val),
                    verbose=False,
                )

                best_iter = inner_model.get_best_iteration()
                self.selected_tree_count_ = (
                    int(best_iter + 1)
                    if best_iter is not None
                    else int(inner_model.tree_count_)
                )

                val_probs = inner_model.predict_proba(X_in_val)[:, 1]
                ap_score = float(average_precision_score(y_in_val, val_probs))
                self.candidate_selection_time_seconds_ = float(time.perf_counter() - sel_start)
                self._inner_eval_results = {
                    "metric": "average_precision",
                    "inner_val_ap": ap_score,
                    "best_iteration": best_iter,
                    "selected_tree_count": self.selected_tree_count_,
                    "all_candidates": {
                        self.candidate_id: {
                            "ap_score": ap_score,
                            "inner_ap": ap_score,
                            "selected_trees": self.selected_tree_count_,
                            "selected_tree_count": self.selected_tree_count_,
                        }
                    },
                    "inner_train_size": len(X_in_tr),
                    "inner_val_size": len(X_in_val),
                    "candidate_selection_time_seconds": self.candidate_selection_time_seconds_,
                }

            # -------------------------------------------------------------
            # Final Model: Refit on 100% of outer-training rows if requested
            # -------------------------------------------------------------
            if self.refit_outer_train:
                refit_start = time.perf_counter()
                refit_params = copy.deepcopy(self.params)
                refit_params["iterations"] = self.selected_tree_count_
                refit_params.pop("early_stopping_rounds", None)
                refit_params.setdefault("random_seed", self.random_seed)
                refit_params.setdefault("thread_count", -1)
                refit_params["verbose"] = False
                refit_params["allow_writing_files"] = False

                final_model = CatBoostClassifier(
                    **refit_params, cat_features=cat_indices
                )
                final_model.fit(X_proc, y_arr, verbose=False)
                self.refit_time_seconds_ = float(time.perf_counter() - refit_start)
                self._inner_eval_results["refit_time_seconds"] = self.refit_time_seconds_
                self._model = final_model
                self._fitted_on_all_outer_rows = True
            else:
                self.refit_time_seconds_ = 0.0
                self._inner_eval_results["refit_time_seconds"] = 0.0
                self._model = inner_model
                self._fitted_on_all_outer_rows = False

        else:
            self.candidate_selection_time_seconds_ = 0.0
            direct_params = copy.deepcopy(self.params)
            direct_params.setdefault("random_seed", self.random_seed)
            direct_params.pop("early_stopping_rounds", None)
            direct_params["verbose"] = False
            direct_params["allow_writing_files"] = False

            refit_start = time.perf_counter()
            direct_model = CatBoostClassifier(**direct_params, cat_features=cat_indices)
            direct_model.fit(X_proc, y_arr, verbose=False)
            self.refit_time_seconds_ = float(time.perf_counter() - refit_start)
            self._model = direct_model
            self.selected_tree_count_ = int(direct_model.tree_count_)
            self._fitted_on_all_outer_rows = True
            self._inner_eval_results = {
                "candidate_selection_time_seconds": 0.0,
                "refit_time_seconds": self.refit_time_seconds_,
                "all_candidates": {},
            }

        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive class probabilities for input features."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Adapter is not fitted. Call fit() first.")

        X_proc = X.copy()
        for col in self.categorical_features:
            if col in X_proc.columns:
                X_proc[col] = X_proc[col].astype(str)

        probs = self._model.predict_proba(X_proc)[:, 1]
        return np.asarray(probs, dtype=float)

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance values computed by the underlying fitted CatBoost model.

        Returns
        -------
        np.ndarray
            Feature importance scores matching predictor order.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("CatBoostAdapter is not fitted. Call fit() first.")
        return np.asarray(self._model.get_feature_importance(), dtype=float)

    def save(self, directory: Path) -> dict[str, Any]:
        """Serialize model artifacts and metadata to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_file = directory / "model.cbm"
        if self._model is not None:
            self._model.save_model(str(model_file))

        metadata = self.get_metadata()
        metadata["model_file"] = str(model_file.name)

        meta_file = directory / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        return metadata

    @classmethod
    def load(cls, directory: Path) -> CatBoostAdapter:
        """Load serialized CatBoost adapter from disk."""
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("catboost is required to load CatBoostAdapter.") from exc

        directory = Path(directory)
        model_file = directory / "model.cbm"
        meta_file = directory / "metadata.json"

        meta: dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        adapter = cls(
            name=meta.get("name", "CatBoost"),
            model_id=meta.get("model_id", "catboost"),
            candidate_id=meta.get("candidate_id"),
            params=meta.get("params"),
            categorical_features=meta.get("categorical_features"),
        )

        if model_file.exists():
            model = CatBoostClassifier()
            model.load_model(str(model_file))
            adapter._model = model
            adapter._fitted = True
            adapter.selected_tree_count_ = meta.get("selected_tree_count")
            adapter._fitted_on_all_outer_rows = meta.get("fitted_on_all_outer_rows", True)

        return adapter

    def get_metadata(self) -> dict[str, Any]:
        """Return model metadata, hyperparameter settings, and feature names."""
        meta: dict[str, Any] = {
            "name": self.name,
            "model_id": self.model_id,
            "candidate_id": self.candidate_id,
            "fitted": self._fitted,
            "selected_tree_count": self.selected_tree_count_,
            "fitted_on_all_outer_rows": self._fitted_on_all_outer_rows,
            "inner_val_fraction": self.inner_val_fraction,
            "early_stopping_rounds": self.early_stopping_rounds,
            "refit_outer_train": self.refit_outer_train,
            "params": copy.deepcopy(self.params),
            "categorical_features": list(self.categorical_features),
            "feature_names": list(self._feature_names),
            "candidate_selection_time_seconds": float(self.candidate_selection_time_seconds_),
            "refit_time_seconds": float(self.refit_time_seconds_),
            "inner_evaluation": copy.deepcopy(self._inner_eval_results),
        }
        if self._fitted:
            meta["selection_evidence"] = self.get_selection_evidence()
        return meta

    def get_selection_evidence(self) -> dict[str, Any]:
        """Return a defensive deep copy of inner selection evidence and timings.

        Returns
        -------
        dict[str, Any]
            Dictionary holding candidate-selection evidence, optimal candidate parameters,
            inner validation scores, and granular timing measurements.

        Raises
        ------
        RuntimeError
            If called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "CatBoostAdapter is not fitted. Call fit() before retrieving selection evidence."
            )

        winning_inner_ap = (
            self._inner_eval_results.get("best_inner_val_ap")
            if "best_inner_val_ap" in self._inner_eval_results
            else self._inner_eval_results.get("inner_val_ap")
        )

        evidence: dict[str, Any] = {
            "winning_candidate_id": self.candidate_id,
            "winning_candidate": self.candidate_id,
            "candidate_id": self.candidate_id,
            "winning_inner_ap": (
                float(winning_inner_ap) if winning_inner_ap is not None else None
            ),
            "selected_tree_count": self.selected_tree_count_,
            "inner_train_rows": self._inner_eval_results.get("inner_train_size"),
            "inner_val_rows": self._inner_eval_results.get("inner_val_size"),
            "inner_validation_rows": self._inner_eval_results.get("inner_val_size"),
            "inner_val_fraction": self.inner_val_fraction,
            "inner_validation_fraction": self.inner_val_fraction,
            "random_seed": self.random_seed,
            "selected_class_weight": self.params.get("auto_class_weights"),
            "all_candidates": copy.deepcopy(
                self._inner_eval_results.get("all_candidates", {})
            ),
            "candidate_selection_time_seconds": float(self.candidate_selection_time_seconds_),
            "refit_time_seconds": float(self.refit_time_seconds_),
            "metric": self._inner_eval_results.get("metric", "average_precision"),
            "fitted_on_all_outer_rows": self._fitted_on_all_outer_rows,
        }
        return copy.deepcopy(evidence)

