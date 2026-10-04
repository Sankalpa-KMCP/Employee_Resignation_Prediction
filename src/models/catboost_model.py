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


def temporal_inner_split_indices(
    snapshot_dates: Sequence[object], inner_val_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Create a strictly time-ordered inner split from outer-training rows.

    Every returned training timestamp is earlier than every validation timestamp.
    Date ties remain in one partition, so this function never lets simultaneous
    snapshots leak across the early-stopping boundary.
    """
    if not 0.0 < inner_val_fraction < 1.0:
        raise ValueError("inner_val_fraction must be strictly between 0 and 1.")
    dates = pd.to_datetime(pd.Series(snapshot_dates), errors="raise", utc=True)
    if dates.isna().any():
        raise ValueError("snapshot_dates cannot contain missing timestamps.")
    if len(dates) < 2 or dates.nunique() < 2:
        raise ValueError("Temporal inner validation requires at least two distinct dates.")

    desired_validation_rows = max(1, int(np.ceil(len(dates) * inner_val_fraction)))
    unique_dates = np.sort(dates.unique())
    candidates: list[tuple[int, pd.Timestamp]] = []
    for boundary in unique_dates[1:]:
        validation_count = int((dates >= boundary).sum())
        candidates.append((abs(validation_count - desired_validation_rows), boundary))
    _, boundary = min(candidates, key=lambda item: (item[0], item[1]))
    train_indices = np.flatnonzero((dates < boundary).to_numpy())
    validation_indices = np.flatnonzero((dates >= boundary).to_numpy())
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Unable to form non-empty strictly ordered temporal inner split.")
    return train_indices, validation_indices


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
        selection_metric: str | None = None,
        inner_seeds: Sequence[int] | None = None,
        final_seeds: Sequence[int] | None = None,
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
        self._models: list[Any] = []

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

        # Optional protocol extensions (all off by default, so behaviour is unchanged):
        #   selection_metric: inner criterion that ranks candidates (average_precision | roc_auc | log_loss)
        #   inner_seeds: repeat the stratified inner split with each seed and average candidate scores
        #   final_seeds: refit the winner with each seed on all outer-train rows and average probabilities
        cfg_map = config if isinstance(config, Mapping) else {}
        inner_cfg = cfg_map.get("inner_validation", {}) or {}
        self.selection_metric = str(
            selection_metric or inner_cfg.get("selection_metric") or "average_precision"
        )
        if self.selection_metric not in ("average_precision", "roc_auc", "log_loss"):
            raise ValueError("selection_metric must be average_precision, roc_auc or log_loss.")
        seeds = inner_seeds if inner_seeds is not None else inner_cfg.get("repeat_seeds")
        self.inner_seeds = [int(x) for x in seeds] if seeds else None
        fseeds = final_seeds if final_seeds is not None else (cfg_map.get("ensemble", {}) or {}).get("final_seeds")
        self.final_seeds = [int(x) for x in fseeds] if fseeds else None

        # GPU capability may be inspected by diagnostics, but diagnostics must
        # not silently select it for fitting. CPU is the reproducible default.
        self.params.setdefault("task_type", "CPU")
        self.params.setdefault("thread_count", -1)
        if str(self.params["task_type"]).upper() != "CPU":
            raise ValueError(
                "CatBoostAdapter supports CPU training only; GPU availability is "
                "reported diagnostically and is not a training default."
            )
        try:
            thread_count = int(self.params["thread_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError("thread_count must be -1 or a positive integer.") from exc
        if thread_count == 0 or thread_count < -1:
            raise ValueError("thread_count must be -1 or a positive integer.")
        self.params["thread_count"] = thread_count

    def _selection_score(self, metrics: Mapping[str, float]) -> float:
        """Higher-is-better inner score for the configured selection metric."""
        if self.selection_metric == "log_loss":
            return -float(metrics["log_loss"])
        return float(metrics[self.selection_metric])

    def _build_inner_splits(self, X_proc, y_arr, snapshot_dates):
        """Inner train/validation partitions carved only from the supplied outer-training rows."""
        if snapshot_dates is None:
            from sklearn.model_selection import train_test_split

            seeds = self.inner_seeds or [self.random_seed]
            splits = []
            for seed in seeds:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_proc, y_arr, test_size=self.inner_val_fraction, stratify=y_arr, random_state=seed
                )
                splits.append((X_tr, X_val, y_tr, y_val))
            return splits, "stratified_random_historical"
        if len(snapshot_dates) != len(X_proc):
            raise ValueError("snapshot_dates must align one-to-one with X rows.")
        train_indices, validation_indices = temporal_inner_split_indices(
            snapshot_dates, self.inner_val_fraction
        )
        y_tr, y_val = y_arr[train_indices], y_arr[validation_indices]
        if len(np.unique(y_tr)) != 2 or len(np.unique(y_val)) != 2:
            raise ValueError(
                "Temporal inner training and validation partitions must each "
                "contain both target classes for CatBoost candidate selection."
            )
        split = (X_proc.iloc[train_indices], X_proc.iloc[validation_indices], y_tr, y_val)
        return [split], "strict_temporal_ordered"

    def _fit_inner(self, params, split, cat_indices):
        """Fit one early-stopped inner model; return (model, trees, metrics, evals_result)."""
        from catboost import CatBoostClassifier
        from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

        X_tr, X_val, y_tr, y_val = split
        inner_params = copy.deepcopy(params)
        inner_params.setdefault("random_seed", self.random_seed)
        inner_params["early_stopping_rounds"] = self.early_stopping_rounds
        inner_params["verbose"] = False
        inner_params["allow_writing_files"] = False
        model = CatBoostClassifier(**inner_params, cat_features=cat_indices)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        best_iter = model.get_best_iteration()
        trees = int(best_iter + 1) if best_iter is not None else int(model.tree_count_)
        probs = model.predict_proba(X_val)[:, 1]
        metrics = {
            "average_precision": float(average_precision_score(y_val, probs)),
            "roc_auc": float(roc_auc_score(y_val, probs)),
            "brier": float(brier_score_loss(y_val, probs)),
            "log_loss": float(log_loss(y_val, probs, labels=[0, 1])),
        }
        return model, trees, metrics, copy.deepcopy(model.get_evals_result())

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        snapshot_dates: Sequence[object] | None = None,
    ) -> CatBoostAdapter:
        """Fit CatBoost model on training fold data with strict nested inner tuning.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor columns for outer-training rows.
        y : pd.Series or np.ndarray
            Binary target values for outer-training rows.
        snapshot_dates : sequence, optional
            Snapshot dates aligned with ``X``. When supplied, CatBoost uses a
            strictly earlier/later inner split for tuning and early stopping.
            Historical static-data workflows omit this argument and retain their
            existing stratified random inner split.

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

        self._models = []
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
            splits, inner_split_strategy = self._build_inner_splits(X_proc, y_arr, snapshot_dates)
            X_in_tr, X_in_val = splits[0][0], splits[0][1]
            sel_start = time.perf_counter()
            # -------------------------------------------------------------
            # Sub-case: Inner Candidate Selection across Candidate Pool
            # -------------------------------------------------------------
            if self.candidate_id in ("auto", "tune") and self._candidate_pool:
                best_score = -np.inf
                best_cid = None
                best_trees = None
                best_ap = None
                candidate_evals: dict[str, dict[str, Any]] = {}

                for cand in self._candidate_pool:
                    cid = str(cand.get("id"))
                    cand_params = copy.deepcopy(cand.get("params", {}))
                    cand_full = copy.deepcopy(self.params)
                    cand_full.update(cand_params)
                    if str(cand_full.get("task_type", "CPU")).upper() != "CPU":
                        raise ValueError(
                            f"Candidate {cid} requests non-CPU training; GPU is not "
                            "permitted by the reproducible CatBoost protocol."
                        )
                    cand_full.setdefault("random_seed", self.random_seed)
                    cand_full.setdefault("thread_count", -1)

                    per_split = [
                        self._fit_inner(cand_full, split, cat_indices)[1:] for split in splits
                    ]
                    trees_list = [t for t, _, _ in per_split]
                    c_trees = int(round(float(np.median(trees_list))))
                    mean_metrics = {
                        k: float(np.mean([m[k] for _, m, _ in per_split]))
                        for k in ("average_precision", "roc_auc", "brier", "log_loss")
                    }
                    c_ap = mean_metrics["average_precision"]
                    c_score = self._selection_score(mean_metrics)
                    c_evals = per_split[0][2]

                    candidate_evals[cid] = {
                        "candidate_id": cid,
                        "ap_score": c_ap,
                        "inner_ap": c_ap,
                        "inner_roc_auc": mean_metrics["roc_auc"],
                        "inner_log_loss": mean_metrics["log_loss"],
                        "brier_score": mean_metrics["brier"],
                        "selection_metric": self.selection_metric,
                        "selection_score": c_score,
                        "n_inner_splits": len(splits),
                        "selected_trees": c_trees,
                        "selected_tree_count": c_trees,
                        "depth": cand_params.get("depth"),
                        "learning_rate": cand_params.get("learning_rate"),
                        "l2_leaf_reg": cand_params.get("l2_leaf_reg"),
                        "auto_class_weights": cand_params.get("auto_class_weights"),
                        "class_weights": cand_params.get("class_weights"),
                        "random_strength": cand_params.get("random_strength"),
                        "bagging_temperature": cand_params.get("bagging_temperature"),
                        "one_hot_max_size": cand_params.get("one_hot_max_size"),
                        "early_stopping_rounds": self.early_stopping_rounds,
                        "evals_result": c_evals,
                    }

                    is_better = False
                    if best_cid is None:
                        is_better = True
                    elif c_score > best_score + 1e-6:
                        is_better = True
                    elif abs(c_score - best_score) <= 1e-6:
                        # Documented tie-break: prefer lower depth, then higher l2_leaf_reg
                        curr_depth = cand_params.get("depth", 6)
                        best_depth = candidate_evals[best_cid].get("depth", 6)
                        if curr_depth < best_depth:
                            is_better = True
                        elif curr_depth == best_depth:
                            curr_l2 = cand_params.get("l2_leaf_reg", 3)
                            best_l2 = candidate_evals[best_cid].get("l2_leaf_reg", 3)
                            if curr_l2 > best_l2:
                                is_better = True

                    if is_better:
                        best_score = c_score
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
                    "metric": self.selection_metric,
                    "best_candidate_id": best_cid,
                    "best_inner_val_ap": best_ap,
                    "best_inner_selection_score": best_score,
                    "selected_tree_count": best_trees,
                    "tie_break_rule": "higher_inner_ap_then_lower_depth_then_higher_l2_then_candidate_id",
                    "all_candidates": candidate_evals,
                    "inner_train_size": len(X_in_tr),
                    "inner_val_size": len(X_in_val),
                    "n_inner_splits": len(splits),
                    "inner_split_strategy": inner_split_strategy,
                    "candidate_selection_time_seconds": self.candidate_selection_time_seconds_,
                }

            # -------------------------------------------------------------
            # Sub-case: Single Candidate Early-Stopping Tuning
            # -------------------------------------------------------------
            else:
                inner_model, trees, metrics, inner_evals = self._fit_inner(
                    self.params, splits[0], cat_indices
                )
                best_iter = inner_model.get_best_iteration()
                self.selected_tree_count_ = trees
                ap_score = metrics["average_precision"]
                roc_score = metrics["roc_auc"]
                brier_score = metrics["brier"]

                self.candidate_selection_time_seconds_ = float(time.perf_counter() - sel_start)
                self._inner_eval_results = {
                    "metric": "average_precision",
                    "inner_val_ap": ap_score,
                    "inner_val_roc_auc": roc_score,
                    "inner_val_brier": brier_score,
                    "inner_val_log_loss": metrics["log_loss"],
                    "best_iteration": best_iter,
                    "selected_tree_count": self.selected_tree_count_,
                    "tie_break_rule": "higher_inner_ap_then_lower_depth_then_higher_l2_then_candidate_id",
                    "evals_result": inner_evals,
                    "all_candidates": {
                        self.candidate_id: {
                            "candidate_id": self.candidate_id,
                            "ap_score": ap_score,
                            "inner_ap": ap_score,
                            "inner_roc_auc": roc_score,
                            "brier_score": brier_score,
                            "selected_trees": self.selected_tree_count_,
                            "selected_tree_count": self.selected_tree_count_,
                            "depth": self.params.get("depth"),
                            "learning_rate": self.params.get("learning_rate"),
                            "l2_leaf_reg": self.params.get("l2_leaf_reg"),
                            "auto_class_weights": self.params.get("auto_class_weights"),
                            "class_weights": self.params.get("class_weights"),
                            "random_strength": self.params.get("random_strength"),
                            "bagging_temperature": self.params.get("bagging_temperature"),
                            "one_hot_max_size": self.params.get("one_hot_max_size"),
                            "early_stopping_rounds": self.early_stopping_rounds,
                            "evals_result": inner_evals,
                        }
                    },
                    "inner_train_size": len(X_in_tr),
                    "inner_val_size": len(X_in_val),
                    "inner_split_strategy": inner_split_strategy,
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

                final_models = []
                for seed in self.final_seeds or [refit_params["random_seed"]]:
                    seed_params = {**refit_params, "random_seed": seed}
                    member = CatBoostClassifier(**seed_params, cat_features=cat_indices)
                    member.fit(X_proc, y_arr, verbose=False)
                    final_models.append(member)
                final_model = final_models[0]
                self._models = final_models
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

        members = self._models or [self._model]
        probs = np.mean([m.predict_proba(X_proc)[:, 1] for m in members], axis=0)
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
        members = self._models or [self._model]
        return np.mean([m.get_feature_importance() for m in members], axis=0).astype(float)

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

        model_filename = meta.get("model_file")
        if model_filename and (directory / model_filename).exists():
            model_file = directory / model_filename
        elif (directory / "catboost_development_model.cbm").exists() and not model_file.exists():
            model_file = directory / "catboost_development_model.cbm"

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
            "selection_metric": self.selection_metric,
            "inner_seeds": self.inner_seeds,
            "final_seeds": self.final_seeds,
            "refit_outer_train": self.refit_outer_train,
            "configured_max_iterations": int(self.params.get("iterations", 600)),
            "selected_best_iteration": int((self.selected_tree_count_ - 1) if self.selected_tree_count_ is not None else 0),
            "final_fitted_tree_count": int(self.selected_tree_count_) if self.selected_tree_count_ is not None else None,
            "tie_break_rule": "higher_inner_ap_then_lower_depth_then_higher_l2_then_candidate_id",
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
            "configured_max_iterations": int(self.params.get("iterations", 600)),
            "selected_best_iteration": int((self.selected_tree_count_ - 1) if self.selected_tree_count_ is not None else 0),
            "final_fitted_tree_count": int(self.selected_tree_count_) if self.selected_tree_count_ is not None else None,
            "tie_break_rule": "higher_inner_ap_then_lower_depth_then_higher_l2_then_candidate_id",
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
