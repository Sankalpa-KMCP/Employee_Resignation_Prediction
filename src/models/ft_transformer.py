"""FT-Transformer Model Adapter for Employee Resignation Prediction (Member 1).

Implements a leakage-safe FT-Transformer pipeline conforming to ModelAdapter protocol with:
- fold-local numerical and categorical preprocessing
- strict nested candidate selection across FT01–FT08 in automatic selection mode
- fold-local positive-class weighting (dynamic_ratio vs unweighted)
- stratified inner validation early stopping on Average Precision
- optional full outer-training refit for the selected epoch count
- comprehensive candidate selection evidence logging

The outer validation fold and formal holdout are never used for model fitting,
class-weight calculation, preprocessing fitting, candidate selection, or early stopping.

The Member 1 notebook uses this implementation; executing its saved-evidence path does not train a new model.
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.base import ModelAdapter


# =============================================================================
# Preprocessing Helpers (Strictly Scoped & Leakage-Safe)
# =============================================================================

def _fit_preprocessors(
    X: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[StandardScaler | None, dict[str, dict[Any, int]], list[int], list[str], list[str]]:
    """Fit StandardScaler and categorical mapping tables on training partition only.

    Parameters
    ----------
    X : pd.DataFrame
        Training data partition.
    categorical_features : list[str]
        List of categorical column names.

    Returns
    -------
    scaler : StandardScaler or None
    category_maps : dict mapping column name to value-to-index mapping
    cat_cardinalities : list of category vocabulary sizes (including 0 for unknown)
    feature_names : list of all feature names in order
    num_features : list of continuous numerical feature names
    """
    feature_names = list(X.columns)
    num_features = [col for col in feature_names if col not in categorical_features]

    scaler: StandardScaler | None = None
    if num_features:
        scaler = StandardScaler()
        scaler.fit(X[num_features])

    category_maps: dict[str, dict[Any, int]] = {}
    cat_cardinalities: list[int] = []

    for col in categorical_features:
        if col not in X.columns:
            continue
        unique_vals = sorted(X[col].dropna().unique())
        # Reserve index 0 for unknown/missing categories. Known categories start from 1.
        mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
        category_maps[col] = mapping
        cat_cardinalities.append(len(mapping) + 1)

    return scaler, category_maps, cat_cardinalities, feature_names, num_features


def _transform_features(
    X: pd.DataFrame,
    categorical_features: list[str],
    num_features: list[str],
    scaler: StandardScaler | None,
    category_maps: dict[str, dict[Any, int]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Transform features into numerical float32 matrix and categorical int64 matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Data to transform.
    categorical_features : list[str]
        List of categorical feature names.
    num_features : list[str]
        List of numerical feature names.
    scaler : StandardScaler or None
        Fitted scaler.
    category_maps : dict
        Fitted categorical mapping tables.

    Returns
    -------
    x_num : np.ndarray of float32 or None
    x_cat : np.ndarray of int64 or None
    """
    x_num: np.ndarray | None = None
    if num_features and scaler is not None:
        x_num = scaler.transform(X[num_features]).astype(np.float32)

    x_cat: np.ndarray | None = None
    if categorical_features:
        cat_arrays = []
        for col in categorical_features:
            if col not in X.columns:
                continue
            mapping = category_maps.get(col, {})
            arr = X[col].map(mapping).fillna(0).values.astype(np.int64)
            cat_arrays.append(arr)
        if cat_arrays:
            x_cat = np.column_stack(cat_arrays)

    return x_num, x_cat


def _make_tensor_dataset(
    x_num: np.ndarray | None,
    x_cat: np.ndarray | None,
    targets: np.ndarray | None = None,
):
    """Wrap numpy matrices into a PyTorch TensorDataset."""
    import torch
    from torch.utils.data import TensorDataset

    tensors = []
    if x_num is not None:
        tensors.append(torch.tensor(x_num, dtype=torch.float32))
    if x_cat is not None:
        tensors.append(torch.tensor(x_cat, dtype=torch.long))
    if targets is not None:
        tensors.append(torch.tensor(targets, dtype=torch.float32).unsqueeze(1))
    return TensorDataset(*tensors)


def _forward_batch(
    model: Any,
    batch: tuple[Any, ...],
    has_num: bool,
    has_cat: bool,
    device: Any,
    has_target: bool = True,
) -> tuple[Any, Any | None]:
    """Execute model forward pass with mixed numerical and categorical batch tensors."""
    if has_num and has_cat:
        b_num = batch[0].to(device)
        b_cat = batch[1].to(device)
        b_y = batch[2].to(device) if has_target else None
        logits = model(b_num, b_cat)
    elif has_num:
        b_num = batch[0].to(device)
        b_y = batch[1].to(device) if has_target else None
        logits = model(b_num, None)
    else:
        b_cat = batch[0].to(device)
        b_y = batch[1].to(device) if has_target else None
        logits = model(None, b_cat)
    return logits, b_y


def _evaluate_candidate(
    X_in_tr: pd.DataFrame,
    y_in_tr: np.ndarray,
    X_in_val: pd.DataFrame,
    y_in_val: np.ndarray,
    candidate_dict: dict[str, Any],
    base_params: dict[str, Any],
    categorical_features: list[str],
    max_epochs: int,
    patience: int,
    inner_seed: int,
    device: Any,
    batch_size: int,
    min_delta: float = 1e-6,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5,
) -> dict[str, Any]:
    """Train single FT-Transformer candidate on inner-train and evaluate on inner-val.

    Preprocessors are fitted strictly on X_in_tr and discarded upon return.
    """
    import torch
    import torch.nn as nn
    from rtdl_revisiting_models import FTTransformer
    from torch.utils.data import DataLoader

    cand_id = str(candidate_dict.get("id", "FT_custom"))
    cand_params = copy.deepcopy(base_params)
    cand_params.update(candidate_dict.get("params", {}))

    # Deterministic seeding for this candidate
    np.random.seed(inner_seed)
    torch.manual_seed(inner_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inner_seed)

    # 1. Fit preprocessors ON INNER-TRAINING ROWS ONLY
    scaler_in, cat_maps_in, cat_cards_in, feat_names_in, num_feats_in = _fit_preprocessors(
        X_in_tr, categorical_features
    )
    x_num_tr, x_cat_tr = _transform_features(
        X_in_tr, categorical_features, num_feats_in, scaler_in, cat_maps_in
    )
    x_num_val, x_cat_val = _transform_features(
        X_in_val, categorical_features, num_feats_in, scaler_in, cat_maps_in
    )

    # 2. Weighting driven by candidate params['weighting']
    weighting = str(cand_params.get("weighting", "none")).lower()
    cand_pos_weight: float | None = None
    if weighting == "dynamic_ratio":
        pos_count = int(np.sum(y_in_tr == 1))
        neg_count = int(np.sum(y_in_tr == 0))
        if pos_count == 0 or neg_count == 0:
            raise ValueError(
                f"Inner training rows have empty class: pos={pos_count}, neg={neg_count}"
            )
        cand_pos_weight = float(neg_count / pos_count)
        pos_weight_tensor = torch.tensor([cand_pos_weight], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        weighting = "none"
        criterion = nn.BCEWithLogitsLoss()

    # 3. Instantiate fresh candidate FTTransformer
    n_cont = x_num_tr.shape[1] if x_num_tr is not None else 0
    cat_cards = cat_cards_in if (x_cat_tr is not None and cat_cards_in) else None

    cand_model = FTTransformer(
        n_cont_features=n_cont,
        cat_cardinalities=cat_cards,
        d_out=1,
        n_blocks=int(cand_params.get("n_blocks", 3)),
        d_block=int(cand_params.get("d_block", 64)),
        attention_n_heads=int(cand_params.get("attention_n_heads", 4)),
        attention_dropout=float(cand_params.get("attention_dropout", 0.1)),
        ffn_d_hidden=int(cand_params.get("ffn_d_hidden", 128)),
        ffn_d_hidden_multiplier=None,
        ffn_dropout=float(cand_params.get("ffn_dropout", 0.1)),
        residual_dropout=float(cand_params.get("residual_dropout", 0.0)),
    ).to(device)

    # 4. DataLoaders
    cand_bs = int(cand_params.get("batch_size", batch_size))
    train_ds = _make_tensor_dataset(x_num_tr, x_cat_tr, y_in_tr)
    val_ds = _make_tensor_dataset(x_num_val, x_cat_val, y_in_val)
    train_loader = DataLoader(train_ds, batch_size=cand_bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cand_bs, shuffle=False)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        cand_model.parameters(),
        lr=float(cand_params.get("learning_rate", 0.001)),
        weight_decay=float(cand_params.get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    # 6. Training loop with early stopping on Average Precision
    best_pr_auc = -np.inf
    best_epoch = 1
    epochs_without_improvement = 0
    has_num = x_num_tr is not None
    has_cat = x_cat_tr is not None

    cand_t0 = time.perf_counter()
    cand_max_epochs = int(cand_params.get("max_epochs", max_epochs))
    cand_patience = int(cand_params.get("patience", patience))

    for epoch in range(1, cand_max_epochs + 1):
        cand_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits, b_y = _forward_batch(cand_model, batch, has_num, has_cat, device)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

        cand_model.eval()
        val_probs_list = []
        val_targets_list = []
        with torch.no_grad():
            for batch in val_loader:
                logits, b_y = _forward_batch(cand_model, batch, has_num, has_cat, device)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                targets = b_y.squeeze(1).cpu().numpy()
                val_probs_list.append(probs)
                val_targets_list.append(targets)

        val_probs_arr = np.concatenate(val_probs_list)
        val_targets_arr = np.concatenate(val_targets_list)
        inner_pr_auc = float(average_precision_score(val_targets_arr, val_probs_arr))

        scheduler.step(inner_pr_auc)

        if inner_pr_auc > best_pr_auc + min_delta:
            best_pr_auc = inner_pr_auc
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cand_patience:
            break

    cand_duration = float(time.perf_counter() - cand_t0)
    selected_epochs = max(1, int(best_epoch))

    # Memory cleanup
    del cand_model, optimizer, scheduler, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "candidate_id": cand_id,
        "inner_ap": float(best_pr_auc),
        "ap_score": float(best_pr_auc),
        "selected_epoch_count": selected_epochs,
        "selected_epochs": selected_epochs,
        "weighting": weighting,
        "pos_weight": cand_pos_weight,
        "inner_pos_weight": cand_pos_weight,
        "actual_pos_weight": cand_pos_weight,
        "fit_time_seconds": cand_duration,
        "runtime_seconds": cand_duration,
        "params": cand_params,
    }


# =============================================================================
# FTTransformerAdapter Main Implementation
# =============================================================================

class FTTransformerAdapter:
    """FT-Transformer model adapter with strict nested candidate selection and outer refit.

    This adapter adheres strictly to the project evaluation methodology:
    - Receives ONLY outer-training rows in `fit(X, y)`
    - Automatically evaluates candidates (FT01–FT08) via stratified inner validation (15% val, 85% train)
    - Early-stops each candidate on inner-validation Average Precision
    - Captures winning candidate and positive integer `selected_epoch_count_`
    - Preprocessing is fitted on inner-train rows only during candidate evaluation
    - Refits fresh model on 100% of outer-training rows using `selected_epoch_count_`
    - Refits fresh preprocessing on 100% of outer-training rows for final inference
    - Recomputes dynamic `pos_weight` on 100% of outer-training rows if winner is dynamic_ratio
    - Outer refit uses NO validation set, NO early stopping, and NO LR scheduler
    - Provides defensive `get_selection_evidence()` with exhaustive diagnostic metadata
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        candidate_id: str | None = None,
        params: dict[str, Any] | None = None,
        name: str = "FT-Transformer",
        model_id: str = "ft_transformer",
        categorical_features: list[str] | None = None,
        inner_val_fraction: float | None = None,
        patience: int | None = None,
        max_epochs: int | None = None,
        refit_outer_train: bool | None = None,
        random_seed: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Resolve a fixed experiment or configured candidate search and initialize unfitted state.

        Training partitions, category mappings and class weights are established later during fit.
        """
        self.name = name
        self.model_id = model_id
        self.candidate_id = candidate_id or "default"
        self._fitted = False
        self._fitted_on_all_outer_rows = False
        self._model: Any = None
        self._scaler: Any = None
        self._category_maps: dict[str, dict[Any, int]] = {}
        self._cat_cardinalities: list[int] = []
        self._feature_names: list[str] = []
        self._num_features: list[str] = []
        self._pos_weight: float | None = None
        self.selected_epoch_count_: int | None = None
        self._inner_eval_results: dict[str, Any] = {}
        self._candidate_pool: list[dict[str, Any]] = []
        self.candidate_selection_time_seconds_: float = 0.0
        self.refit_time_seconds_: float = 0.0
        self._epochs_trained: int = 0
        self._best_epoch: int | None = None
        self._best_inner_pr_auc: float | None = None
        self._inner_train_size: int | None = None
        self._inner_validation_size: int | None = None

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
            if "params" in config and isinstance(config["params"], Mapping):
                resolved = dict(config)
                self.candidate_id = str(resolved.get("candidate_id", self.candidate_id))
            elif "candidates" in config and isinstance(config["candidates"], Sequence):
                from src.models.config import resolve_candidate

                self._candidate_pool = [
                    copy.deepcopy(dict(c))
                    for c in config["candidates"]
                    if isinstance(c, Mapping)
                ]

                if candidate_id in ("auto", "tune"):
                    self.candidate_id = "auto"
                    resolved = resolve_candidate(config, candidate_id=None)
                else:
                    resolved = resolve_candidate(config, candidate_id=candidate_id)
                    self.candidate_id = str(resolved.get("candidate_id", "default"))
            else:
                resolved = dict(config)

            model_meta = resolved.get("model", {})
            self.name = str(model_meta.get("name", self.name))
            self.model_id = str(model_meta.get("model_id", self.model_id))

            repro = resolved.get("reproducibility", {})
            self.random_seed = int(
                random_seed
                if random_seed is not None
                else repro.get("seed", repro.get("random_seed", 42))
            )

            inner_val = resolved.get("inner_validation", {})
            self.inner_val_fraction = float(
                inner_val_fraction
                if inner_val_fraction is not None
                else inner_val.get("fraction", 0.15)
            )
            self.patience = int(
                patience
                if patience is not None
                else inner_val.get("patience", 15)
            )
            self.max_epochs = int(
                max_epochs
                if max_epochs is not None
                else inner_val.get("max_epochs", 100)
            )
            self.refit_outer_train = bool(
                refit_outer_train
                if refit_outer_train is not None
                else inner_val.get("refit_outer_train", True)
            )

            self.params = copy.deepcopy(dict(resolved.get("params", {})))
            self.batch_size = int(
                batch_size if batch_size is not None else self.params.get("batch_size", 512)
            )

        # ---------------------------------------------------------------------
        # Case B: Initialized via Direct Parameters (Unit Test / Direct Mode)
        # ---------------------------------------------------------------------
        elif params is not None:
            self.candidate_id = candidate_id or "direct"
            self.params = copy.deepcopy(params)
            self.random_seed = int(
                random_seed
                if random_seed is not None
                else self.params.get("seed", self.params.get("random_seed", 42))
            )
            self.inner_val_fraction = float(
                inner_val_fraction
                if inner_val_fraction is not None
                else self.params.get("validation_fraction_inner", 0.15)
            )
            self.patience = int(
                patience
                if patience is not None
                else self.params.get("patience", 15)
            )
            self.max_epochs = int(
                max_epochs
                if max_epochs is not None
                else self.params.get("max_epochs", 100)
            )
            self.refit_outer_train = bool(
                refit_outer_train
                if refit_outer_train is not None
                else False
            )
            self.batch_size = int(
                batch_size if batch_size is not None else self.params.get("batch_size", 512)
            )

        # ---------------------------------------------------------------------
        # Case C: Zero-Argument Fallback (Protocol Compliance & Inspection)
        # ---------------------------------------------------------------------
        else:
            self.params = {
                "n_blocks": 3,
                "d_block": 64,
                "attention_n_heads": 4,
                "ffn_d_hidden": 128,
                "ffn_dropout": 0.1,
                "attention_dropout": 0.1,
                "residual_dropout": 0.0,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "seed": 42,
                "weighting": "none",
                "max_epochs": 100,
                "patience": 15,
                "validation_fraction_inner": 0.15,
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
                "min_delta": 1e-6,
                "batch_size": 512,
            }
            self.random_seed = 42
            self.inner_val_fraction = 0.15
            self.patience = 15
            self.max_epochs = 100
            self.refit_outer_train = False
            self.batch_size = 512

    @property
    def selected_epoch_count(self) -> int | None:
        """Expose positive integer count of executed refit epochs."""
        return self.selected_epoch_count_

    # =========================================================================
    # Preprocessing Interface
    # =========================================================================

    def _preprocess(
        self,
        X: pd.DataFrame,
        fit: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Scale numerical features and encode categorical features."""
        if fit:
            (
                self._scaler,
                self._category_maps,
                self._cat_cardinalities,
                self._feature_names,
                self._num_features,
            ) = _fit_preprocessors(X, self.categorical_features)

        return _transform_features(
            X,
            self.categorical_features,
            self._num_features,
            self._scaler,
            self._category_maps,
        )

    # =========================================================================
    # Model Fitting & Candidate Selection
    # =========================================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> FTTransformerAdapter:
        """Fit FT-Transformer model on outer-training data with strict nested tuning.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor columns for outer-training rows only.
        y : pd.Series or np.ndarray
            Binary target values for outer-training rows only.
        epochs : int or None
            Optional override for maximum epochs.
        batch_size : int or None
            Optional override for batch size.

        Returns
        -------
        FTTransformerAdapter
            Fitted adapter instance.
        """
        try:
            import torch
            import torch.nn as nn
            from rtdl_revisiting_models import FTTransformer
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise ImportError(
                "torch and rtdl-revisiting-models are required for FTTransformerAdapter."
            ) from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Basic validations
        self._feature_names = list(X.columns)
        y_arr = np.asarray(y, dtype=np.float32)

        if len(X) != len(y_arr):
            raise ValueError("X and y must contain the same number of rows.")

        unique_targets = set(np.unique(y_arr).tolist())
        if not unique_targets.issubset({0.0, 1.0}):
            raise ValueError("FTTransformerAdapter requires binary target encoded as 0 and 1.")

        if len(unique_targets) != 2:
            raise ValueError("Training data must contain both positive and negative target classes.")

        fit_batch_size = int(batch_size or self.batch_size)
        fit_max_epochs = int(epochs or self.max_epochs)

        # Check if inner validation is enabled
        use_inner_val = (
            self.inner_val_fraction > 0.0
            and self.patience > 0
            and len(X) >= 20
        )

        # =====================================================================
        # Branch 1: Automatic Candidate Selection Mode across FT01–FT08
        # =====================================================================
        if self.candidate_id in ("auto", "tune") and self._candidate_pool and use_inner_val:
            # 1. Partition outer-training rows into stratified inner-train and inner-val
            row_indices = np.arange(len(X))
            inner_tr_idx, inner_val_idx = train_test_split(
                row_indices,
                test_size=self.inner_val_fraction,
                random_state=self.random_seed,
                stratify=y_arr,
            )

            X_in_tr = X.iloc[inner_tr_idx].copy().reset_index(drop=True)
            y_in_tr = y_arr[inner_tr_idx]
            X_in_val = X.iloc[inner_val_idx].copy().reset_index(drop=True)
            y_in_val = y_arr[inner_val_idx]

            self._inner_train_size = len(X_in_tr)
            self._inner_validation_size = len(X_in_val)

            sel_start = time.perf_counter()
            best_ap = -1.0
            best_cid: str | None = None
            best_epochs: int = 1
            candidate_evals: dict[str, Any] = {}

            # 2. Evaluate each candidate independently
            for cand in self._candidate_pool:
                cid = str(cand.get("id"))
                eval_res = _evaluate_candidate(
                    X_in_tr=X_in_tr,
                    y_in_tr=y_in_tr,
                    X_in_val=X_in_val,
                    y_in_val=y_in_val,
                    candidate_dict=cand,
                    base_params=self.params,
                    categorical_features=self.categorical_features,
                    max_epochs=fit_max_epochs,
                    patience=self.patience,
                    inner_seed=self.random_seed,
                    device=device,
                    batch_size=fit_batch_size,
                    min_delta=float(self.params.get("min_delta", 1e-6)),
                    scheduler_factor=float(self.params.get("scheduler_factor", 0.5)),
                    scheduler_patience=int(self.params.get("scheduler_patience", 5)),
                )
                candidate_evals[cid] = eval_res

                # Deterministic selection on highest inner AP
                c_ap = eval_res["inner_ap"]
                if c_ap > best_ap:
                    best_ap = c_ap
                    best_cid = cid
                    best_epochs = eval_res["selected_epoch_count"]

            self.candidate_selection_time_seconds_ = float(time.perf_counter() - sel_start)
            self.candidate_id = best_cid or str(self._candidate_pool[0].get("id"))
            self.selected_epoch_count_ = max(1, int(best_epochs))
            self._best_epoch = self.selected_epoch_count_
            self._best_inner_pr_auc = float(best_ap)

            # Update params with winning candidate overrides
            for cand in self._candidate_pool:
                if str(cand.get("id")) == self.candidate_id:
                    self.params.update(cand.get("params", {}))
                    break

            self._inner_eval_results = {
                "metric": "average_precision",
                "best_candidate_id": self.candidate_id,
                "best_inner_val_ap": float(best_ap),
                "selected_epoch_count": self.selected_epoch_count_,
                "inner_train_size": len(X_in_tr),
                "inner_val_size": len(X_in_val),
                "candidate_selection_time_seconds": self.candidate_selection_time_seconds_,
                "all_candidates": candidate_evals,
            }

            # 3. Outer Refit Stage: 100% of outer-training rows
            refit_start = time.perf_counter()
            self._refit_fresh_outer_model(
                X_outer=X,
                y_outer=y_arr,
                device=device,
                selected_epochs=self.selected_epoch_count_,
                batch_size=fit_batch_size,
            )
            self.refit_time_seconds_ = float(time.perf_counter() - refit_start)
            self._inner_eval_results["refit_time_seconds"] = self.refit_time_seconds_
            self._fitted_on_all_outer_rows = True

        # =====================================================================
        # Branch 2: Single-Candidate Mode with Inner Early Stopping & Refit
        # =====================================================================
        elif use_inner_val:
            row_indices = np.arange(len(X))
            inner_tr_idx, inner_val_idx = train_test_split(
                row_indices,
                test_size=self.inner_val_fraction,
                random_state=self.random_seed,
                stratify=y_arr,
            )

            X_in_tr = X.iloc[inner_tr_idx].copy().reset_index(drop=True)
            y_in_tr = y_arr[inner_tr_idx]
            X_in_val = X.iloc[inner_val_idx].copy().reset_index(drop=True)
            y_in_val = y_arr[inner_val_idx]

            self._inner_train_size = len(X_in_tr)
            self._inner_validation_size = len(X_in_val)

            sel_start = time.perf_counter()
            single_cand_dict = {"id": self.candidate_id, "params": self.params}
            eval_res = _evaluate_candidate(
                X_in_tr=X_in_tr,
                y_in_tr=y_in_tr,
                X_in_val=X_in_val,
                y_in_val=y_in_val,
                candidate_dict=single_cand_dict,
                base_params=self.params,
                categorical_features=self.categorical_features,
                max_epochs=fit_max_epochs,
                patience=self.patience,
                inner_seed=self.random_seed,
                device=device,
                batch_size=fit_batch_size,
                min_delta=float(self.params.get("min_delta", 1e-6)),
                scheduler_factor=float(self.params.get("scheduler_factor", 0.5)),
                scheduler_patience=int(self.params.get("scheduler_patience", 5)),
            )

            self.candidate_selection_time_seconds_ = float(time.perf_counter() - sel_start)
            self.selected_epoch_count_ = max(1, int(eval_res["selected_epoch_count"]))
            self._best_epoch = self.selected_epoch_count_
            self._best_inner_pr_auc = float(eval_res["inner_ap"])

            self._inner_eval_results = {
                "metric": "average_precision",
                "best_candidate_id": self.candidate_id,
                "best_inner_val_ap": self._best_inner_pr_auc,
                "inner_val_ap": self._best_inner_pr_auc,
                "selected_epoch_count": self.selected_epoch_count_,
                "inner_train_size": len(X_in_tr),
                "inner_val_size": len(X_in_val),
                "candidate_selection_time_seconds": self.candidate_selection_time_seconds_,
                "all_candidates": {self.candidate_id: eval_res},
            }

            if self.refit_outer_train:
                refit_start = time.perf_counter()
                self._refit_fresh_outer_model(
                    X_outer=X,
                    y_outer=y_arr,
                    device=device,
                    selected_epochs=self.selected_epoch_count_,
                    batch_size=fit_batch_size,
                )
                self.refit_time_seconds_ = float(time.perf_counter() - refit_start)
                self._inner_eval_results["refit_time_seconds"] = self.refit_time_seconds_
                self._fitted_on_all_outer_rows = True
            else:
                # Select epochs above, then train fresh on inner-training rows only.
                refit_start = time.perf_counter()
                # Preprocess inner training data
                self._preprocess(X_in_tr, fit=True)
                # Fit model on inner train for selected epochs
                self._refit_fresh_outer_model(
                    X_outer=X_in_tr,
                    y_outer=y_in_tr,
                    device=device,
                    selected_epochs=self.selected_epoch_count_,
                    batch_size=fit_batch_size,
                )
                self._fitted_on_all_outer_rows = False
                self.refit_time_seconds_ = float(time.perf_counter() - refit_start)
                self._inner_eval_results["refit_time_seconds"] = self.refit_time_seconds_

        # =====================================================================
        # Branch 3: Direct Mode (Zero inner-val or small dataset)
        # =====================================================================
        else:
            self.candidate_selection_time_seconds_ = 0.0
            self.selected_epoch_count_ = max(1, fit_max_epochs)
            self._best_epoch = self.selected_epoch_count_

            refit_start = time.perf_counter()
            self._refit_fresh_outer_model(
                X_outer=X,
                y_outer=y_arr,
                device=device,
                selected_epochs=self.selected_epoch_count_,
                batch_size=fit_batch_size,
            )
            self.refit_time_seconds_ = float(time.perf_counter() - refit_start)

            self._inner_eval_results = {
                "candidate_selection_time_seconds": 0.0,
                "refit_time_seconds": self.refit_time_seconds_,
                "all_candidates": {},
            }

        self._fitted = True
        return self

    def _refit_fresh_outer_model(
        self,
        X_outer: pd.DataFrame,
        y_outer: np.ndarray,
        device: Any,
        selected_epochs: int,
        batch_size: int,
    ) -> None:
        """Train a fresh model on the supplied rows for the selected epoch count.

        The caller supplies all outer-training rows for a full refit, or only
        inner-training rows when refit_outer_train=False.

        Guarantees:
        - Fresh StandardScaler and category vocabularies fitted on X_outer
        - pos_weight recomputed from y_outer if weighting == 'dynamic_ratio'
        - Ordinary unweighted BCE if weighting == 'none'
        - Exactly `selected_epochs` epochs executed (positive integer >= 1)
        - NO validation set
        - NO early stopping
        - NO learning-rate scheduler
        """
        import torch
        import torch.nn as nn
        from rtdl_revisiting_models import FTTransformer
        from torch.utils.data import DataLoader

        # Deterministic seed for refit
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        # 1. Fit fresh preprocessors on the supplied final-training partition.
        (
            self._scaler,
            self._category_maps,
            self._cat_cardinalities,
            self._feature_names,
            self._num_features,
        ) = _fit_preprocessors(X_outer, self.categorical_features)

        x_num_outer, x_cat_outer = _transform_features(
            X_outer, self.categorical_features, self._num_features, self._scaler, self._category_maps
        )

        # 2. Derive pos_weight from that same partition's labels.
        weighting = str(self.params.get("weighting", "none")).lower()
        if weighting == "dynamic_ratio":
            pos_count = int(np.sum(y_outer == 1))
            neg_count = int(np.sum(y_outer == 0))
            if pos_count == 0 or neg_count == 0:
                raise ValueError(
                    f"Outer training rows have empty class: pos={pos_count}, neg={neg_count}"
                )
            self._pos_weight = float(neg_count / pos_count)
            pos_weight_tensor = torch.tensor([self._pos_weight], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self._pos_weight = None
            criterion = nn.BCEWithLogitsLoss()

        # 3. Instantiate completely fresh FTTransformer architecture
        n_cont = x_num_outer.shape[1] if x_num_outer is not None else 0
        cat_cards = (
            self._cat_cardinalities
            if (x_cat_outer is not None and self._cat_cardinalities)
            else None
        )

        refit_model = FTTransformer(
            n_cont_features=n_cont,
            cat_cardinalities=cat_cards,
            d_out=1,
            n_blocks=int(self.params.get("n_blocks", 3)),
            d_block=int(self.params.get("d_block", 64)),
            attention_n_heads=int(self.params.get("attention_n_heads", 4)),
            attention_dropout=float(self.params.get("attention_dropout", 0.1)),
            ffn_d_hidden=int(self.params.get("ffn_d_hidden", 128)),
            ffn_d_hidden_multiplier=None,
            ffn_dropout=float(self.params.get("ffn_dropout", 0.1)),
            residual_dropout=float(self.params.get("residual_dropout", 0.0)),
        ).to(device)

        # 4. DataLoader and Optimizer (NO scheduler, NO validation)
        outer_ds = _make_tensor_dataset(x_num_outer, x_cat_outer, y_outer)
        refit_bs = int(self.params.get("batch_size", batch_size))
        outer_loader = DataLoader(outer_ds, batch_size=refit_bs, shuffle=True)

        optimizer = torch.optim.AdamW(
            refit_model.parameters(),
            lr=float(self.params.get("learning_rate", 0.001)),
            weight_decay=float(self.params.get("weight_decay", 1e-5)),
        )

        has_num = x_num_outer is not None
        has_cat = x_cat_outer is not None

        # 5. Train for exactly selected_epochs
        epochs_to_run = max(1, int(selected_epochs))
        for epoch in range(1, epochs_to_run + 1):
            refit_model.train()
            for batch in outer_loader:
                optimizer.zero_grad()
                logits, b_y = _forward_batch(refit_model, batch, has_num, has_cat, device)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

        self._model = refit_model
        self._fitted_on_all_outer_rows = True
        self._epochs_trained = epochs_to_run

    # =========================================================================
    # Prediction Interface
    # =========================================================================

    def predict_proba(
        self,
        X: pd.DataFrame,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Predict positive-class probabilities for input features."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Adapter is not fitted. Call fit() first.")

        import torch
        from torch.utils.data import DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._model.eval()

        # Transform using preprocessors fitted on outer-training rows
        x_num, x_cat = self._preprocess(X, fit=False)
        dataset = _make_tensor_dataset(x_num, x_cat)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        has_num = x_num is not None
        has_cat = x_cat is not None
        probs_list = []

        with torch.no_grad():
            for batch in loader:
                logits, _ = _forward_batch(
                    self._model, batch, has_num, has_cat, device, has_target=False
                )
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                probs_list.append(probs)

        if probs_list:
            probs_arr = np.concatenate(probs_list)
            if probs_arr.ndim == 0:
                probs_arr = np.expand_dims(probs_arr, 0)
            return probs_arr.astype(float)
        return np.array([], dtype=float)

    # =========================================================================
    # Serialization (Save & Load)
    # =========================================================================

    def save(self, directory: Path) -> dict[str, Any]:
        """Serialize model weights and preprocessing state to disk."""
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_weights_file = directory / "ft_transformer_weights.pt"
        if self._model is not None:
            torch.save(self._model.state_dict(), model_weights_file)

        preprocessor_file = directory / "ft_transformer_preprocessor.pkl"
        with open(preprocessor_file, "wb") as file:
            pickle.dump(
                {
                    "scaler": self._scaler,
                    "category_maps": self._category_maps,
                    "cat_cardinalities": self._cat_cardinalities,
                    "params": self.params,
                    "categorical_features": self.categorical_features,
                    "feature_names": self._feature_names,
                    "num_features": self._num_features,
                    "pos_weight": self._pos_weight,
                    "best_epoch": self._best_epoch,
                    "selected_epoch_count": self.selected_epoch_count_,
                    "best_inner_pr_auc": self._best_inner_pr_auc,
                    "epochs_trained": self._epochs_trained,
                    "inner_train_size": self._inner_train_size,
                    "inner_validation_size": self._inner_validation_size,
                    "candidate_id": self.candidate_id,
                    "fitted_on_all_outer_rows": self._fitted_on_all_outer_rows,
                    "candidate_selection_time_seconds": self.candidate_selection_time_seconds_,
                    "refit_time_seconds": self.refit_time_seconds_,
                    "inner_eval_results": self._inner_eval_results,
                },
                file,
            )

        metadata = self.get_metadata()
        metadata["weights_file"] = model_weights_file.name
        metadata["preprocessor_file"] = preprocessor_file.name

        meta_file = directory / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        return metadata

    @classmethod
    def load(cls, directory: Path) -> FTTransformerAdapter:
        """Load a serialized FT-Transformer adapter from disk."""
        import torch
        from rtdl_revisiting_models import FTTransformer

        directory = Path(directory)
        preprocessor_file = directory / "ft_transformer_preprocessor.pkl"
        model_weights_file = directory / "ft_transformer_weights.pt"

        adapter = cls()
        if not (preprocessor_file.exists() and model_weights_file.exists()):
            return adapter

        with open(preprocessor_file, "rb") as file:
            state = pickle.load(file)

        adapter._scaler = state["scaler"]
        adapter._category_maps = state["category_maps"]
        adapter._cat_cardinalities = state["cat_cardinalities"]
        adapter.params = state["params"]
        adapter.categorical_features = state["categorical_features"]
        adapter._feature_names = state["feature_names"]
        adapter._num_features = state["num_features"]
        adapter._pos_weight = state.get("pos_weight")
        adapter._best_epoch = state.get("best_epoch")
        adapter.selected_epoch_count_ = state.get(
            "selected_epoch_count", state.get("best_epoch")
        )
        adapter._best_inner_pr_auc = state.get("best_inner_pr_auc")
        adapter._epochs_trained = state.get("epochs_trained", 0)
        adapter._inner_train_size = state.get("inner_train_size")
        adapter._inner_validation_size = state.get("inner_validation_size")
        adapter.candidate_id = state.get("candidate_id", "default")
        adapter._fitted_on_all_outer_rows = state.get("fitted_on_all_outer_rows", False)
        adapter.candidate_selection_time_seconds_ = float(
            state.get("candidate_selection_time_seconds", 0.0)
        )
        adapter.refit_time_seconds_ = float(state.get("refit_time_seconds", 0.0))
        adapter._inner_eval_results = state.get("inner_eval_results", {})

        n_cont = len(adapter._num_features)
        cat_cards = (
            adapter._cat_cardinalities if adapter._cat_cardinalities else None
        )

        model = FTTransformer(
            n_cont_features=n_cont,
            cat_cardinalities=cat_cards,
            d_out=1,
            n_blocks=int(adapter.params.get("n_blocks", 3)),
            d_block=int(adapter.params.get("d_block", 64)),
            attention_n_heads=int(adapter.params.get("attention_n_heads", 4)),
            attention_dropout=float(adapter.params.get("attention_dropout", 0.1)),
            ffn_d_hidden=int(adapter.params.get("ffn_d_hidden", 128)),
            ffn_d_hidden_multiplier=None,
            ffn_dropout=float(adapter.params.get("ffn_dropout", 0.1)),
            residual_dropout=float(adapter.params.get("residual_dropout", 0.0)),
        )
        model.load_state_dict(
            torch.load(model_weights_file, map_location="cpu", weights_only=True)
        )
        adapter._model = model
        adapter._fitted = True
        return adapter

    # =========================================================================
    # Metadata & Evidence Reporting
    # =========================================================================

    def get_metadata(self) -> dict[str, Any]:
        """Return configuration, training diagnostics, and selection evidence."""
        meta: dict[str, Any] = {
            "name": self.name,
            "model_id": self.model_id,
            "candidate_id": self.candidate_id,
            "selected_candidate": self.candidate_id,
            "fitted": self._fitted,
            "selected_epoch_count": self.selected_epoch_count_,
            "best_epoch": self.selected_epoch_count_ or self._best_epoch,
            "best_inner_pr_auc": self._best_inner_pr_auc,
            "fitted_on_all_outer_rows": self._fitted_on_all_outer_rows,
            "refit_status": self._fitted_on_all_outer_rows,
            "inner_val_fraction": self.inner_val_fraction,
            "patience": self.patience,
            "max_epochs": self.max_epochs,
            "refit_outer_train": self.refit_outer_train,
            "final_fit_strategy": "fresh_model_for_selected_epoch_count",
            "final_fit_partition": (
                "outer_training" if self._fitted_on_all_outer_rows else "inner_training"
            ) if self._fitted else None,
            "best_weights_restored": False,
            "selection_checkpoint_used_for_final_predictions": False,
            "selection_learning_rate_scheduler": (
                "ReduceLROnPlateau" if self._inner_eval_results.get("all_candidates") else None
            ),
            "final_fit_learning_rate_scheduler": None,
            "random_seed": self.random_seed,
            "params": copy.deepcopy(self.params),
            "weighting_strategy": self.params.get("weighting", "none"),
            "categorical_features": list(self.categorical_features),
            "feature_names": list(self._feature_names),
            "num_features": list(self._num_features),
            "class_weighting": (
                ("outer_training_fold_positive_class_weight" if self._fitted_on_all_outer_rows
                 else "inner_training_positive_class_weight")
                if self._pos_weight is not None
                else "unweighted_bce"
            ),
            "pos_weight": self._pos_weight,
            "epochs_trained": self._epochs_trained,
            "inner_train_size": self._inner_train_size,
            "inner_validation_size": self._inner_validation_size,
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
                "FTTransformerAdapter is not fitted. Call fit() before retrieving selection evidence."
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
            "selected_epoch_count": self.selected_epoch_count_,
            "inner_train_rows": self._inner_eval_results.get("inner_train_size"),
            "inner_validation_rows": self._inner_eval_results.get("inner_val_size"),
            "inner_val_rows": self._inner_eval_results.get("inner_val_size"),
            "inner_val_fraction": self.inner_val_fraction,
            "inner_validation_fraction": self.inner_val_fraction,
            "random_seed": self.random_seed,
            "selected_weighting": self.params.get("weighting", "none"),
            "selected_pos_weight": self._pos_weight,
            "candidate_selection_time_seconds": float(self.candidate_selection_time_seconds_),
            "refit_time_seconds": float(self.refit_time_seconds_),
            "fitted_on_all_outer_rows": self._fitted_on_all_outer_rows,
            "metric": self._inner_eval_results.get("metric", "average_precision"),
            "all_candidates": copy.deepcopy(
                self._inner_eval_results.get("all_candidates", {})
            ),
        }
        return copy.deepcopy(evidence)
