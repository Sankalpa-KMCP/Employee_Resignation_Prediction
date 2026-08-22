"""FT-Transformer Model Adapter for Employee Resignation Prediction (Member 1).

Implements a leakage-safe FT-Transformer pipeline with:

- fold-local numerical preprocessing
- fold-local categorical preprocessing
- fold-local positive-class weighting
- stratified inner validation
- PR-AUC-based early stopping
- ReduceLROnPlateau learning-rate scheduling
- best-model restoration

The outer validation fold and formal holdout are never used for model fitting,
class-weight calculation, preprocessing fitting, or early stopping.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelAdapter


class FTTransformerAdapter:
    """FT-Transformer adapter with leakage-safe weighted training."""

    def __init__(
        self,
        name: str = "FT-Transformer",
        model_id: str = "ft_transformer",
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:

        self.name = name
        self.model_id = model_id

        self.params = params or {
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

            # Training controls
            "max_epochs": 100,
            "patience": 15,
            "validation_fraction_inner": 0.15,
            "scheduler_factor": 0.5,
            "scheduler_patience": 5,
            "min_delta": 1e-6,
        }

        self.categorical_features = categorical_features or [
            "Department",
            "Gender",
            "Job_Title",
            "Education_Level",
        ]

        self._model: Any = None
        self._scaler: Any = None

        self._category_maps: dict[str, dict[Any, int]] = {}

        self._feature_names: list[str] = []
        self._num_features: list[str] = []
        self._cat_cardinalities: list[int] = []

        # Class weighting
        self._pos_weight: float | None = None

        # Training diagnostics
        self._best_epoch: int | None = None
        self._best_inner_pr_auc: float | None = None
        self._epochs_trained: int = 0
        self._inner_train_size: int | None = None
        self._inner_validation_size: int | None = None

        self._fitted = False

    # ============================================================
    # Preprocessing
    # ============================================================

    def _preprocess(
        self,
        X: pd.DataFrame,
        fit: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Scale numerical features and encode categorical features."""

        from sklearn.preprocessing import StandardScaler

        X_proc = X.copy()

        self._num_features = [
            column
            for column in self._feature_names
            if column not in self.categorical_features
        ]

        # --------------------------------------------------------
        # Fit preprocessing on training data only
        # --------------------------------------------------------

        if fit:

            self._scaler = StandardScaler()

            if self._num_features:
                self._scaler.fit(
                    X_proc[self._num_features]
                )

            self._category_maps = {}
            self._cat_cardinalities = []

            for col in self.categorical_features:

                if col not in X_proc.columns:
                    continue

                unique_vals = sorted(
                    X_proc[col]
                    .dropna()
                    .unique()
                )

                # Reserve 0 for unknown / missing categories.
                # Known categories start from 1.
                mapping = {
                    value: index + 1
                    for index, value
                    in enumerate(unique_vals)
                }

                self._category_maps[col] = mapping

                # +1 because index 0 is reserved for unknown.
                self._cat_cardinalities.append(
                    len(mapping) + 1
                )

        # --------------------------------------------------------
        # Numerical matrix
        # --------------------------------------------------------

        x_num = None

        if (
            self._num_features
            and self._scaler is not None
        ):

            x_num = (
                self._scaler
                .transform(
                    X_proc[self._num_features]
                )
                .astype(np.float32)
            )

        # --------------------------------------------------------
        # Categorical matrix
        # --------------------------------------------------------

        x_cat = None

        if self.categorical_features:

            cat_arrays = []

            for col in self.categorical_features:

                if col not in X_proc.columns:
                    continue

                mapping = self._category_maps.get(
                    col,
                    {},
                )

                # Unknown categories map to the reserved 0 index.
                arr = (
                    X_proc[col]
                    .map(mapping)
                    .fillna(0)
                    .values
                    .astype(np.int64)
                )

                cat_arrays.append(arr)

            if cat_arrays:
                x_cat = np.column_stack(
                    cat_arrays
                )

        return x_num, x_cat

    # ============================================================
    # Model fitting
    # ============================================================

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        epochs: int | None = None,
        batch_size: int = 512,
    ) -> FTTransformerAdapter:
        """Fit a weighted FT-Transformer using inner-validation early stopping."""

        try:
            import torch
            import torch.nn as nn

            from rtdl_revisiting_models import FTTransformer

            from sklearn.metrics import average_precision_score
            from sklearn.model_selection import train_test_split

            from torch.utils.data import (
                DataLoader,
                TensorDataset,
            )

        except ImportError as exc:
            raise ImportError(
                "torch, scikit-learn, and "
                "rtdl-revisiting-models are required "
                "for FTTransformerAdapter."
            ) from exc

        # --------------------------------------------------------
        # Basic setup
        # --------------------------------------------------------

        self._feature_names = list(
            X.columns
        )

        y_arr = np.asarray(
            y,
            dtype=np.float32,
        )

        if len(X) != len(y_arr):
            raise ValueError(
                "X and y must contain the same number of rows."
            )

        unique_targets = set(
            np.unique(y_arr).tolist()
        )

        if not unique_targets.issubset(
            {0.0, 1.0}
        ):
            raise ValueError(
                "FTTransformerAdapter requires "
                "a binary target encoded as 0 and 1."
            )

        if len(unique_targets) != 2:
            raise ValueError(
                "Training data must contain both "
                "positive and negative target classes."
            )

        # --------------------------------------------------------
        # Device
        # --------------------------------------------------------

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # --------------------------------------------------------
        # Reproducibility
        # --------------------------------------------------------

        seed = int(
            self.params.get(
                "seed",
                42,
            )
        )

        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # --------------------------------------------------------
        # Training configuration
        # --------------------------------------------------------

        max_epochs = int(
            epochs
            if epochs is not None
            else self.params.get(
                "max_epochs",
                100,
            )
        )

        patience = int(
            self.params.get(
                "patience",
                15,
            )
        )

        validation_fraction = float(
            self.params.get(
                "validation_fraction_inner",
                0.15,
            )
        )

        scheduler_factor = float(
            self.params.get(
                "scheduler_factor",
                0.5,
            )
        )

        scheduler_patience = int(
            self.params.get(
                "scheduler_patience",
                5,
            )
        )

        min_delta = float(
            self.params.get(
                "min_delta",
                1e-6,
            )
        )

        if not 0.0 < validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction_inner must be "
                "strictly between 0 and 1."
            )

        if max_epochs <= 0:
            raise ValueError(
                "max_epochs must be greater than zero."
            )

        if patience <= 0:
            raise ValueError(
                "patience must be greater than zero."
            )

        # --------------------------------------------------------
        # Inner stratified split
        # --------------------------------------------------------
        #
        # IMPORTANT:
        # X and y supplied here contain ONLY the current
        # outer-training fold.
        #
        # The outer validation fold is not passed to fit().
        # --------------------------------------------------------

        row_indices = np.arange(
            len(X)
        )

        (
            inner_train_idx,
            inner_val_idx,
        ) = train_test_split(
            row_indices,
            test_size=validation_fraction,
            random_state=seed,
            stratify=y_arr,
        )

        X_inner_train = (
            X.iloc[inner_train_idx]
            .copy()
            .reset_index(drop=True)
        )

        X_inner_val = (
            X.iloc[inner_val_idx]
            .copy()
            .reset_index(drop=True)
        )

        y_inner_train = (
            y_arr[inner_train_idx]
            .astype(np.float32)
        )

        y_inner_val = (
            y_arr[inner_val_idx]
            .astype(np.float32)
        )

        self._inner_train_size = len(
            X_inner_train
        )

        self._inner_validation_size = len(
            X_inner_val
        )

        # --------------------------------------------------------
        # Fit preprocessing using INNER TRAINING data only
        # --------------------------------------------------------

        x_num_train, x_cat_train = self._preprocess(
            X_inner_train,
            fit=True,
        )

        x_num_val, x_cat_val = self._preprocess(
            X_inner_val,
            fit=False,
        )

        # --------------------------------------------------------
        # Inner-training positive-class weight
        # --------------------------------------------------------

        positive_count = int(
            np.sum(
                y_inner_train == 1
            )
        )

        negative_count = int(
            np.sum(
                y_inner_train == 0
            )
        )

        if positive_count == 0:
            raise ValueError(
                "Inner training data contains no "
                "positive samples."
            )

        if negative_count == 0:
            raise ValueError(
                "Inner training data contains no "
                "negative samples."
            )

        self._pos_weight = (
            negative_count
            / positive_count
        )

        pos_weight_tensor = torch.tensor(
            [self._pos_weight],
            dtype=torch.float32,
            device=device,
        )

        # --------------------------------------------------------
        # FT-Transformer architecture
        # --------------------------------------------------------

        n_cont_features = (
            x_num_train.shape[1]
            if x_num_train is not None
            else 0
        )

        cat_cardinalities = (
            self._cat_cardinalities
            if x_cat_train is not None
            else None
        )

        self._model = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_out=1,

            n_blocks=self.params.get(
                "n_blocks",
                3,
            ),

            d_block=self.params.get(
                "d_block",
                64,
            ),

            attention_n_heads=self.params.get(
                "attention_n_heads",
                4,
            ),

            attention_dropout=self.params.get(
                "attention_dropout",
                0.1,
            ),

            ffn_d_hidden=self.params.get(
                "ffn_d_hidden",
                128,
            ),

            # Required by rtdl_revisiting_models 0.0.2.
            ffn_d_hidden_multiplier=None,

            ffn_dropout=self.params.get(
                "ffn_dropout",
                0.1,
            ),

            residual_dropout=self.params.get(
                "residual_dropout",
                0.0,
            ),

        ).to(device)

        # --------------------------------------------------------
        # Tensor helper
        # --------------------------------------------------------

        def make_dataset(
            x_num: np.ndarray | None,
            x_cat: np.ndarray | None,
            targets: np.ndarray,
        ) -> TensorDataset:

            tensors = []

            if x_num is not None:
                tensors.append(
                    torch.tensor(
                        x_num,
                        dtype=torch.float32,
                    )
                )

            if x_cat is not None:
                tensors.append(
                    torch.tensor(
                        x_cat,
                        dtype=torch.long,
                    )
                )

            tensors.append(
                torch.tensor(
                    targets,
                    dtype=torch.float32,
                ).unsqueeze(1)
            )

            return TensorDataset(
                *tensors
            )

        train_dataset = make_dataset(
            x_num_train,
            x_cat_train,
            y_inner_train,
        )

        val_dataset = make_dataset(
            x_num_val,
            x_cat_val,
            y_inner_val,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        # --------------------------------------------------------
        # Optimizer
        # --------------------------------------------------------

        optimizer = torch.optim.AdamW(
            self._model.parameters(),

            lr=float(
                self.params.get(
                    "learning_rate",
                    0.001,
                )
            ),

            weight_decay=float(
                self.params.get(
                    "weight_decay",
                    1e-5,
                )
            ),
        )

        # --------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

        # --------------------------------------------------------
        # Weighted binary classification loss
        # --------------------------------------------------------

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor
        )

        # --------------------------------------------------------
        # Training / validation helpers
        # --------------------------------------------------------

        def forward_batch(
            batch: tuple[torch.Tensor, ...],
            has_num: bool,
            has_cat: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:

            if has_num and has_cat:

                b_num = batch[0].to(
                    device
                )

                b_cat = batch[1].to(
                    device
                )

                b_y = batch[2].to(
                    device
                )

                logits = self._model(
                    b_num,
                    b_cat,
                )

            elif has_num:

                b_num = batch[0].to(
                    device
                )

                b_y = batch[1].to(
                    device
                )

                logits = self._model(
                    b_num,
                    None,
                )

            else:

                b_cat = batch[0].to(
                    device
                )

                b_y = batch[1].to(
                    device
                )

                logits = self._model(
                    None,
                    b_cat,
                )

            return logits, b_y

        has_num = (
            x_num_train is not None
        )

        has_cat = (
            x_cat_train is not None
        )

        # --------------------------------------------------------
        # Early-stopping state
        # --------------------------------------------------------

        best_pr_auc = -np.inf
        best_epoch = 0
        best_state_dict = None

        epochs_without_improvement = 0

        # --------------------------------------------------------
        # Training loop
        # --------------------------------------------------------

        for epoch in range(
            1,
            max_epochs + 1,
        ):

            # ====================================================
            # TRAIN
            # ====================================================

            self._model.train()

            for batch in train_loader:

                optimizer.zero_grad()

                logits, b_y = forward_batch(
                    batch,
                    has_num,
                    has_cat,
                )

                loss = criterion(
                    logits,
                    b_y,
                )

                loss.backward()

                optimizer.step()

            # ====================================================
            # INNER VALIDATION
            # ====================================================

            self._model.eval()

            validation_probabilities = []
            validation_targets = []

            with torch.no_grad():

                for batch in val_loader:

                    logits, b_y = forward_batch(
                        batch,
                        has_num,
                        has_cat,
                    )

                    probabilities = (
                        torch.sigmoid(
                            logits
                        )
                        .squeeze(1)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    targets = (
                        b_y
                        .squeeze(1)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    validation_probabilities.append(
                        probabilities
                    )

                    validation_targets.append(
                        targets
                    )

            validation_probabilities_arr = np.concatenate(
                validation_probabilities
            )

            validation_targets_arr = np.concatenate(
                validation_targets
            )

            inner_pr_auc = float(
                average_precision_score(
                    validation_targets_arr,
                    validation_probabilities_arr,
                )
            )

            # Scheduler monitors PR-AUC.
            scheduler.step(
                inner_pr_auc
            )

            # ====================================================
            # BEST MODEL / EARLY STOPPING
            # ====================================================

            if (
                inner_pr_auc
                > best_pr_auc + min_delta
            ):

                best_pr_auc = (
                    inner_pr_auc
                )

                best_epoch = epoch

                # Store best weights on CPU.
                best_state_dict = {
                    key: value.detach()
                    .cpu()
                    .clone()

                    for key, value
                    in self._model.state_dict().items()
                }

                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

            self._epochs_trained = epoch

            if (
                epochs_without_improvement
                >= patience
            ):
                break

        # --------------------------------------------------------
        # Restore best PR-AUC model
        # --------------------------------------------------------

        if best_state_dict is None:
            raise RuntimeError(
                "Early stopping failed to capture "
                "a valid model state."
            )

        self._model.load_state_dict(
            best_state_dict
        )

        self._model.to(
            device
        )

        self._best_epoch = int(
            best_epoch
        )

        self._best_inner_pr_auc = float(
            best_pr_auc
        )

        self._fitted = True

        return self

    # ============================================================
    # Probability prediction
    # ============================================================

    def predict_proba(
        self,
        X: pd.DataFrame,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Predict positive-class probabilities."""

        if (
            not self._fitted
            or self._model is None
        ):
            raise RuntimeError(
                "Adapter is not fitted. "
                "Call fit() first."
            )

        import torch

        from torch.utils.data import (
            DataLoader,
            TensorDataset,
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self._model.to(
            device
        )

        self._model.eval()

        x_num, x_cat = self._preprocess(
            X,
            fit=False,
        )

        tensors = []

        if x_num is not None:

            tensors.append(
                torch.tensor(
                    x_num,
                    dtype=torch.float32,
                )
            )

        if x_cat is not None:

            tensors.append(
                torch.tensor(
                    x_cat,
                    dtype=torch.long,
                )
            )

        dataset = TensorDataset(
            *tensors
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        probs_list = []

        with torch.no_grad():

            for batch in loader:

                if (
                    x_num is not None
                    and x_cat is not None
                ):

                    b_num = batch[0].to(
                        device
                    )

                    b_cat = batch[1].to(
                        device
                    )

                    logits = self._model(
                        b_num,
                        b_cat,
                    )

                elif x_num is not None:

                    b_num = batch[0].to(
                        device
                    )

                    logits = self._model(
                        b_num,
                        None,
                    )

                else:

                    b_cat = batch[0].to(
                        device
                    )

                    logits = self._model(
                        None,
                        b_cat,
                    )

                probs = (
                    torch.sigmoid(
                        logits
                    )
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )

                probs_list.append(
                    probs
                )

        return np.concatenate(
            probs_list
        ).astype(float)

    # ============================================================
    # Serialization
    # ============================================================

    def save(
        self,
        directory: Path,
    ) -> dict[str, Any]:
        """Serialize model weights and preprocessing state."""

        import torch

        directory = Path(
            directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_weights_file = (
            directory
            / "ft_transformer_weights.pt"
        )

        if self._model is not None:

            torch.save(
                self._model.state_dict(),
                model_weights_file,
            )

        preprocessor_file = (
            directory
            / "ft_transformer_preprocessor.pkl"
        )

        with open(
            preprocessor_file,
            "wb",
        ) as file:

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

                    "best_inner_pr_auc": (
                        self._best_inner_pr_auc
                    ),

                    "epochs_trained": (
                        self._epochs_trained
                    ),

                    "inner_train_size": (
                        self._inner_train_size
                    ),

                    "inner_validation_size": (
                        self._inner_validation_size
                    ),
                },
                file,
            )

        metadata = self.get_metadata()

        metadata["weights_file"] = (
            model_weights_file.name
        )

        metadata["preprocessor_file"] = (
            preprocessor_file.name
        )

        return metadata

    # ============================================================
    # Loading
    # ============================================================

    @classmethod
    def load(
        cls,
        directory: Path,
    ) -> FTTransformerAdapter:
        """Load a serialized FT-Transformer adapter."""

        import torch

        from rtdl_revisiting_models import (
            FTTransformer,
        )

        directory = Path(
            directory
        )

        preprocessor_file = (
            directory
            / "ft_transformer_preprocessor.pkl"
        )

        model_weights_file = (
            directory
            / "ft_transformer_weights.pt"
        )

        adapter = cls()

        if not (
            preprocessor_file.exists()
            and model_weights_file.exists()
        ):
            return adapter

        with open(
            preprocessor_file,
            "rb",
        ) as file:

            state = pickle.load(
                file
            )

        adapter._scaler = (
            state["scaler"]
        )

        adapter._category_maps = (
            state["category_maps"]
        )

        adapter._cat_cardinalities = (
            state["cat_cardinalities"]
        )

        adapter.params = (
            state["params"]
        )

        adapter.categorical_features = (
            state["categorical_features"]
        )

        adapter._feature_names = (
            state["feature_names"]
        )

        adapter._num_features = (
            state["num_features"]
        )

        adapter._pos_weight = (
            state.get(
                "pos_weight"
            )
        )

        adapter._best_epoch = (
            state.get(
                "best_epoch"
            )
        )

        adapter._best_inner_pr_auc = (
            state.get(
                "best_inner_pr_auc"
            )
        )

        adapter._epochs_trained = (
            state.get(
                "epochs_trained",
                0,
            )
        )

        adapter._inner_train_size = (
            state.get(
                "inner_train_size"
            )
        )

        adapter._inner_validation_size = (
            state.get(
                "inner_validation_size"
            )
        )

        n_cont = len(
            adapter._num_features
        )

        cat_cards = (
            adapter._cat_cardinalities
            if adapter._cat_cardinalities
            else None
        )

        model = FTTransformer(
            n_cont_features=n_cont,
            cat_cardinalities=cat_cards,
            d_out=1,

            n_blocks=adapter.params.get(
                "n_blocks",
                3,
            ),

            d_block=adapter.params.get(
                "d_block",
                64,
            ),

            attention_n_heads=adapter.params.get(
                "attention_n_heads",
                4,
            ),

            attention_dropout=adapter.params.get(
                "attention_dropout",
                0.1,
            ),

            ffn_d_hidden=adapter.params.get(
                "ffn_d_hidden",
                128,
            ),

            # Required by rtdl_revisiting_models 0.0.2.
            ffn_d_hidden_multiplier=None,

            ffn_dropout=adapter.params.get(
                "ffn_dropout",
                0.1,
            ),

            residual_dropout=adapter.params.get(
                "residual_dropout",
                0.0,
            ),
        )

        model.load_state_dict(
            torch.load(
                model_weights_file,
                map_location="cpu",
            )
        )

        adapter._model = model
        adapter._fitted = True

        return adapter

    # ============================================================
    # Metadata
    # ============================================================

    def get_metadata(
        self,
    ) -> dict[str, Any]:
        """Return configuration and training diagnostics."""

        return {
            "name": self.name,
            "model_id": self.model_id,
            "fitted": self._fitted,

            "params": self.params,

            "categorical_features": (
                self.categorical_features
            ),

            "feature_names": (
                self._feature_names
            ),

            "class_weighting": (
                "inner_training_fold_positive_class_weight"
                if self._pos_weight is not None
                else "not_yet_fitted"
            ),

            "pos_weight": (
                self._pos_weight
            ),

            "best_epoch": (
                self._best_epoch
            ),

            "best_inner_pr_auc": (
                self._best_inner_pr_auc
            ),

            "epochs_trained": (
                self._epochs_trained
            ),

            "inner_train_size": (
                self._inner_train_size
            ),

            "inner_validation_size": (
                self._inner_validation_size
            ),
        }