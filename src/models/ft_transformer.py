"""FT-Transformer Model Adapter for Employee Resignation Prediction (Member 1).

Conforms to the ModelAdapter protocol using PyTorch and RTDL Feature Tokenizer
Transformer architecture with fold-local scaling and embeddings.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelAdapter


class FTTransformerAdapter:
    """FT-Transformer model adapter with fold-local preprocessing."""

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
        }
        self.categorical_features = categorical_features or [
            "Department",
            "Gender",
            "Job_Title",
            "Education_Level",
        ]
        self._model: Any = None
        self._scaler: Any = None
        self._category_maps: dict[str, dict[str, int]] = {}
        self._feature_names: list[str] = []
        self._num_features: list[str] = []
        self._cat_cardinalities: list[int] = []
        self._fitted = False

    def _preprocess(
        self,
        X: pd.DataFrame,
        fit: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Preprocess inputs into scaled continuous matrix and categorical integer matrix."""
        from sklearn.preprocessing import StandardScaler

        X_proc = X.copy()
        self._num_features = [
            c for c in self._feature_names if c not in self.categorical_features
        ]

        if fit:
            self._scaler = StandardScaler()
            if self._num_features:
                self._scaler.fit(X_proc[self._num_features])

            self._category_maps = {}
            self._cat_cardinalities = []

            for col in self.categorical_features:
                if col in X_proc.columns:
                    unique_vals = sorted(X_proc[col].dropna().unique())
                    mapping = {
                        val: idx
                        for idx, val in enumerate(unique_vals)
                    }

                    self._category_maps[col] = mapping
                    self._cat_cardinalities.append(len(mapping))

        x_num = None

        if self._num_features and self._scaler is not None:
            x_num = (
                self._scaler
                .transform(X_proc[self._num_features])
                .astype(np.float32)
            )

        x_cat = None

        if self.categorical_features:
            cat_arrays = []

            for col in self.categorical_features:
                if col in X_proc.columns:
                    mapping = self._category_maps.get(col, {})

                    arr = (
                        X_proc[col]
                        .map(mapping)
                        .fillna(0)
                        .values
                        .astype(np.int64)
                    )

                    cat_arrays.append(arr)

            if cat_arrays:
                x_cat = np.column_stack(cat_arrays)

        return x_num, x_cat

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        epochs: int = 50,
        batch_size: int = 512,
    ) -> FTTransformerAdapter:
        """Fit fold-local preprocessors and FT-Transformer network on training fold data."""

        try:
            import torch
            import torch.nn as nn

            from rtdl_revisiting_models import FTTransformer
            from torch.utils.data import DataLoader, TensorDataset

        except ImportError as exc:
            raise ImportError(
                "torch and rtdl-revisiting-models are required "
                "for FTTransformerAdapter."
            ) from exc

        self._feature_names = list(X.columns)

        x_num, x_cat = self._preprocess(
            X,
            fit=True,
        )

        y_arr = np.asarray(
            y,
            dtype=np.float32,
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Set seed
        torch.manual_seed(
            self.params.get("seed", 42)
        )

        n_cont_features = (
            x_num.shape[1]
            if x_num is not None
            else 0
        )

        cat_cardinalities = (
            self._cat_cardinalities
            if x_cat is not None
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
            # We already specify ffn_d_hidden directly above,
            # so the multiplier is intentionally disabled.
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
                y_arr,
                dtype=torch.float32,
            ).unsqueeze(1)
        )

        dataset = TensorDataset(*tensors)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.params.get(
                "learning_rate",
                0.001,
            ),
            weight_decay=self.params.get(
                "weight_decay",
                1e-5,
            ),
        )

        criterion = nn.BCEWithLogitsLoss()

        self._model.train()

        for _ in range(epochs):

            for batch in loader:

                optimizer.zero_grad()

                if x_num is not None and x_cat is not None:

                    b_num = batch[0].to(device)
                    b_cat = batch[1].to(device)
                    b_y = batch[2].to(device)

                    out = self._model(
                        b_num,
                        b_cat,
                    )

                elif x_num is not None:

                    b_num = batch[0].to(device)
                    b_y = batch[1].to(device)

                    out = self._model(
                        b_num,
                        None,
                    )

                else:

                    b_cat = batch[0].to(device)
                    b_y = batch[1].to(device)

                    out = self._model(
                        None,
                        b_cat,
                    )

                loss = criterion(
                    out,
                    b_y,
                )

                loss.backward()
                optimizer.step()

        self._fitted = True

        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Predict positive class probabilities for input features."""

        if not self._fitted or self._model is None:
            raise RuntimeError(
                "Adapter is not fitted. Call fit() first."
            )

        import torch

        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
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

        dataset = TensorDataset(*tensors)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        probs_list = []

        with torch.no_grad():

            for batch in loader:

                if x_num is not None and x_cat is not None:

                    b_num = batch[0].to(device)
                    b_cat = batch[1].to(device)

                    logits = self._model(
                        b_num,
                        b_cat,
                    )

                elif x_num is not None:

                    b_num = batch[0].to(device)

                    logits = self._model(
                        b_num,
                        None,
                    )

                else:

                    b_cat = batch[0].to(device)

                    logits = self._model(
                        None,
                        b_cat,
                    )

                probs = (
                    torch.sigmoid(logits)
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )

                probs_list.append(probs)

        return np.concatenate(
            probs_list
        ).astype(float)

    def save(
        self,
        directory: Path,
    ) -> dict[str, Any]:
        """Serialize FT-Transformer model parameters and preprocessors."""

        import torch

        directory = Path(directory)

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
        ) as f:

            pickle.dump(
                {
                    "scaler": self._scaler,
                    "category_maps": self._category_maps,
                    "cat_cardinalities": self._cat_cardinalities,
                    "params": self.params,
                    "categorical_features": self.categorical_features,
                    "feature_names": self._feature_names,
                    "num_features": self._num_features,
                },
                f,
            )

        metadata = self.get_metadata()

        metadata["weights_file"] = (
            str(model_weights_file.name)
        )

        metadata["preprocessor_file"] = (
            str(preprocessor_file.name)
        )

        return metadata

    @classmethod
    def load(
        cls,
        directory: Path,
    ) -> FTTransformerAdapter:
        """Load serialized FT-Transformer adapter from disk."""

        import torch

        from rtdl_revisiting_models import FTTransformer

        directory = Path(directory)

        preprocessor_file = (
            directory
            / "ft_transformer_preprocessor.pkl"
        )

        model_weights_file = (
            directory
            / "ft_transformer_weights.pt"
        )

        adapter = cls()

        if (
            preprocessor_file.exists()
            and model_weights_file.exists()
        ):

            with open(
                preprocessor_file,
                "rb",
            ) as f:
                state = pickle.load(f)

            adapter._scaler = state["scaler"]

            adapter._category_maps = (
                state["category_maps"]
            )

            adapter._cat_cardinalities = (
                state["cat_cardinalities"]
            )

            adapter.params = state["params"]

            adapter.categorical_features = (
                state["categorical_features"]
            )

            adapter._feature_names = (
                state["feature_names"]
            )

            adapter._num_features = (
                state["num_features"]
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

    def get_metadata(
        self,
    ) -> dict[str, Any]:
        """Return model metadata, hyperparameter settings, and feature names."""

        return {
            "name": self.name,
            "model_id": self.model_id,
            "fitted": self._fitted,
            "params": self.params,
            "categorical_features": self.categorical_features,
            "feature_names": self._feature_names,
        }