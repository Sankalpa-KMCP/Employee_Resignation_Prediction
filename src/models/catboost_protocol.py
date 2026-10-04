"""Thin helper that runs the CatBoost nested-CV protocol for audits and experiments.

It always goes through ``run_5fold_cv`` (fresh adapter per fold, holdout guard,
OOF validation), so experiments cannot bypass the leakage safeguards.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.catboost_model import CatBoostAdapter
from src.models.config import load_model_config
from src.models.runner import run_5fold_cv


@dataclass
class ProtocolRun:
    """Outputs of one nested-CV run."""

    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    adapters: list[CatBoostAdapter] = field(default_factory=list)
    fold_metadata: list[dict[str, Any]] = field(default_factory=list)
    fit_seconds: float = 0.0


def build_config(
    config_name: str = "catboost",
    candidates: Sequence[Mapping[str, Any]] | None = None,
    fixed_overrides: Mapping[str, Any] | None = None,
    inner_overrides: Mapping[str, Any] | None = None,
    ensemble: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the repo config and apply experiment overrides to a deep copy."""
    cfg = copy.deepcopy(load_model_config(config_name, strict_schema=True))
    if candidates is not None:
        cfg["candidates"] = [copy.deepcopy(dict(c)) for c in candidates]
    cfg.setdefault("fixed_params", {}).update(dict(fixed_overrides or {}))
    cfg.setdefault("inner_validation", {}).update(dict(inner_overrides or {}))
    if ensemble:
        cfg["ensemble"] = dict(ensemble)
    return cfg


def run_protocol(
    data: pd.DataFrame,
    split: pd.DataFrame,
    features: Sequence[str],
    config: Mapping[str, Any],
    target_override: np.ndarray | None = None,
) -> ProtocolRun:
    """Run strict nested CV; ``target_override`` swaps in permuted labels for shuffle tests.

    The override replaces labels row-for-row on the development frame only; the
    runner still validates the OOF frame against that same (permuted) source.
    """
    frame = data.drop(columns=["cv_fold"], errors="ignore").copy()
    if target_override is not None:
        frame["Resigned"] = np.asarray(target_override, dtype=int)
    captured: list[CatBoostAdapter] = []

    def factory() -> CatBoostAdapter:
        adapter = CatBoostAdapter(config=config, candidate_id="auto")
        captured.append(adapter)
        return adapter

    result = run_5fold_cv(factory, frame, split, list(features), expected_model_name="CatBoost Classifier")
    return ProtocolRun(
        oof=result.predictions.copy(),
        fold_metrics=result.fold_metrics.copy(),
        adapters=captured,
        fold_metadata=copy.deepcopy(result.metadata["fold_metadata"]),
        fit_seconds=float(result.fit_time_seconds_total),
    )
