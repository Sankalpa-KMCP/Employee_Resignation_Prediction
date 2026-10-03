"""Ranking metrics for the academic retention-prioritization simulation.

Used for the exploratory group simulation, not an operational employee decision policy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def select_top_k(df: pd.DataFrame, score_col: str, capacity: float) -> pd.DataFrame:
    """Select round(n * capacity) rows; break score ties by ID, then input order.

    Identifiers are only tie breakers, never predictive features.
    """
    if not np.isfinite(capacity) or not 0 < capacity < 1:
        raise ValueError("capacity must be between 0 and 1")
    if df.empty or score_col not in df:
        raise ValueError("A non-empty frame and score column are required")
    scores = df[score_col].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite")
    columns, ascending = [score_col], [False]
    if "Employee_ID" in df:
        if df.Employee_ID.isna().any() or not df.Employee_ID.is_unique:
            raise ValueError("Employee_ID must be present and unique")
        columns.append("Employee_ID")
        ascending.append(True)
    k = max(1, int(round(len(df) * capacity)))
    # Sort numeric values even if a caller supplied numeric strings.
    sortable = df.assign(**{score_col: scores})
    return sortable.sort_values(columns, ascending=ascending, kind="stable").head(k)


def evaluate_ranking(df: pd.DataFrame, score_col: str, capacity: float,
                     value_col: str | None = None) -> dict:
    """Summarize positive outcomes captured by the highest-scoring capacity fraction.

    Return precision, recall and lift relative to supplied target prevalence.
    These are retrospective diagnostics, not evidence that a retention intervention works.
    """
    if 'y_true' not in df or not df.y_true.isin([0, 1]).all():
        raise ValueError("y_true must contain binary labels")
    selected = select_top_k(df, score_col, capacity)
    k = len(selected)
    # Observed resignations measure retrospective capture, not preventable resignations.
    positives=float(df.y_true.sum()); captured=float(selected.y_true.sum())
    prevalence=float(df.y_true.mean())
    precision=captured/k; recall=captured/positives if positives else np.nan
    result = {'capacity':capacity,'selected_count':k,'captured_positive_count':int(captured),'precision_at_capacity':precision,'recall_at_capacity':recall,'lift_at_capacity':precision/prevalence if prevalence else np.nan}
    if value_col is not None:
        values = df[value_col].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Proxy values must be finite and non-negative")
        total_positive_value = float((df.y_true * df[value_col]).sum())
        captured_value = float((selected.y_true * selected[value_col]).sum())
        total_value = float(df[value_col].sum())
        result.update({
            "captured_positive_proxy_value": captured_value,
            "value_weighted_recall_at_capacity": captured_value / total_positive_value if total_positive_value else np.nan,
            "selected_proxy_value_share": float(selected[value_col].sum()) / total_value if total_value else np.nan,
        })
    return result
