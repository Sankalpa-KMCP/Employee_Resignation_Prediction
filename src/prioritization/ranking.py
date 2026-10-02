"""Ranking metrics for the academic retention-prioritization simulation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def evaluate_ranking(df: pd.DataFrame, score_col: str, capacity: float) -> dict:
    if not 0<capacity<1: raise ValueError("capacity must be between 0 and 1")
    if score_col not in df or 'y_true' not in df: raise ValueError("Required columns are missing")
    k=max(1,int(round(len(df)*capacity))); selected=df.nlargest(k,score_col)
    positives=float(df.y_true.sum()); captured=float(selected.y_true.sum())
    prevalence=float(df.y_true.mean())
    precision=captured/k; recall=captured/positives if positives else 0.0
    return {'capacity':capacity,'selected_count':k,'captured_positive_count':int(captured),'precision_at_capacity':precision,'recall_at_capacity':recall,'lift_at_capacity':precision/prevalence if prevalence else np.nan}
