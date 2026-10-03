"""Shared plotting helpers for aggregate evaluation figures.

Evaluation notebooks can use this helper to export aggregate figures.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def save_metric_bar(df: pd.DataFrame, metric: str, output: str|Path, reference: float|None=None, title: str|None=None) -> Path:
    """Save a model metric bar chart and return its output path.

    The dataframe needs model and metric columns; reference adds an optional baseline line.
    """
    if "model" not in df or metric not in df: raise ValueError("Dataframe must contain model and requested metric columns.")
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(9,5)); plt.bar(df["model"],df[metric])
    if reference is not None: plt.axhline(reference,linestyle="--")
    plt.xticks(rotation=35,ha="right"); plt.ylabel(metric.replace("_"," ").title()); plt.title(title or metric.replace("_"," ").title()); plt.tight_layout(); plt.savefig(p,dpi=160); plt.close(); return p
