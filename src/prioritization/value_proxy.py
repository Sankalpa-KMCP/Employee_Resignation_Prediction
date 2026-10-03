"""Development-only synthetic retention-value proxy utilities.

The group simulation uses these synthetic scores only to illustrate a ranking exercise.
"""
from __future__ import annotations
import pandas as pd


def minmax(series: pd.Series) -> pd.Series:
    """Scale a series to its observed [0, 1] range; return zeros when the range is constant."""
    lo=float(series.min()); hi=float(series.max())
    if hi==lo: return pd.Series(0.0,index=series.index)
    return (series.astype(float)-lo)/(hi-lo)


def build_value_proxy(df: pd.DataFrame) -> pd.Series:
    """Average normalized performance, salary and tenure as an academic ranking proxy.

    Return one synthetic score per row; it is not measured economic value or intervention benefit.
    """
    required=['Performance_Score','Monthly_Salary','Years_At_Company']
    if missing:=set(required)-set(df.columns): raise ValueError(f"Missing columns: {sorted(missing)}")
    # Equal weighting is an illustrative design choice, not a learned business valuation.
    return (minmax(df[required[0]])+minmax(df[required[1]])+minmax(df[required[2]]))/3.0
