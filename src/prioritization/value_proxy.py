"""Development-only synthetic retention-value proxy utilities."""
from __future__ import annotations
import pandas as pd


def minmax(series: pd.Series) -> pd.Series:
    lo=float(series.min()); hi=float(series.max())
    if hi==lo: return pd.Series(0.0,index=series.index)
    return (series.astype(float)-lo)/(hi-lo)


def build_value_proxy(df: pd.DataFrame) -> pd.Series:
    required=['Performance_Score','Monthly_Salary','Years_At_Company']
    if missing:=set(required)-set(df.columns): raise ValueError(f"Missing columns: {sorted(missing)}")
    return (minmax(df[required[0]])+minmax(df[required[1]])+minmax(df[required[2]]))/3.0
