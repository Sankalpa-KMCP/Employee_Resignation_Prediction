"""Helpers for the frozen development/holdout split registry."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

DEFAULT_REGISTRY=Path(__file__).resolve().parents[2]/"data"/"processed"/"split_registry.csv"

def load_split_registry(path: str|Path=DEFAULT_REGISTRY) -> pd.DataFrame:
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    df=pd.read_csv(p)
    required={"Employee_ID","partition","cv_fold"}
    if missing:=required-set(df.columns): raise ValueError(f"Split registry missing columns: {sorted(missing)}")
    if df.Employee_ID.isna().any() or df.Employee_ID.duplicated().any(): raise ValueError("Employee_ID must be complete and unique in split registry.")
    if not set(df.partition.unique()).issubset({"development","holdout"}): raise ValueError("Unexpected partition label.")
    return df

def development_registry(df: pd.DataFrame|None=None) -> pd.DataFrame:
    r=load_split_registry() if df is None else df
    out=r[r.partition=="development"].copy()
    if out.cv_fold.isna().any(): raise ValueError("Development rows require cv_fold values.")
    return out

def holdout_registry(df: pd.DataFrame|None=None) -> pd.DataFrame:
    r=load_split_registry() if df is None else df
    return r[r.partition=="holdout"].copy()
