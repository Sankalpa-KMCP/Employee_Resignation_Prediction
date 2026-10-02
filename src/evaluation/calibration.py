"""Development-only calibration diagnostics for binary classifiers."""
from __future__ import annotations
import numpy as np
import pandas as pd


def reliability_table(y_true, y_probability, n_bins: int = 10) -> pd.DataFrame:
    y=np.asarray(y_true,dtype=int); p=np.asarray(y_probability,dtype=float)
    if y.ndim!=1 or p.ndim!=1 or len(y)!=len(p) or len(y)==0:
        raise ValueError("y_true and y_probability must be non-empty 1D arrays of equal length.")
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if n_bins < 2: raise ValueError("n_bins must be at least 2.")
    edges=np.linspace(0,1,n_bins+1)
    bins=np.clip(np.digitize(p,edges[1:-1],right=False),0,n_bins-1)
    rows=[]
    for b in range(n_bins):
        m=bins==b
        if not m.any():
            rows.append((b,edges[b],edges[b+1],0,np.nan,np.nan,0.0))
        else:
            rows.append((b,edges[b],edges[b+1],int(m.sum()),float(p[m].mean()),float(y[m].mean()),float(m.mean())))
    return pd.DataFrame(rows,columns=['bin','lower','upper','count','mean_probability','observed_rate','fraction'])


def expected_calibration_error(y_true, y_probability, n_bins: int = 10) -> float:
    table=reliability_table(y_true,y_probability,n_bins)
    valid=table['count']>0
    return float((table.loc[valid,'fraction']*(table.loc[valid,'mean_probability']-table.loc[valid,'observed_rate']).abs()).sum())
