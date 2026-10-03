"""Development-only calibration diagnostics for binary classifiers.

Group diagnostics use saved probabilities; these helpers do not train calibration models.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def reliability_table(y_true, y_probability, n_bins: int = 10) -> pd.DataFrame:
    """Group paired labels and probabilities into equal-width bins on [0, 1].

    Return counts, mean scores, observed positive rates and bin fractions.
    Empty bins keep NaN means so they cannot suggest evidence of calibration.
    """
    y=np.asarray(y_true,dtype=int); p=np.asarray(y_probability,dtype=float)
    if y.ndim!=1 or p.ndim!=1 or len(y)!=len(p) or len(y)==0:
        raise ValueError("y_true and y_probability must be non-empty 1D arrays of equal length.")
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if n_bins < 2: raise ValueError("n_bins must be at least 2.")
    # Equal-width bins summarize reliability without optimizing bin boundaries on outcomes.
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
    """Return the bin-frequency-weighted absolute gap between mean probability and observed rate.

    This descriptive score depends on the chosen bin count; it does not fit a calibrator.
    """
    table=reliability_table(y_true,y_probability,n_bins)
    # Empty bins have undefined means and contribute no weight to the reported error.
    valid=table['count']>0
    return float((table.loc[valid,'fraction']*(table.loc[valid,'mean_probability']-table.loc[valid,'observed_rate']).abs()).sum())
