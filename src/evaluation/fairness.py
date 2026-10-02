"""Aggregate subgroup diagnostics for the academic fairness analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix


def subgroup_classification_metrics(df: pd.DataFrame, group_col: str, threshold: float=0.5, min_count: int=30) -> pd.DataFrame:
    required={group_col,'y_true','y_probability'}
    if missing:=required-set(df.columns): raise ValueError(f"Missing columns: {sorted(missing)}")
    rows=[]
    for group,g in df.groupby(group_col,dropna=False):
        y=g.y_true.astype(int).to_numpy(); p=g.y_probability.astype(float).to_numpy(); pred=(p>=threshold).astype(int)
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        stable=len(g)>=min_count and len(np.unique(y))==2
        rows.append({group_col:group,'n':len(g),'prevalence':float(y.mean()),'average_precision':float(average_precision_score(y,p)) if stable else np.nan,'roc_auc':float(roc_auc_score(y,p)) if stable else np.nan,'precision':float(tp/(tp+fp)) if tp+fp else 0.0,'recall':float(tp/(tp+fn)) if tp+fn else 0.0,'false_positive_rate':float(fp/(fp+tn)) if fp+tn else 0.0,'predicted_positive_rate':float(pred.mean()),'stable_rate_estimates':bool(stable)})
    return pd.DataFrame(rows)


def top_k_representation(df: pd.DataFrame, group_col: str, capacity: float=0.20) -> pd.DataFrame:
    if not 0<capacity<1: raise ValueError("capacity must be between 0 and 1")
    n=max(1,int(round(len(df)*capacity)))
    selected=df.nlargest(n,'y_probability')
    pop=df[group_col].value_counts(dropna=False); sel=selected[group_col].value_counts(dropna=False)
    rows=[]
    for group,count in pop.items():
        pop_share=count/len(df); sel_count=int(sel.get(group,0)); sel_share=sel_count/len(selected)
        rows.append({group_col:group,'population_count':int(count),'selected_count':sel_count,'population_share':float(pop_share),'selected_share':float(sel_share),'representation_ratio':float(sel_share/pop_share) if pop_share else np.nan})
    return pd.DataFrame(rows)
