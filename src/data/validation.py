"""Dataset integrity checks used by notebooks and QA.

Audit notebooks can call these checks before deriving model inputs.
"""
from __future__ import annotations
import pandas as pd

EXPECTED_COLUMNS=["Employee_ID","Department","Gender","Age","Job_Title","Hire_Date","Years_At_Company","Education_Level","Performance_Score","Monthly_Salary","Work_Hours_Per_Week","Projects_Handled","Overtime_Hours","Sick_Days","Remote_Work_Frequency","Team_Size","Training_Hours","Promotions","Employee_Satisfaction_Score","Resigned"]

def validate_raw_dataset(df: pd.DataFrame, expected_rows: int=100_000) -> dict:
    """Check the supplied dataframe against the canonical schema, row count and binary target.

    Return an audit summary including duplicate count and resignation prevalence.
    Duplicate rows are reported; the explicit rejection checks concern IDs and missing values.
    """
    if list(df.columns)!=EXPECTED_COLUMNS: raise ValueError("Raw dataset schema or column order differs from the canonical contract.")
    if len(df)!=expected_rows: raise ValueError(f"Expected {expected_rows} rows, found {len(df)}.")
    if df.Employee_ID.isna().any() or df.Employee_ID.duplicated().any(): raise ValueError("Employee_ID must be complete and unique.")
    if int(df.isna().sum().sum())!=0: raise ValueError("Canonical raw dataset is expected to contain zero missing values.")
    target=set(df.Resigned.dropna().unique())
    if not target.issubset({True,False,0,1}): raise ValueError("Resigned must be binary.")
    return {"rows":len(df),"columns":len(df.columns),"missing_values":0,"duplicate_rows":int(df.duplicated().sum()),"positive_rate":float(df.Resigned.astype(int).mean())}
