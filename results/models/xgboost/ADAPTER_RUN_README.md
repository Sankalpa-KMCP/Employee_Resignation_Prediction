# XGBoost adapter and notebook evaluation

The results in this directory are produced by the member 3 adapter notebook.
The separate `strict_nested/` directory contains the branch's earlier script run;
the analysis registry and existing group outputs continue to refer to that run.
Do not mix the two runs or their metrics. Preprocessing variants are sensitivity
experiments and do not replace the primary one-hot pipeline.

The notebook was executed in a fresh Windows kernel after adapting its path.
Consult run_metadata.json for the execution environment and exact provenance.
Earlier Linux results differed from Windows results; thread count affects
selection and predictions in observed experiments. A fixed seed alone does not
establish reproducibility across environments. No formal holdout rows are used.

## Validation

- XGBoost adapter: 53 tests passed in the Windows Anaconda environment.
- Data, model configuration and notebook structure: 68 tests passed.
- Full suite: initially 264 passed, 22 failed and 1 skipped. Twenty-one
  failures required missing CatBoost/Torch dependencies; the other was a
  copied input's CRLF bytes. Restoring canonical LF bytes made all data
  validation tests pass. No new dependencies were installed.
- The original working folder and a separate backup retain the earlier work.
