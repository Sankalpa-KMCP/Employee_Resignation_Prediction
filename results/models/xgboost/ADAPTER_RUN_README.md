# XGBoost adapter and notebook evaluation

The canonical `oof_predictions.csv` in this directory contains Member 3's P1
one-hot, eight-candidate strict nested evaluation. `configs/analysis_registry.yaml`
selects this file for the current group comparison, calibration, fairness and
prioritization workflows. P2/P3 preprocessing variants are descriptive sensitivity
experiments and do not replace this primary pipeline.

The separate `strict_nested/` directory contains an earlier independent script
run. It is retained as historical evidence and is not the current group input.
The `exploratory_fixed_param_run/` directory also retains a distinct historical
experiment. Do not mix predictions or metrics from these runs.

The notebook was executed in a fresh Windows kernel after adapting its path.
Consult run_metadata.json for the execution environment and exact provenance.
Earlier Linux results differed from Windows results; thread count affects
selection and predictions in observed experiments. A fixed seed alone does not
establish reproducibility across environments. No formal holdout rows are used.

## Historical validation record

The counts below describe earlier adapter integration, not the current checkout's
test status. Use the current test suite and canonical OOF checks for verification.

- XGBoost adapter: 53 tests passed in the Windows Anaconda environment.
- Data, model configuration and notebook structure: 68 tests passed.
- Full suite: initially 264 passed, 22 failed and 1 skipped. Twenty-one
  failures required missing CatBoost/Torch dependencies; the other was a
  copied input's CRLF bytes. Restoring canonical LF bytes made all data
  validation tests pass. No new dependencies were installed.
- The original working folder and a separate backup retain the earlier work.
