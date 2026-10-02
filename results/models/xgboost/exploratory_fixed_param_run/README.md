# XGBoost exploratory fixed-parameter run (SUPERSEDED)

Historical evidence only. Do not use these files for model comparison.

- Produced by the earlier version of `07_m3_xgboost.ipynb` with one fixed parameter set for every fold
  (n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8).
- `scale_pos_weight` (8.99) was computed from all 80,000 development rows, including each fold's
  outer-validation rows, so this run does not meet the strict nested protocol (Project Context v4.6.2, Section 8.16).
- The environment was not recorded; the fold-1 result did not reproduce under the pinned
  requirements (AP 0.0971 vs 0.0989 saved).
- Reported pooled AP 0.1026, ROC-AUC 0.5061 (development OOF).

Superseded by the strict nested evaluation in `results/models/xgboost/` (see `run_metadata.json`).
