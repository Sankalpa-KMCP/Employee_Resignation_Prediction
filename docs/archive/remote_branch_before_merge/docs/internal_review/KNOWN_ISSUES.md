# Known Issues from External Review

This file is an internal cleanup checklist, not an assignment deliverable.

1. Final group report is not yet present.
2. Six-model group comparison notebook is still incomplete/scaffolded.
3. Fairness-analysis notebook is still incomplete/scaffolded.
4. XGBoost has five-fold OOF results but lacks genuine multi-configuration tuning evidence.
5. EBM has five-fold OOF results but lacks genuine multi-configuration tuning evidence.
6. The current FT-Transformer notebook/config state requires consistency cleanup before clean rerun.
7. The LightGBM `fit()` API mismatch (`eval_X`/`eval_y`) found in the original review ZIP was repaired to use `eval_set` in this refactored copy; focused LightGBM tests pass. Stored historical LightGBM results were not regenerated.
8. Some historical README/context documents contain stale implementation-status text; current operational status is summarized in the root README.
9. The outlier contribution notebook still depends on Kaggle download logic rather than the bundled local raw CSV.
10. Final holdout must remain locked until development decisions are frozen.
