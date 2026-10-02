# Main merge preservation record

The architecture refactor was merged with main commit `5861c9e`. Active notebooks use the new group/team-contribution layout. Superseded original notebook and guide bytes are retained under `docs/archive/pre_main_merge/` when they do not exist unchanged elsewhere. Historical result filenames and `AGENTS.md` remain available. Modified originals also remain in the merged Git history.

The raw dataset and frozen split registry are unchanged. Archived notebooks are historical references, not active workflows.

| Original main path removed or relocated | Original content retained at |
|---|---|
| `AGENTS.md` | `AGENTS.md` |
| `EVALUATION_GUIDE.md` | `docs/archive/pre_main_merge/EVALUATION_GUIDE.md` |
| `models/ebm/.gitkeep` | `models/ebm/.gitkeep` |
| `models/ft_transformer/.gitkeep` | `models/ft_transformer/.gitkeep` |
| `models/lightgbm/.gitkeep` | `models/lightgbm/.gitkeep` |
| `models/preprocessing/.gitkeep` | `models/preprocessing/.gitkeep` |
| `models/tabnet/.gitkeep` | `models/tabnet/.gitkeep` |
| `models/xgboost/.gitkeep` | `models/xgboost/.gitkeep` |
| `notebooks/05_member_1/05_m1_ft_transformer.ipynb` | `docs/archive/pre_main_merge/notebooks/05_member_1/05_m1_ft_transformer.ipynb` |
| `notebooks/06_member_2/06_m2_catboost.ipynb` | `docs/archive/pre_main_merge/notebooks/06_member_2/06_m2_catboost.ipynb` |
| `notebooks/07_member_3/07_m3_xgboost.ipynb` | `docs/archive/pre_main_merge/notebooks/07_member_3/07_m3_xgboost.ipynb` |
| `notebooks/08_member_4/08_m4_lightgbm.ipynb` | `docs/archive/pre_main_merge/notebooks/08_member_4/08_m4_lightgbm.ipynb` |
| `notebooks/09_member_5/09_m5_ebm.ipynb` | `notebooks/team_contributions/models/member_5_ebm/09_m5_ebm.ipynb` |
| `notebooks/09_member_5/09_m5_ebm_historical.ipynb` | `docs/archive/pre_main_merge/notebooks/09_member_5/09_m5_ebm_historical.ipynb` |
| `notebooks/10_member_6/10_m6_tabnet.ipynb` | `notebooks/team_contributions/models/member_6_tabnet/10_m6_tabnet.ipynb` |
| `notebooks/11_model_comparison.ipynb` | `docs/archive/pre_main_merge/notebooks/11_model_comparison.ipynb` |
| `notebooks/12_calibration.ipynb` | `docs/archive/pre_main_merge/notebooks/12_calibration.ipynb` |
| `notebooks/13_fairness_analysis.ipynb` | `docs/archive/pre_main_merge/notebooks/13_fairness_analysis.ipynb` |
| `notebooks/14_retention_prioritization.ipynb` | `docs/archive/pre_main_merge/notebooks/14_retention_prioritization.ipynb` |
| `notebooks/15_final_holdout_evaluation.ipynb` | `docs/archive/pre_main_merge/notebooks/15_final_holdout_evaluation.ipynb` |
| `notebooks/group_pipeline.ipynb` | `docs/archive/pre_main_merge/notebooks/group_pipeline.ipynb` |
| `notebooks/member_contributions/IT24100879_Outlier_Analysis.ipynb` | `docs/archive/pre_main_merge/notebooks/member_contributions/IT24100879_Outlier_Analysis.ipynb` |
| `notebooks/member_contributions/IT25100142_PCA.ipynb` | `notebooks/team_contributions/preprocessing/IT25100142_PCA.ipynb` |
| `notebooks/member_contributions/IT25100385_Categorical_Encoding.ipynb` | `notebooks/team_contributions/preprocessing/IT25100385_Categorical_Encoding.ipynb` |
| `notebooks/member_contributions/IT25100429_Standardization.ipynb` | `docs/archive/pre_main_merge/notebooks/member_contributions/IT25100429_Standardization.ipynb` |
| `notebooks/member_contributions/IT25100445_Feature_Selection.ipynb` | `docs/archive/pre_main_merge/notebooks/member_contributions/IT25100445_Feature_Selection.ipynb` |
| `notebooks/member_contributions/IT25100565_Data_Quality.ipynb` | `docs/archive/pre_main_merge/notebooks/member_contributions/IT25100565_Data_Quality.ipynb` |
| `notebooks/member_contributions/outputs/monthly_salary_before_after_scaling.png` | `docs/archive/pre_main_merge/notebooks/member_contributions/outputs/monthly_salary_before_after_scaling.png` |
| `reports/.gitkeep` | `reports/.gitkeep` |
| `results/calibration/.gitkeep` | `results/calibration/.gitkeep` |
| `results/comparison/.gitkeep` | `results/comparison/.gitkeep` |
| `results/fairness/.gitkeep` | `results/fairness/.gitkeep` |
| `results/models/catboost/catboost_fold_metrics.csv` | `results/models/catboost/catboost_fold_metrics.csv` |
| `results/models/catboost/catboost_oof_predictions.csv` | `results/models/catboost/catboost_oof_predictions.csv` |
| `results/models/ebm/.gitkeep` | `results/models/ebm/.gitkeep` |
| `results/models/ebm/ebm_fold_metrics.csv` | `results/models/ebm/ebm_fold_metrics.csv` |
| `results/models/ebm/ebm_oof_predictions.csv` | `results/models/ebm/ebm_oof_predictions.csv` |
| `results/models/ebm/ebm_summary_metrics.json` | `results/models/ebm/ebm_summary_metrics.json` |
| `results/models/ft_transformer/.gitkeep` | `results/models/ft_transformer/.gitkeep` |
| `results/models/ft_transformer/ft_transformer_oof_predictions.csv` | `results/models/ft_transformer/ft_transformer_oof_predictions.csv` |
| `results/models/lightgbm/.gitkeep` | `results/models/lightgbm/.gitkeep` |
| `results/models/tabnet/.gitkeep` | `results/models/tabnet/.gitkeep` |
| `results/models/xgboost/.gitkeep` | `results/models/xgboost/.gitkeep` |
| `results/models/xgboost/confusion_matrix_xgboost.png` | `results/models/xgboost/confusion_matrix_xgboost.png` |
| `results/models/xgboost/feature_importances.csv` | `results/models/xgboost/feature_importances.csv` |
| `results/models/xgboost/pr_curve_xgboost.png` | `results/models/xgboost/pr_curve_xgboost.png` |
| `results/models/xgboost/xgboost_fold_metrics.csv` | `results/models/xgboost/xgboost_fold_metrics.csv` |
| `results/models/xgboost/xgboost_oof_predictions.csv` | `results/models/xgboost/xgboost_oof_predictions.csv` |
| `results/models/xgboost/xgboost_summary_metrics.json` | `results/models/xgboost/xgboost_summary_metrics.json` |
| `results/pca/.gitkeep` | `results/pca/.gitkeep` |
| `results/prioritization/.gitkeep` | `results/prioritization/.gitkeep` |

## Remote branch historical files

Historical files from remote branch commit `3b6eca1` that earlier local work removed are also preserved.

| Original remote branch path | Original content retained at |
|---|---|
| `UPGRADE_CHANGELOG.md` | `docs/archive/remote_branch_before_merge/UPGRADE_CHANGELOG.md` |
| `docs/MIGRATION_NOTES.md` | `docs/archive/remote_branch_before_merge/docs/MIGRATION_NOTES.md` |
| `docs/REFACTOR_VALIDATION.md` | `docs/archive/remote_branch_before_merge/docs/REFACTOR_VALIDATION.md` |
| `docs/archive/EVALUATION_GUIDE_pre_refactor.md` | `docs/archive/pre_main_merge/EVALUATION_GUIDE.md` |
| `docs/archive/EVALUATION_GUIDE_pre_upgrade.md` | `docs/archive/remote_branch_before_merge/docs/archive/EVALUATION_GUIDE_pre_upgrade.md` |
| `docs/archive/README_pre_refactor.md` | `docs/archive/remote_branch_before_merge/docs/archive/README_pre_refactor.md` |
| `docs/internal_review/KNOWN_ISSUES.md` | `docs/archive/remote_branch_before_merge/docs/internal_review/KNOWN_ISSUES.md` |
| `models/catboost/.gitkeep` | `models/ebm/.gitkeep` |
| `notebooks/team_contributions/preprocessing/outputs/monthly_salary_before_after_scaling.png` | `docs/archive/pre_main_merge/notebooks/member_contributions/outputs/monthly_salary_before_after_scaling.png` |

## Validation

The complete merged test suite passed: **295 tests**. Every preservation mapping was verified against its original Git blob. The raw dataset and frozen split registry match main exactly. The four group-analysis notebooks were previously executed with the same formal EBM inputs; full 60-fit EBM retraining was not repeated for this merge.
