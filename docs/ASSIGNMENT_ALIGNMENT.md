# IT2011 Final-Evaluation Alignment

| Requirement | Evidence in upgraded repository | Status |
|---|---|---|
| One model per student | Six model notebooks under `notebooks/team_contributions/models/` | Evidenced |
| Model suitability / implementation explanation | Model notebooks + configs + `src/models/` | Evidenced |
| Hyperparameter tuning / varieties | CatBoost, XGBoost, LightGBM, TabNet have candidate-selection evidence | Partial across all six |
| Different preprocessing / hyperparameter variants | Model-specific adapters/configs and tuning artifacts | Evidenced, except EBM final tuning and FT final strict-nested rerun |
| Relevant metrics | AP, ROC-AUC, Brier, threshold metrics, fold stability | Evidenced |
| Validation method | Frozen 80/20 split and five development CV folds | Evidenced |
| Compare six individual models | `notebooks/group/02_model_comparison.ipynb`, `results/comparison/` | Completed on development OOF |
| Discuss challenges / expected behavior | Report draft, comparison conclusion, weak-signal analysis | Evidenced |
| Ethics / bias mitigation | `notebooks/group/04_fairness_analysis.ipynb`, `results/fairness/`, report draft | Evidenced |
| AI usage declaration | Context AI log + report draft section | Requires group review before submission |
| Final report max 15 pages / PDF / Group ID filename | Draft content exists in `reports/FINAL_REPORT_DRAFT.md` | Final PDF cannot be completed without verified Group ID |
| Formal holdout | Locked in `notebooks/group/06_final_holdout_evaluation.ipynb` | Correctly not executed before freeze |

## Unresolved items that must not be hidden

1. Official Group ID and personal identity mapping are not present in the repository.
2. EBM has not evidenced a genuine multi-candidate tuning run.
3. FT-Transformer has saved controlled-screening/OOF evidence, but the final strict-nested candidate workflow has not been independently reproduced in this package.
4. The final holdout remains intentionally untouched because no final model is frozen.
