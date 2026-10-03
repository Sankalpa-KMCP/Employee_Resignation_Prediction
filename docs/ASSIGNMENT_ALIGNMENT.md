# IT2011 Final-Evaluation Alignment

| Requirement | Evidence in upgraded repository | Status |
|---|---|---|
| One model per student | Six model notebooks under `notebooks/team_contributions/models/` | Evidenced |
| Model suitability / implementation explanation | Model notebooks + configs + `src/models/` | Evidenced |
| Hyperparameter tuning / varieties | Five models have nested candidate-selection evidence; FT has historical controlled screening and completed fixed-FT02 epoch selection | Evidenced with different validation strength; equivalent nested candidate tuning is not claimed |
| Different preprocessing / hyperparameter variants | XGBoost and EBM full five-fold comparisons; bounded paired exploratory checks for FT-Transformer, CatBoost, LightGBM and TabNet in `reports/INDIVIDUAL_MODEL_EVIDENCE.md` | Evidenced with explicitly different validation strength; FT completed scope is fixed FT02 plus historical screening |
| Relevant metrics | AP, ROC-AUC, Brier, threshold metrics, fold stability | Evidenced |
| Validation method | Frozen 80/20 split and five development CV folds | Evidenced |
| Compare six individual models | `notebooks/group/02_model_comparison.ipynb`, `results/comparison/` | Completed on development OOF |
| Discuss challenges / expected behavior | Report draft, comparison conclusion, weak-signal analysis | Evidenced |
| Ethics / bias mitigation | `notebooks/group/04_fairness_analysis.ipynb`, `results/fairness/`, report draft | Evidenced |
| AI usage declaration | Context AI log + report draft section | Requires group review before submission |
| Final report max 15 pages / PDF / Group ID filename | Nine-page Word report at `reports/FINAL_REPORT.docx`, with editable source at `reports/FINAL_REPORT_DRAFT.md` | Word delivered as requested; identity placeholders remain. Recheck the page limit after filling them and confirm the assessment's submission format. No PDF generated. |
| Formal holdout | Locked in `notebooks/group/06_final_holdout_evaluation.ipynb` | Correctly not executed before freeze |

## Unresolved items that must not be hidden

1. Official Group ID and personal identity mapping are not present in the repository.
2. EBM nested development results are imported from `main` and manifest-verified; a full local retraining of the 60 experimental fits has not been performed.
3. FT-Transformer has completed GPU fixed-FT02 OOF evidence, with historical screening. Its optional eight-candidate extension was not executed; original per-fold adapter diagnostics are unavailable. Its final fit uses inner-training rows only, so protocols are not equivalent across all six models.
4. Formal holdout evaluation remains locked because no final model is frozen. Historical aggregate audit exposure is disclosed in [the holdout protocol](FINAL_HOLDOUT_PROTOCOL.md); pristine label-blind status is not claimed.
