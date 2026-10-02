# Current Evaluation Guide

This guide reflects the upgraded repository state. ## Headline evidence

| Member | Model | Development evaluation | Tuning status |
|---|---|---|---|
| 1 | FT-Transformer | Saved 5-fold OOF evidence | Controlled screening exists; final strict-nested 8-candidate rerun not independently reproduced |
| 2 | CatBoost | 5-fold OOF | Multi-candidate nested tuning evidenced |
| 3 | XGBoost | 5-fold OOF | Upgraded 8-candidate strict nested tuning evidenced |
| 4 | LightGBM | 5-fold OOF | Strict nested tuning evidenced |
| 5 | EBM | 5-fold OOF | Current evidence is fixed/default configuration; bounded tuning script is ready but not executed |
| 6 | TabNet | 5-fold OOF | Strict nested tuning evidenced |

## Canonical group-stage evidence

- Comparison: `notebooks/group/02_model_comparison.ipynb` and `results/comparison/`
- Calibration diagnostics: `notebooks/group/03_calibration.ipynb` and `results/calibration/`
- Ethics/fairness: `notebooks/group/04_fairness_analysis.ipynb` and `results/fairness/`
- Optional prioritization simulation: `notebooks/group/05_retention_prioritization.ipynb` and `results/prioritization/`
- Final holdout: `notebooks/group/06_final_holdout_evaluation.ipynb` — intentionally locked

## Current conclusion

Development prevalence is approximately 0.1001. The strongest pooled AP is the upgraded nested XGBoost result at approximately 0.101115, with ROC-AUC approximately 0.504539. This is a negligible improvement over no-skill. The correct conclusion is weak/absent predictive signal, not model success.

Do not unlock the final holdout merely to search for a better result. A final model should only be frozen after the group resolves the remaining FT/EBM tuning evidence and decides whether the weak-signal result warrants a corroborating holdout run.

## Viva emphasis

Each member must explain model suitability, representation/preprocessing, important hyperparameters, the number/type of variants evaluated, class imbalance treatment, validation design, metrics, limitations and saved evidence. Every member should also understand why AP is primary, why OOF alignment matters, and why the negative result is academically defensible.
