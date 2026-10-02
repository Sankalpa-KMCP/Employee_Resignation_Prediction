# Current Evaluation Guide

This guide reflects the current repository state.

## Headline evidence

| Member | Model | Development evaluation | Tuning status |
|---|---|---|---|
| 1 | FT-Transformer | Saved 5-fold OOF evidence | Controlled screening exists; final strict-nested 8-candidate rerun not independently reproduced |
| 2 | CatBoost | 5-fold OOF | Multi-candidate nested tuning evidenced |
| 3 | XGBoost | 5-fold OOF | Member 3 P1 one-hot 8-candidate strict nested tuning; P2/P3 descriptive preprocessing sensitivity |
| 4 | LightGBM | 5-fold OOF | Strict nested tuning evidenced |
| 5 | EBM | 5-fold OOF | Completed nested selection among three raw candidates, with three standardized sensitivity pairs |
| 6 | TabNet | 5-fold OOF | Strict nested tuning evidenced |

## Canonical group-stage evidence

- Comparison: `notebooks/group/02_model_comparison.ipynb` and `results/comparison/`
- Calibration diagnostics: `notebooks/group/03_calibration.ipynb` and `results/calibration/`
- Ethics/fairness: `notebooks/group/04_fairness_analysis.ipynb` and `results/fairness/`
- Optional prioritization simulation: `notebooks/group/05_retention_prioritization.ipynb` and `results/prioritization/`
- Final holdout: `notebooks/group/06_final_holdout_evaluation.ipynb` — intentionally locked

## Current conclusion

Development prevalence is approximately 0.1001. The current Member 3 P1 XGBoost result has the strongest pooled AP at approximately 0.101181, with ROC-AUC approximately 0.502471. The group comparison uses pooled AP for its numerical ranking and also reports mean outer-fold AP; TabNet is highest on the latter. The tiny differences do not establish useful predictive signal.

The FT historical weighted run is reproducibly configured as FT02 and loaded read-only by default. Its modern eight-candidate nested rerun remains unexecuted. EBM tuning is complete. The formal holdout remains locked while the group decides whether the weak-signal result warrants a corroborating run.

See `reports/INDIVIDUAL_MODEL_EVIDENCE.md` for each member's preprocessing and parameter comparison, including the limits of the small exploratory salary-ablation exercise.

## Viva emphasis

Each member must explain model suitability, representation/preprocessing, important hyperparameters, the number/type of variants evaluated, class imbalance treatment, validation design, metrics, limitations and saved evidence. Every member should also understand why AP is primary, why OOF alignment matters, and why the negative result is academically defensible.
