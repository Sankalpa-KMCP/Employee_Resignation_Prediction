# Current Evaluation Guide

This guide reflects the current repository state.

## Headline evidence

| Member | Model | Development evaluation | Tuning status |
|---|---|---|---|
| 1 | FT-Transformer | Completed GPU FT02 5-fold OOF | Fixed FT02; inner-AP epoch selection and fresh inner-training fit; historical screening is descriptive |
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
- Final holdout: `notebooks/group/06_final_holdout_evaluation.ipynb` — default Run All produces a readiness report; actual evaluation remains locked

## Group-stage interpretation and execution

Calibration reports raw OOF reliability, occupied-bin counts, Brier error relative to a constant development-prevalence reference, and outer-fold diagnostics. It does not fit a calibrator on its own evaluation labels or claim a frozen probability source. Calibration and discrimination must be interpreted separately.

Fairness attaches demographics using `load_development_data`, which isolates development rows before parsing their features and targets. Age groups come from `configs/feature_contract.yaml`. Subgroup tables retain positive, negative and predicted-positive denominators. Undefined precision, recall or false-positive rates remain missing, not fabricated zeros. AP/AUC size flags and max-minus-min gaps are descriptive reporting aids, not proof of fairness or statistical significance.

Prioritization recomputes the pooled-AP leader from validated OOF files. Its risk source remains exploratory and uncalibrated. Proxy MinMaxScalers fit each outer-training development partition, then transform its validation rows with clipping and a constant-component-to-zero policy. The equal-weight primary proxy, no-salary mean and epsilon-adjusted multiplicative sensitivity use the approved components; the multiplicative epsilon is explicitly 0.01. All sensitivity rankings report value-weighted capture against the same primary proxy. Capacity stays at 20% with 10%/30% sensitivity, and 200 seed-42 random allocations provide an empirical allocation range, not model-performance confidence intervals. Exported tables contain no employee-level rankings.

The holdout notebook runs readiness checks without reading employee data or deserializing a pipeline. Actual evaluation uses the separate one-time evaluator in `src/evaluation/holdout.py`, requires a hash-bound trusted development freeze and explicit activation, and preserves a started receipt after failures. See [Final holdout protocol](FINAL_HOLDOUT_PROTOCOL.md) for the supported prediction-only scope, manifest fields and limitations.

## Current conclusion

Development prevalence is approximately 0.1001. The current Member 3 P1 XGBoost result has the strongest pooled AP at approximately 0.101181, with ROC-AUC approximately 0.502471. The group comparison uses pooled AP for its numerical ranking and also reports mean outer-fold AP; TabNet is highest on the latter. The tiny differences do not establish useful predictive signal.

The canonical FT02 GPU run `FT02_20261003T025004Z` is loaded read-only by default: AP 0.099842, ROC-AUC 0.498905, Brier 0.247200, runtime 8.52 minutes. Selection weights are not reused: a fresh model fits the inner-training rows for the chosen epoch count. The optional eight-candidate search was not executed, and equivalent candidate tuning across all six models is not claimed. EBM tuning is complete. The formal holdout remains locked while the group decides whether the weak-signal result warrants a corroborating run.

See `reports/INDIVIDUAL_MODEL_EVIDENCE.md` for each member's preprocessing and parameter comparison, including the limits of the small exploratory salary-ablation exercise.

## Viva emphasis

Each member must explain model suitability, representation/preprocessing, important hyperparameters, the number/type of variants evaluated, class imbalance treatment, validation design, metrics, limitations and saved evidence. Every member should also understand why AP is primary, why OOF alignment matters, and why the negative result is academically defensible.
