# Final Viva Brief

## Shared 30-second project summary

We evaluated six different model families on the same frozen 80,000-row development partition using five saved CV folds and kept a separate 20,000-row holdout quarantined. Average Precision was the primary metric because resignation prevalence is about 10%. After tuning/evaluation, every model remained around the no-skill baseline. Our conclusion is therefore a defensible negative result: the available predictors do not provide strong out-of-sample resignation signal, and we do not recommend operational deployment.

## Model evidence to know

- FT-Transformer: canonical GPU FT02 five-fold OOF, AP ≈ 0.099842, ROC-AUC ≈ 0.498905, runtime 8.52 minutes. Inner validation chooses epochs, then a fresh model fits inner-training rows only. Historical screening supports the fixed candidate; the optional eight-candidate search was not executed.
- CatBoost: strict nested candidate evidence.
- XGBoost: Member 3's current P1 one-hot eight-candidate nested run; pooled AP ≈ 0.101181. P2/P3 are sensitivity evidence.
- LightGBM: strict nested candidate evidence; adapter API issue repaired in current source.
- EBM: completed five-fold nested selection among three raw candidates plus standardized sensitivity pairs; pooled AP ≈ 0.099000.
- TabNet: strict nested candidate evidence.

The six members' compact preprocessing, parameter, metric and validation tables are in `reports/INDIVIDUAL_MODEL_EVIDENCE.md`. The short salary-masking results for Members 1, 2, 4 and 6 are exploratory teaching evidence, not substitutes for the full five-fold OOF results.

## Questions every member should be able to answer

1. Why is Average Precision more informative than accuracy for this imbalanced target?
2. What is an OOF prediction and why does every development row appear exactly once?
3. Why is the formal holdout still locked?
4. What preprocessing is fitted inside folds rather than globally?
5. Which hyperparameters were varied for your model and how was the candidate selected?
6. What does your model's AP/ROC-AUC imply relative to prevalence/chance?
7. Why can a good Brier score coexist with poor discrimination?
8. What are the ethical risks of employee-resignation prediction?
9. Why is the low-performance conclusion academically valid rather than a coding failure?
10. What would you improve with better data?

## Safe conclusion

Do not say “XGBoost is a successful predictor.” Say: “XGBoost is the current numerical development leader, but the gain over the no-skill baseline is negligible, so we cannot claim practically useful predictive signal.”

## Twenty minute demonstration schedule

The implementation evaluation allocates 20 minutes to the group. This rehearsal plan reserves four minutes for questions; adjust it to the examiner's instructions. Use saved code and outputs rather than launching long training runs during the viva.

| Time | Presenter | Demonstration |
|---|---|---|
| 0–2 minutes | [INSERT GROUP PRESENTER] | Problem, dataset provenance limits, target imbalance, frozen 80,000/20,000 split and common OOF contract. |
| 2–4 minutes | [INSERT MEMBER 1 NAME AND IT NUMBER] | FT representation, fixed FT02 scope, epoch selection, weighting and paired salary-mask results. |
| 4–6 minutes | [INSERT MEMBER 2 NAME AND IT NUMBER] | Native CatBoost categories, eight candidates, inner selection, balancing and missing serialized artifact disclosure. |
| 6–8 minutes | [INSERT MEMBER 3 NAME AND IT NUMBER] | XGBoost P1/P2/P3, eight candidates, training-only transformations, inner AP selection and outer refit. |
| 8–10 minutes | [INSERT MEMBER 4 NAME AND IT NUMBER] | LightGBM categorical mapping, strict nested selection, selected iterations and preprocessing sensitivity. |
| 10–12 minutes | [INSERT MEMBER 5 NAME AND IT NUMBER] | EBM additive terms/interactions, three raw and three standardized variants, formal manifest and imported evidence limits. |
| 12–14 minutes | [INSERT MEMBER 6 NAME AND IT NUMBER] | TabNet masks, fold-local codes/scaling, eight candidates, class balance and early stopping. |
| 14–16 minutes | [INSERT GROUP PRESENTER] | Six-model comparison, AP versus prevalence, calibration versus discrimination, fairness and capacity results, locked holdout. |
| 16–20 minutes | All members | Examiner questions and concise explanations of limitations and future improvements. |

## Files to open before the viva

Each member's model notebook is under `notebooks/team_contributions/models/`; open the corresponding configuration in `configs/models/` alongside it. Default saved-review mode must remain selected. The paths below are the formal group evidence, not interchangeable historical runs.

| Model | Model notebook | Canonical result directory |
|---|---|---|
| FT-Transformer | `member_1_ft_transformer/05_m1_ft_transformer.ipynb` | `results/models/ft_transformer/reruns/FT02_20261003T025004Z/` |
| CatBoost | `member_2_catboost/06_m2_catboost.ipynb` | `results/models/catboost/` |
| XGBoost | `member_3_xgboost/07_m3_xgboost.ipynb` | `results/models/xgboost/` |
| LightGBM | `member_4_lightgbm/08_m4_lightgbm.ipynb` | `results/models/lightgbm/strict_nested/` |
| EBM | `member_5_ebm/09_m5_ebm.ipynb` | `results/models/ebm/individual_IT25100429/` |
| TabNet | `member_6_tabnet/10_m6_tabnet.ipynb` | `results/models/tabnet/` |

For each model, show input preparation, the configured candidate settings, inner-selection logic and saved outer metrics. Use `reports/INDIVIDUAL_MODEL_EVIDENCE.md` to discuss measured preprocessing varieties. The FT file is named `ft_transformer_final_oof_predictions.csv`; the other formal directories contain `oof_predictions.csv`. Group comparison reads the exact paths in `configs/analysis_registry.yaml`.

For the group discussion, open `notebooks/group/02_model_comparison.ipynb` and its AP chart. Then show a reliability table from notebook 03, group denominators and representation from notebook 04, and the 20% capacity comparison from notebook 05. Notebook 06 demonstrates readiness checks; it does not contain observed formal-holdout scores. Missing model weights do not invalidate saved OOF metrics, but prevent claims that every serialized model is packaged for deployment.

## Answer guide

- AP summarizes positive-class ranking across thresholds. Compare it with prevalence 0.1001; a majority-class model can reach 89.99% accuracy with zero recall.
- An OOF score is produced while that employee's outer fold is excluded from training and selection. All 80,000 development IDs appear once; inner splits choose parameters without using outer-validation labels.
- The formal holdout is reserved for one frozen development decision. Repeatedly using it to change parameters would turn it into another validation set.
- Pooled AP ranks the group inputs; mean fold AP describes fold results separately. XGBoost leads pooled AP, while TabNet leads mean fold AP. Their tiny gains are not demonstrated practical improvement.
- Brier is mean squared probability error. Near-prevalence scores can have low Brier and ECE while providing poor ranking. The constant-prevalence Brier is approximately 0.09008; EBM does not beat it.
- At 20% capacity, risk-only lift is 1.018981. The random allocations provide a comparison range, not evidence of an effective intervention. The value proxy is a synthetic index, not employee worth.
- Group diagnostics do not certify fairness. Missing denominators stay missing, and sensitive-feature ablations are still proposed rather than measured.
- Better future work starts with verified data provenance, a prediction time and outcome horizon, plausible pre-outcome predictors and validation across time or employers.

## Member confirmation checklist

These items require actual group review and rehearsal; this prepared brief does not confirm anyone's readiness.

- [ ] Replace the Group ID, name and IT-number placeholders in the Word report.
- [ ] Each member can locate and explain their preprocessing, model function and varied parameters.
- [ ] Each member can distinguish measured variants, historical evidence and unexecuted extensions.
- [ ] Each member can explain inner selection, OOF metrics and the near-baseline conclusion without reading a script.
- [ ] The group has reviewed the AI-use declaration against its actual tool usage.
- [ ] Rehearse the complete demonstration within 16 minutes, keeping four minutes for questions.
