# Final Viva Brief

## Shared 30-second project summary

We evaluated six different model families on the same frozen 80,000-row development partition using five saved CV folds and kept a separate 20,000-row holdout quarantined. Average Precision was the primary metric because resignation prevalence is about 10%. After tuning/evaluation, every model remained around the no-skill baseline. Our conclusion is therefore a defensible negative result: the available predictors do not provide strong out-of-sample resignation signal, and we do not recommend operational deployment.

## Model evidence to know

- FT-Transformer: saved controlled-screening and five-fold OOF evidence; final strict-nested rerun still unresolved.
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
