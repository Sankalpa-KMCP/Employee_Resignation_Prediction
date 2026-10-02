# Final Viva Brief

## Shared 30-second project summary

We evaluated six different model families on the same frozen 80,000-row development partition using five saved CV folds and kept a separate 20,000-row holdout quarantined. Average Precision was the primary metric because resignation prevalence is about 10%. After tuning/evaluation, every model remained around the no-skill baseline. Our conclusion is therefore a defensible negative result: the available predictors do not provide strong out-of-sample resignation signal, and we do not recommend operational deployment.

## Model evidence to know

- FT-Transformer: saved controlled-screening and five-fold OOF evidence; final strict-nested rerun still unresolved.
- CatBoost: strict nested candidate evidence.
- XGBoost: eight-candidate strict nested upgrade; pooled AP ≈ 0.101115.
- LightGBM: strict nested candidate evidence; adapter API issue repaired in current source.
- EBM: five-fold default-configuration OOF; do **not** call it tuned yet.
- TabNet: strict nested candidate evidence.

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
