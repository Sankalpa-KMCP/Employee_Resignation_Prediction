# IT24100879 — supplementary Regularized Logistic Regression

Standalone L2 Logistic Regression without PCA, contributed by IT24100879.
Uses the approved predictors and frozen five-fold development registry. All
scaling and encoding are training-local. Ten C/weight candidates per outer fold
are selected by minimum log loss on a stratified 15% inner validation split,
then refitted on the entire outer training partition (50 inner fits + 5 refits).
No rows are dropped or resampled; class imbalance is addressed through stratified
splitting, none/balanced weighting candidates and prevalence-based comparisons.

## Observed results

Development rows: 80,000; prevalence: 0.100100.
OOF AP: 0.099964; mean fold AP:
0.100301 ± 0.001158 (descriptive SD).
ROC-AUC: 0.498385 (chance 0.5).
Log loss: 0.325420; fold-training prevalence baseline:
0.325303.
Brier: 0.090101; prevalence baseline:
0.090080.

```text
 fold candidate_id     C class_weight  average_precision  log_loss
    1          L01 0.001         none           0.099523  0.325389
    2          L01 0.001         none           0.102165  0.325272
    3          L01 0.001         none           0.100688  0.325420
    4          L01 0.001         none           0.099411  0.325593
    5          L01 0.001         none           0.099719  0.325425
```

Calibration is evaluated with OOF reliability bins; probabilities are not
post-hoc recalibrated. Low probability error can reflect learning prevalence,
without useful ranking. Small score differences do not establish operational
usefulness. Coefficients are regularized conditional associations, not causes.

## Academic prioritization

Combined score = OOF probability × equal-weight normalized performance/salary/
tenure proxy, with fold-training min/max bounds and clipping. Selection is within
each outer validation fold. See ranking_summary.csv for 10%, 20%, 30% capacity.
Random expected lift is 1; the line is not an empirical allocation experiment.
The score is neither employee worth nor intervention benefit. No model is frozen
for deployment, and the formal holdout is excluded. Local record-level OOF is
ignored by Git. The shared PCA benchmark and group registry are unchanged.

## Reproduction

Run `python scripts/run_logistic_regression_experiment.py`, then Restart and Run All
in `11_IT24100879_logistic_regression_extra.ipynb` (saved review by default).
Runtime versions: scikit-learn 1.9.1, pandas 3.0.6,
NumPy 2.5.3, Matplotlib 3.11.2.

Observed conclusion: essentially chance ranking (AP 0.099964; ROC-AUC 0.498385),
with log loss and Brier slightly worse than prevalence baselines. All folds chose
C=0.001 and no class weighting. At 20% capacity combined-score mean lift is 1.00025.
No useful resignation discrimination is demonstrated.
