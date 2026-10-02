# Final Validation Status

This file records the state of the upgraded package, not a claim that every optional dependency was executable in the review environment.

## Verified

- Raw dataset: 100,000 rows × 20 columns, zero missing values, zero full-row duplicates, 10.01% positive target rate.
- Frozen registry: 80,000 development rows, 20,000 holdout rows, five development folds.
- Six canonical OOF files align to the same 80,000 development Employee_IDs.
- Formal holdout is absent from model-comparison/fairness/calibration inputs.
- XGBoost upgraded nested search completed: 8 candidates × 5 outer folds with inner development-only selection.
- Group comparison, calibration diagnostics, fairness diagnostics and exploratory prioritization notebooks execute from saved development evidence.
- LightGBM adapter uses the supported `eval_set` interface in the current source.

## Not independently verified in this environment

- Final strict-nested FT-Transformer execution, because `rtdl-revisiting-models` is not installed in the review environment and network package installation was unavailable.
- EBM multi-candidate tuning, because the supplied package contains only a fixed/default EBM OOF run and the `interpret` dependency was unavailable in the review environment.
- Official Group ID / member identity mapping, because these are not established by the repository.
- Formal holdout performance, intentionally not evaluated before a final development freeze.
