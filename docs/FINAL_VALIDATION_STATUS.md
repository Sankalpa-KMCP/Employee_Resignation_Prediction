# Final Validation Status

This file records the state of the upgraded package, not a claim that every optional dependency was executable in the review environment.

## Verified

- Raw dataset: 100,000 rows × 20 columns, zero missing values, zero full-row duplicates, 10.01% positive target rate.
- Frozen registry: 80,000 development rows, 20,000 holdout rows, five development folds.
- Canonical FT GPU run `FT02_20261003T025004Z`: AP 0.099842, ROC-AUC 0.498905, Brier 0.247200, 8.52 minutes; fixed FT02 with inner epoch selection followed by a fresh inner-training fit. Saved-result loading is the default.
- Six canonical OOF files align to the same 80,000 development Employee_IDs.
- Formal holdout is absent from model-comparison/fairness/calibration inputs.
- Member 3 XGBoost primary P1 one-hot nested search completed: 8 candidates × 5 outer folds with inner development-only selection; this is the canonical group input. P2/P3 are descriptive preprocessing comparisons.
- Group comparison, calibration diagnostics, fairness diagnostics and exploratory prioritization notebooks execute from saved development evidence.
- EBM formal artifacts from `main`: 80,000 nested-development OOF rows, five folds, six executed varieties, three raw primary candidates, zero holdout overlap; formal loader and Windows CRLF checksum checks pass.
- LightGBM adapter uses the supported `eval_set` interface in the current source.

## Not independently verified in this environment

- Optional eight-candidate FT search and full outer-training refit: not executed and outside the completed fixed-FT02 scope. Original GPU exports did not retain per-fold adapter diagnostics; their corrected protocol metadata explicitly discloses that gap.
- Full local EBM retraining: the imported six-variety nested-development run is supported by saved artifacts; all 60 fits have not been independently rerun on this machine.
- Official Group ID / member identity mapping, because these are not established by the repository.
- Formal holdout performance, intentionally not evaluated before a final development freeze.
