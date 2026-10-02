# Upgrade Changelog — 2026-10-02

This package was upgraded from `Employee_Resignation_Prediction_Refactored.zip` through a review/fix/re-review cycle.

## Major corrections and additions

- completed the development-only six-model comparison, calibration, fairness and optional prioritization notebooks/results;
- executed an 8-candidate × 5-fold strict-nested XGBoost development search and preserved the rerunnable script/evidence;
- corrected XGBoost documentation/config status from fixed-only to current nested evidence;
- repaired the LightGBM sklearn `eval_set` interface in the current adapter;
- aligned the FT-Transformer notebook with the current candidate-registry config without fabricating a rerun;
- corrected EBM notebook status/metrics and explicitly retained the unresolved tuning requirement;
- added a ready-to-run bounded EBM nested-tuning script for a compatible InterpretML environment;
- replaced placeholder shared data/evaluation/prioritization modules with working utilities;
- converted the outlier-contribution notebook from remote Kaggle download to the bundled canonical raw dataset and executed it successfully;
- added `pytest.ini` so plain `pytest` resolves the local `src` package;
- added assignment alignment, viva brief, final validation status and a complete evidence-backed report draft;
- updated active README/evaluation documentation and added a dated live-state addendum to the canonical context;
- preserved the locked formal holdout; no holdout performance was generated.

## Deliberately unresolved

- official Group ID and personal identity mapping are not established by the repository;
- final strict-nested FT-Transformer execution was not independently reproduced because the review environment lacks `rtdl-revisiting-models`;
- EBM multi-candidate tuning has not been executed because the review environment lacks `interpret`;
- no final model is frozen because development discrimination remains near no-skill.
