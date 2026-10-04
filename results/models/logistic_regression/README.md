# Supplementary Regularized Logistic Regression

Standalone L2 Logistic Regression without PCA, supplementary to Member 5's EBM.
It uses the approved predictors and frozen five development folds. The group model
registry and existing PCA benchmark retain their current roles.

Run `python scripts/run_logistic_regression_experiment.py` from the project root.
No new project dependencies are required. The notebook defaults to saved review;
enable `RUN_EXPERIMENT` explicitly to regenerate the evidence.

Ten candidates combine C = 0.001, 0.01, 0.1, 1, 10 with none/balanced class weights.
Each outer training fold contains a stratified 15% inner validation split. Minimum
inner log loss selects the candidate before one full outer-training refit. Numeric
scaling and one-hot encoding are always fitted inside the current training set.
No threshold tuning, PCA, resampling, post-hoc calibration or holdout evaluation is
performed. Failures to converge stop execution.

Saved evidence includes inner candidates, selections, outer metrics, reliability
bins, coefficient illustrations and capacity-ranking summaries. Calibration and
ranking discrimination are separate: nearly constant prevalence probabilities
can be calibrated while offering little useful ranking. Compare log loss and
Brier against outer-training prevalence baselines, and AP against prevalence.

The academic combined score is `OOF probability * retention_value_proxy`. The proxy
equally averages performance, salary and tenure min/max normalization fitted on
each outer training set, with validation values clipped to [0, 1]. This matches the
group proxy's components but uses fold-local bounds. Capacity selection is evaluated
within each outer fold, at 10%, 20%, 30%; averages and SD are descriptive. The random
line represents expected lift 1, not an empirical random-allocation experiment.
This score estimates neither employee worth nor intervention benefit.

The record-level OOF file is ignored by Git and remains local. Aggregate evidence
can be shared; a deployable full-development fitted model is not created.
