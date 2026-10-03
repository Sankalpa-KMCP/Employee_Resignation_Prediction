# Final holdout protocol

`notebooks/group/06_final_holdout_evaluation.ipynb` defaults to readiness inspection. It writes aggregate prerequisites under `results/holdout_readiness/` and does not read raw employee data, model weights or formal holdout predictions. A model ranking leader alone is insufficient for evaluation.

## Required freeze

The group must approve all development decisions before formal access and save a trusted complete pipeline or project adapter as `models/frozen/pipeline.joblib`. It must expose `predict_proba` on the exact approved feature DataFrame. A one-dimensional result is the positive-class probability; a two-column result requires explicit binary `classes_`. Preprocessing and the chosen probability transformation must already be inside that artifact. The evaluator never fits a component. Joblib deserialization is supported only for trusted local artifacts.

The comparison decision must set `final_model_frozen=true` and explicitly name `final_model`. Calibration must record `probability_source_frozen=true` and `probability_source` as `raw`, `sigmoid` or `isotonic`. The raw option requires a deliberate development decision; it is not an automatic substitute for a required calibrator. Prioritization must set `policy_frozen=true` and explicitly record `holdout_scope="prediction_only"` for this evaluator. Optional downstream holdout corroboration requires a separately frozen evaluator before any access and is outside this evaluator's scope.

Create `models/frozen/freeze.json` with these fields. The notebook displays an illustrative schema but does not write a real freeze.

| Fields | Required meaning |
| --- | --- |
| `freeze_id`, `pipeline_frozen` | Approved freeze identifier and affirmative declaration |
| `model` | Matches the comparison decision's `final_model` |
| `training_partition`, `training_rows` | `development` and the contract's complete development row count |
| `feature_columns` | Exact contract predictor names and order |
| `pipeline_path` | Repository-relative trusted artifact path |
| `probability_source`, `includes_frozen_probability_source` | Matches calibration decision; selected transformation is already included |
| `threshold`, `threshold_source` | Numeric cutoff in [0,1] and development-only rationale |
| `downstream_policy_frozen`, `holdout_scope` | Affirmative policy freeze and explicit `prediction_only` scope |
| `formal_holdout_used_for_selection` | Explicitly false |
| `historical_audit_exposure_disclosed` | Explicitly true; earlier aggregate audit exposure remains a limitation |
| `artifacts` | Repository-relative path → exact-byte SHA-256 mapping |

Bind exactly the pipeline, feature contract, analysis registry and comparison/calibration/fairness/prioritization decision JSON files in `artifacts`; extra entries are rejected so readiness cannot accidentally hash raw employee data. The joblib pipeline must resolve under `models/frozen/`. Hashes prevent silently accepting changed files. They do not reconstruct training provenance: full-development training and lack of formal-holdout-guided selection remain recorded group attestations. Paths must resolve within the checkout.

## One-time assessment

Actual evaluation requires explicit activation and every freeze check to pass. The evaluator verifies normalized source/split checksums, then uses ID-only inspection to parse only the formal holdout's predictors and labels. Every assigned holdout ID must be present exactly once. All approved rows and class imbalance are retained.

Before target parsing or prediction, exclusive creation of `results/final_holdout/assessment_receipt.json` claims the assessment. An empty reserved directory or `.gitkeep` is permitted. Any receipt or result blocks repeat assessment; a failure leaves the started receipt in place and requires investigation, never an automatic retry or deletion to obtain better results. This local filesystem guard is not tamper-proof against someone deliberately removing evidence.

Exports include aggregate classification metrics at the frozen threshold, AP/ROC-AUC, Brier/ECE, reliability bins, PR/ROC curves and percentile intervals from 2,000 stratified bootstrap resamples with seed 42. Resamples condition on the fixed prediction set and do not measure retraining uncertainty. Employee-level predictions are not exported. The notebook adds corroboration figures and the development-versus-holdout AP/AUC comparison. Raw OOF Brier must not be interpreted as a like-for-like calibration comparison if the final probability source changed.

No formal holdout result may drive subsequent model, preprocessing, calibration, threshold, proxy, capacity or strategy selection. The historical preliminary-audit exposure prevents describing the assessment as a guaranteed pristine independent test.
