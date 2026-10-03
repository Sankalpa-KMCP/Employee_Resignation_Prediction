# IT25100429 individual contribution

## Scope and demonstration
My responsibilities are numerical standardization and the Member 5 Explainable Boosting Machine classifier. Standardization supports the model investigation; EBM is the implemented classifier satisfying the individual model requirement.

Open `notebooks/team_contributions/preprocessing/IT25100429_Standardization.ipynb` for numerical feature selection, raw/scaled statistics, salary histograms, mathematical explanation and a training-fold-only scaler demonstration. The whole-development scaling section is descriptive, not a reusable CV transformer.

Open `notebooks/team_contributions/models/member_5_ebm/09_m5_ebm.ipynb` for model suitability, implementation, six varieties, bounded tuning, validation rationale, metrics, parameter/preprocessing comparison, conclusions and model explanations. The helper is `src/models/ebm_experiments.py`.

## Assessment evidence map

| Individual requirement | Evidence |
| --- | --- |
| Design and implement a classifier | InterpretML EBM with native nominal/continuous feature types |
| Dataset/problem suitability | Mixed tabular HR predictors, binary target, interpretable additive effects |
| Libraries/functions/hyperparameters | Notebook implementation section and helper source |
| Parameter tuning method | Three-setting bounded manual grid; inner-validation AP selection |
| Number and variety trained | Three hyperparameter settings times two numerical preprocessing representations: six varieties; 30 inner fits and 30 outer refits |
| Metrics for every variety | `candidate_comparison.csv` and `candidate_fold_metrics.csv` |
| Appropriate validation | Frozen five outer folds, stratified inner split, train-only preprocessing, holdout exclusion |
| Compare varieties and conclude | Paired scaling table, comparison plot, nested metrics and generated conclusion |
| Limitations/improvements | Notebook discussion of weak signal, selection variability, threshold/calibration and future data |
| Code/output demonstration | Executed notebooks, figures, inner selection log and OOF prediction checks |

## Result locations and group integration
Canonical results are under `results/models/ebm/individual_IT25100429/`. Use its `summary_metrics.json`, `fold_metrics.csv` and `oof_predictions.csv` for the group's EBM comparison, rather than historical default-only metrics in the parent directory. Pooled out-of-fold Average Precision is the primary group ranking metric. Mean outer-fold AP and its standard deviation describe fold stability; they are a different aggregation.

The notebook now defaults to validated saved evidence. `RUN_EBM_TRAINING=True` deliberately repeats 60 experimental fits plus one explanation fit in a new `results/models/ebm/reruns/` directory. New runs do not replace the canonical group input automatically. The notebook's parameter table is generated from `configs/models/ebm.yaml`: E02 has two leaves and E03 has three.

Raw preprocessing remains the primary EBM representation. Standardized numerical variants are a sensitivity experiment and do not alter the approved features. Outer validation scores do not select the primary candidates. A predeclared .001 absolute inner AP margin prefers fewer interactions and leaves when practically equivalent. This is not a significance claim.

The historical notebook is retained as `09_m5_ebm_historical.ipynb`. No final group model, calibrator or decision threshold has been frozen. The final holdout is not evaluated. The other five individual models and group analysis remain separate responsibilities.

## Viva prompts
- EBM sums an intercept, feature functions and optional interaction functions on the log-odds scale; probabilities come from the logistic link.
- Nominal fields have no natural numeric ordering, so EBM receives explicit nominal types.
- StandardScaler uses training means and population standard deviations; validation distributions must not inform those statistics.
- Five folds are repeated validation partitions, not five tuned model varieties.
- Average Precision matters at roughly 10% positive prevalence; approximately 90% accuracy alone can describe a model detecting no resignations.
- Internal early stopping monitors log loss, while the external bounded candidate comparison uses inner AP.
- Global term importance measures the fitted model's contribution magnitudes, not causal effects or guaranteed useful predictive signal.

## Reproduction environment
The verification environment is separate from the project's pinned requirements. Exact versions are recorded in the generated run metadata and `reports/IT25100429_verification_requirements.txt`. Use a suitable Python environment, select its Jupyter kernel, open each notebook and restart/run all from the project root. Full EBM execution retrains all 60 experimental fits plus one explanation-only fit on outer training. Runtime depends on the machine.

Source: https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html

## Verified development findings

Both notebooks were executed from a fresh kernel with zero error outputs. The targeted test suite passed 35 tests. Each of the six varieties has 80,000 validated out-of-fold rows, and the primary nested selection procedure has a separate 80,000-row OOF file. All prediction IDs are development IDs with zero holdout overlap.

Primary mean fold AP: 0.099558 (fold SD 0.002073); pooled AP: 0.099000; ROC-AUC: 0.495976; Brier: 0.090109. Recall and F1 at .50 are zero despite accuracy .8999. This run does not demonstrate useful resignation prediction.

Raw and standardized versions gave identical OOF probabilities for each parameter setting. Standardization supplied no EBM performance benefit in this experiment. E01 had the highest fixed-variety mean fold AP (.099570), with zero interactions and two leaves, but this descriptive ranking must not be substituted for an unbiased nested-selection estimate. The inner-selected raw configurations were E02 for folds 1–2, E01 for folds 3–4 and E03 for fold 5.

The environment used Python 3.14 and InterpretML 0.7.8, with other exact versions in the verification requirements. Execution in the formal Python 3.11 environment remains to be checked. Individual marks still depend on personally explaining and demonstrating the work.

## Upstream refactored-branch verification record

The contribution is restored on `refactor/file-architecture` using the new notebook locations. The standardization root resolver searches all parent directories. The canonical six-variety script replaces the unexecuted eight-candidate script. All four group-analysis notebooks now use the portable manifest loader for EBM. The existing parameter settings and trained OOF artifacts are unchanged; relocation verification uses these validated artifacts rather than repeating all 60 experimental fits.

Refactored-branch verification completed: 96 relevant tests passed (including CRLF portability, strict config, frozen-holdout guards, group utilities and notebook architecture). Both relocated individual notebooks were verified from their own directories. EBM verification consumed the unchanged manifest-validated trained artifacts and ran its explanation demonstration; the full 60-fit training cell remains available for deliberate reruns. Group comparison, calibration diagnostics, fairness and prioritization notebooks executed successfully using the canonical EBM input, with derived outputs refreshed. The final holdout notebook was not executed. No native Windows run, commit or push was performed.
