# IT25100565 — Member 2 CatBoost: leakage-free nested evaluation

CatBoost is Member 2's assigned headline model. This experiment replaces the previous `06_m2_catboost.ipynb`. The previous notebook relied on external scripts and pre-computed artefacts and exceeded 4 MB; everything about the model now runs inside that single notebook, which is 0.5 MB. The group registry (`configs/analysis_registry.yaml`) now points CatBoost to this experiment's strict nested OOF predictions.

## Files

| Item | Path |
|---|---|
| Preprocessing notebook | `notebooks/team_contributions/preprocessing/IT25100565_Preprocessing.ipynb` |
| Model notebook | `notebooks/team_contributions/models/member_2_catboost/06_m2_catboost.ipynb` |
| Processed data (git-ignored) | `data/processed/catboost/employee_resignation_clean.csv` |
| Evidence (committed) | `results/models/catboost/` |
| Group OOF input | `results/models/catboost/oof_predictions.csv` (+ `fold_metrics.csv`, `summary_metrics.json`) |
| Serialized model (git-ignored) | `models/catboost/catboost_resignation_model.cbm` + `metadata.json` |

The earlier CatBoost evidence (the previous files in `results/models/catboost/`, `improvement/`, `thesis_figures/`, and `models/catboost/model.cbm` / `catboost_development_model.cbm`) was removed in this update; it remains in Git history. `results/models/catboost/` and `models/catboost/` now contain only this experiment's files. The old CatBoost-only Python files (10 scripts in `scripts/`, `src/evaluation/audit_plots.py`, `src/evaluation/audit_stats.py`, `src/models/catboost_protocol.py`, all six `tests/test_catboost_*` files) and the configs `audited_catboost.yaml` / `legacy_catboost.yaml` were also removed. `src/models/catboost_model.py` and `configs/models/catboost.yaml` are kept because the shared model package (`src/models/__init__.py`), the group preprocessing-sensitivity script and the shared group tests use them; this notebook does not.

## Implementation and validation

- **Model:** `catboost.CatBoostClassifier`, the correct variant for a binary target. `CatBoostRegressor` would treat the label as a continuous value, and `CatBoostRanker` needs query groups.
- **Native inputs:** the four categorical predictors are passed as `cat_features` through `Pool` with no manual encoding, so CatBoost's ordered target statistics apply. The 13 numeric predictors are passed raw, with no scaling. `text_features` and `embedding_features` are not applicable.
- **Settings:** `loss_function` and `eval_metric` are `Logloss`, with `boost_from_average=True` and `random_seed=42`.
- **Data:** the 17 contract predictors and the frozen split registry are used. All work uses the 80,000 development rows only, and the 20,000-row holdout is discarded right after the join.
- **Leakage controls:** every fold trains a fresh model. Early stopping uses a stratified 15% inner split of the training folds, never the validation fold. CatBoost's categorical statistics and class weights are fold-local. ID overlap is asserted in every fold. No SMOTE is used.
- **Stage A:** four preprocessing varieties are trained with default parameters. They are the contract baseline, a version without the redundant `Monthly_Salary`, three row-wise workload ratios, and label-encoded categories as a comparison only. A variety replaces the baseline only after a significant paired one-sided t-test (p < 0.05).
- **Stage B:** 25 randomly sampled configurations plus the default are tested with `ParameterSampler`. The search covers depth, learning rate, L2, random strength, bagging temperature, border count, one-hot size and class weighting (none, Balanced, SqrtBalanced). Selection uses the one-standard-error rule, then the lowest pooled OOF log-loss.
- **Strict nested CV, the group input:** inside each outer fold, all 26 candidates are scored on the inner split. The lowest inner log-loss wins, and the winner is refit on all outer-training rows to predict the untouched outer fold.
- **Fit count:** 150 Stage A/B fold models, 130 inner and 5 outer nested fits, and 1 final development refit.
- **Baselines:** the group's saved `DummyClassifier(prior)` and `DummyClassifier(most_frequent)` OOF files from `results/baselines/` are used.

## Results (development OOF, 80,000 rows)

Stage A varieties (default parameters):

```text
          config  mean_fold_AP   fold_SD  ROC-AUC  log_loss
     P1_baseline      0.101660  0.001492 0.501989  0.325316
    P2_no_salary      0.101435  0.000832 0.502391  0.325336
   P3_engineered      0.101354  0.002315 0.501938  0.325357
P4_label_encoded      0.099880  0.000380 0.500955  0.325361
```

No variety beat P1 significantly, so the contract feature set was kept.

Stage B best mean fold AP was B05 at 0.102387, a SqrtBalanced configuration with log-loss 0.3976. Four configurations fell within one SE. B21 had the lowest log-loss among them and was selected (depth 6, learning rate 0.0318, L2 7.95, 10 trees): non-nested mean fold AP 0.102332 ± 0.001976.

The strict nested CV is the unbiased estimate and the group input:

```text
cv_fold selected_config trees  average_precision  roc_auc  brier_score
      1             B15   110           0.097485 0.495636     0.090100
      2             B15    77           0.099285 0.492275     0.090083
      3             B18   123           0.101263 0.506117     0.090116
      4             B18     1           0.099861 0.496529     0.090101
      5             B18   102           0.103005 0.504191     0.090109
```

The nested results are:
- Mean fold AP is 0.100180 ± 0.002081 (descriptive SD), and pooled AP is 0.099860.
- Pooled ROC-AUC is 0.499605, Brier score is 0.090102 and log-loss is 0.325422.
- Prevalence is 0.1001. The Dummy Prior has pooled AP 0.100085 and Brier 0.090080.
- Nesting removed +0.0022 of mean fold AP that came from selecting on the same folds.

| Comparison | Point estimate | 95% bootstrap CI | Contains 0 |
|---|---|---|---|
| PR-AUC: CatBoost − Dummy Prior | −0.000225 | −0.002283 to +0.002367 | Yes |
| ROC-AUC − 0.5 | −0.000395 | −0.007001 to +0.006266 | Yes |

The paired 5-fold t-test against the Dummy Prior gives p = 0.935.

The threshold metrics below are descriptive, because each threshold is chosen on the same OOF predictions:
- **0.50:** CatBoost predicts no resignations, the same as both Dummies, with accuracy 0.8999 and recall 0.
- **Top-10% HR budget (0.1056):** it flags 10.0% of employees with precision 0.1003 and recall 0.1001, which equals random selection.
- **F1-optimal:** it flags 99.99% of employees.

## Effect on the group comparison

The group comparison (`02_model_comparison.ipynb`) and the calibration, fairness and prioritization notebooks (`03`–`05`) were re-run with the new registry input. The ranking below is by mean fold AP:

| Model | Mean fold AP ± SD | Pooled AP | ROC-AUC | Brier |
|---|---|---|---|---|
| TabNet | 0.101700 ± 0.003876 | 0.099965 | 0.499317 | 0.186177 |
| XGBoost | 0.101643 ± 0.000990 | 0.101181 | 0.502471 | 0.175329 |
| LightGBM | 0.100202 ± 0.002446 | 0.100153 | 0.498740 | 0.180215 |
| **CatBoost (this experiment)** | 0.100180 ± 0.002081 | 0.099860 | 0.499605 | 0.090102 |
| EBM | 0.099558 ± 0.002073 | 0.099000 | 0.495976 | 0.090109 |
| FT-Transformer | 0.098065 ± 0.002638 | 0.098648 | 0.497980 | 0.153086 |

The group leaders are unchanged: XGBoost by pooled AP and TabNet by mean fold AP. No model is frozen and the holdout is untouched. The previous CatBoost entry was 0.100690 ± 0.002784 mean fold AP, 0.099346 pooled AP and Brier 0.210753, the last from a class-weighted model. The new entry is unweighted and well calibrated, so CatBoost is now the **lowest raw-Brier model** in `results/calibration/calibration_decision.json`, with 0.090102 against EBM's 0.090109. That difference is negligible. `reports/Six_Model_Comparison.md`, `reports/INDIVIDUAL_MODEL_EVIDENCE.md` and `reports/FINAL_REPORT_DRAFT.md` still quote the previous CatBoost row and should be updated from the table above.

## Interpretation and limitations

- **Metrics:** AP is the primary metric for the 10% minority class. ROC-AUC describes ranking, Brier and log-loss describe probability quality, and recall, precision and F1 describe detection at a threshold. Accuracy alone is misleading, because the majority-class Dummy already scores 0.8999.
- **No predictive signal:** every Stage A/B result, the nested estimate and the bootstrap intervals are consistent with no predictive signal. The learning curve shows training log-loss falling while validation log-loss stays at the base-rate level, which is overfitting to noise that early stopping contains.
- **Feature importance:** the importances describe what the fold models used, not causes.
- **Synthetic data:** the dataset appears synthetic and has an independent target. It has uniform features, a deterministic salary formula, an identical hire-time timestamp and impossible hire ages.
- **Sensitive attributes:** Gender and Age are included because the contract requires them.
- **Threshold metrics:** these are descriptive.
- **Holdout:** the formal holdout was not evaluated.

## Demonstration

Open `06_m2_catboost.ipynb` and Run All. The defaults are `RUN_CATBOOST_TRAINING = False` (saved review) and `EVALUATE_HOLDOUT = False`. Saved review loads and validates the evidence with `src.evaluation.inputs.load_saved_model_evidence` and never trains or writes files. If the git-ignored model or processed file is absent, it rebuilds them in memory.

Section 13 predicts a manually entered employee in three ways:
- **Terminal:** `input()` prompts, one field at a time, with validation and re-asking.
- **Form:** an `ipywidgets` form.
- **Dictionary:** a dictionary cell.

When there is no keyboard, a recorded terminal session is replayed.

Setting `RUN_CATBOOST_TRAINING = True` retrains everything in about 25–35 minutes on a laptop, into a new `results/models/catboost/reruns/<UTC>/` folder. The canonical evidence changes only when that run is adopted by copying its files into `results/models/catboost/` and its `model/` files into `models/catboost/`. Run the preprocessing notebook first in a fresh clone if you want the processed file on disk.

Runtime versions: catboost 1.2.10, scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3.

## Observed conclusion

Strict nested CatBoost achieved mean fold AP 0.100180 and pooled AP 0.099860, against a 0.1001 prevalence reference, with ROC-AUC 0.4996. The 95% bootstrap interval of its AP gain over the Dummy Prior includes zero. No preprocessing variety or hyperparameter configuration produced a significant improvement. The small edge of the best Stage B configuration disappeared under nesting. No useful resignation discrimination is demonstrated. CatBoost's main contribution is well-calibrated base-rate probabilities, with the lowest raw Brier score of the six models, and a leakage-free reference procedure.
