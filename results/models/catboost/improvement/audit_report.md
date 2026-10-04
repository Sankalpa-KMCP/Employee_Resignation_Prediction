# CatBoost audit report

_Rendered from machine-readable evidence by `scripts/render_catboost_audit_report.py`. Every number below is computed; none is typed._

## 1. Checklist (before → after)

| # | Item | Status (before) | Evidence | Severity | Class | After |
|---|---|---|---|---|---|---|
| 1.1 | Imports resolve; versions match requirements.txt | WARN | all 13 notebook imports resolve (`improvement/phase1/integrity.json`); mismatches: {'numpy': 'installed 2.5.2 vs pinned 2.5.1'} | Low | C/D | n/a (unchanged) |
| 1.2 | One-command, top-to-bottom runnable script | FAIL | Before the audit no script ran load→nested CV→OOF→metrics→figures; notebook cell 3 sets `RUN_CATBOOST_TRAINING = False` so a default run only re-reads saved CSVs. Created `scripts/run_catboost_audit.py`. | Medium | D | PASS (script created; see Phase 4 run log) |
| 1.3 | CPU-only deterministic training | PASS | two same-seed nested-CV runs: max abs diff = 0.0e+00; vs saved OOF = 9.7e-17; GPU used = False (`phase1/determinism.json`) | — | — | n/a (unchanged) |
| 2.1 | Canonical data path (`data/raw` + `split_registry.csv`) | WARN | cell 5 reads the raw CSV directly (`pd.read_csv(RAW_DATA_PATH)`) and merges the registry: canonical files, but bypasses the hash-verified loader `load_development_data` (uses loader: False) | Low | D | PASS (new notebook uses `load_audit_inputs`) |
| 2.2 | Row counts, folds, stratification, uniqueness, missing, dtypes | PASS | dev rows 80,000; holdout IDs in registry 20,000; overlap 0; fold sizes [16000]; fold positive rates 0.10006–0.10013 (overall 0.10010); duplicate IDs 0; missing predictor cells 0 | — | — | n/a (unchanged) |
| 2.3 | Formal holdout not read | FAIL | cell 5 builds `holdout_df = merged_df[merged_df['partition']=='holdout']` from the raw CSV, i.e. holdout predictors AND `Resigned` are parsed into memory (only row counts/IDs are used afterwards; no scoring, no plot, no decision). Violates the spirit of hard rule 1 / the protocol's ID-only inspection. | High | A/D | PASS (new notebook never builds a holdout frame) |
| 2.4 | Native categoricals, no needless scaling/encoding | PASS | `cat_features` = ['Department', 'Gender', 'Job_Title', 'Education_Level'] passed natively; dtypes numeric for the other 13; notebook applies scaler/PCA/winsor: False | — | — | n/a (unchanged) |
| 2.5 | Preprocessing-member outputs (outliers, scaling, encoding, selection, PCA) | PASS | None are applied to CatBoost. Appropriate for a tree ensemble: splits are invariant to monotone scaling, categoricals use ordered target statistics fitted inside each training fold. | — | — | n/a (unchanged) |
| 2.6 | 17 predictors equal the feature contract | PASS | notebook reads `primary_model_features` from the contract (source: feature_contract['primary_model_features']); count 17; list identical to contract (no mismatch) | — | — | n/a (unchanged) |
| 3.1 | Nested CV structure | PASS | inner split = `train_test_split` on the rows passed to `fit()` (True); early stopping `eval_set` = inner val (True); refit on 100% outer-train (True); outer-val only in `predict_proba` | — | — | n/a (unchanged) |
| 3.2 | Early-stopping set-up (PRAUC, patience 50) | FAIL | selected trees per fold [2, 2, 7, 1, 67]; PRAUC on ~960 inner positives is noisy. Probe (`phase1/early_stopping_probe.csv`):

| variant | trees per fold | PR-AUC | ROC-AUC | Brier | LogLoss |
|---|---|---|---|---|---|
| baseline_PRAUC_p50 | [2, 2, 7, 1, 67] | 0.09935 | 0.50016 | 0.21075 | 0.60521 |
| PRAUC_p200 | [2, 2, 7, 1, 257] | 0.10003 | 0.49960 | 0.23931 | 0.67159 |
| Logloss_p100 | [103, 7, 7, 26, 79] | 0.10016 | 0.49813 | 0.18576 | 0.54564 |
| Logloss_p200 | [103, 7, 7, 26, 79] | 0.10016 | 0.49813 | 0.18576 | 0.54564 |
| AUC_p100 | [2, 2, 7, 1, 257] | 0.10003 | 0.49960 | 0.23931 | 0.67159 |
| Logloss_p200_bfa | [23, 7, 7, 5, 16] | 0.10016 | 0.49737 | 0.18593 | 0.54582 |

Switching to Logloss/AUC or patience 200 changes which trees/candidates win but PR-AUC stays within 0.09935–0.10016 (prevalence 0.10010) and ROC-AUC within 0.49737–0.50016. | Medium | B | PASS: selection on `log_loss`, patience 200, 3 repeated inner splits; trees per fold [4, 15, 1, 57, 31] |
| 3.3 | Brier 0.211 for an 'unweighted' winner (priority) | FAIL | **Not a refit/weights bug.** (i) CatBoost leaves `boost_from_average` off for Logloss by default (`get_params()` reports None; explicit False gives identical output), so a model starts at p=0.5; with lr=0.03 two trees only reach mean p=0.4765 (fold-1 manual reproduction, C03 unweighted, 2 trees: Brier 0.23173). With `boost_from_average=True` the same model gives mean p=0.1001, Brier 0.09005. (ii) The nested CV did **not** use C03 in every fold: per-fold winners were fold 1: C03 (unweighted, 2 trees, mean p=0.476), fold 2: C03 (unweighted, 2 trees, mean p=0.476), fold 3: C07 (Balanced, 7 trees, mean p=0.500), fold 4: C04 (Balanced, 1 trees, mean p=0.500), fold 5: C05 (unweighted, 67 trees, mean p=0.116); the summary JSON reports only `C03`. Balanced candidates inflate probabilities toward 0.5. The saved OOF reproduces exactly from the current code (1e-16), so there is no hidden state or misalignment. | High | B (+D reporting) | PASS: raw Brier 0.09008 vs prior 0.09008; winners ['R02', 'R07', 'R09', 'R20', 'R12'] (all unweighted: True) |
| 3.4 | Pooled OOF mixes probability scales across folds | FAIL | fold-mean predicted probability ranges 0.116–0.500; pooled AP/ROC/Brier over such a mixture are not properties of any single model. Per-fold metrics are the valid unit. | Medium | B/D | PASS: fold-mean probability 0.1001–0.1002 (prevalence 0.1001) |
| 3.5 | No state carried between folds | PASS | `run_5fold_cv` constructs `adapter_factory()` per fold (found: True); the post-selection `self.params.update(...)` lives on a per-fold object; rerun equals saved OOF | — | — | n/a (unchanged) |
| 3.6 | `catboost_development_model.cbm` reproduces predictions | WARN | loads; tree count 2 = metadata 2; a refit from the stored parameters reproduces 2,000 sampled predictions (max abs diff 0.0e+00). But it is a 2-tree model with mean p=0.476: reproducible, not useful. | Medium | B | new model `models/catboost/audit/model.cbm`: 28 trees (old file untouched) |
| 4.1 | Baselines on the same folds/rows/metric code | PASS | Dummy prior, Dummy most_frequent, Logistic Regression and CatBoost raw/calibrated computed from OOF with `audit_stats.compare_models` (`phase1/baseline_comparison_before.csv`). | — | — | n/a (unchanged) |
| 4.2 | CatBoost vs baseline (verdict rule: CI lower bound > 0 and Δ ≥ 0.01) | FAIL (does not beat baseline) | ΔPR-AUC vs Dummy prior = -0.00074 (95% CI -0.00313, +0.00184); ΔROC-AUC = +0.00024 (CI -0.01000, +0.00994); ΔBrier = +0.12067; better_than_baseline = False | High (finding, not a defect) | C | better_than_baseline = False |
| 5.1 | Excluded columns absent from the feature matrix | PASS | excluded columns in feature list: none; runner raises on forbidden/post-outcome names (True) | — | — | n/a (unchanged) |
| 5.2 | Single-feature leakage / 'too good to be true' scan (all 19 non-target columns incl. Employee_ID, Hire_Date) | PASS | suspicious (AUC>0.6 or <0.4): none; largest distance of single-feature AUC from 0.5 = 0.0092 (Remote_Work_Frequency); max Cramér's V 0.0066; max MI 7.2e-05 nats | — | — | n/a (unchanged) |
| 5.3 | Duplicates / near-duplicates across train and validation | PASS | duplicate feature rows 0; cross-boundary duplicates 0; fold-1 sample nearest-neighbour distance min 0.755 (none below 1e-6) | — | — | n/a (unchanged) |
| 5.4 | Calibration cross-fitted, never scored on its own fit rows | WARN | Notebook cell 21 fits the Platt map on other folds' OOF rows (own fold excluded). Second-order caveat: those other-fold models were trained on rows that include the scored fold, so this is standard OOF cross-fitting, not nested calibration. Unit test `test_crossfit_calibration_never_uses_the_scored_fold` guards the first-order property. | Low | B | n/a (unchanged) |
| 5.5 | No statistic computed on full development data before the split | PASS | Model fitting statistics (CatBoost ordered target statistics) are computed inside `fit()` per fold. Notebook cells 7/9 compute descriptive full-development statistics (fold-support counts, univariate audit); they feed no model. | — | — | n/a (unchanged) |
| 5.6 | Label-shuffle test (≥10 permutations, full nested CV) | WARN | The previous notebook has a single shuffled-target run (Exp 8 in `catboost_leakage_safe_experiments.csv`), not ≥10. Full test is run on the final protocol in Phase 4. | Medium | D | PASS: 10 permutations (see section below) |
| 5.7 | OOF row order / labels aligned by explicit Employee_ID join | PASS | 80,000 rows joined 1:1; label mismatches 0; fold mismatches 0; ROC-AUC from CSV 0.500155 vs summary 0.500155 | — | — | n/a (unchanged) |
| 5.8 | Formal holdout never scored or used | WARN | No receipt (`results/final_holdout/assessment_receipt.json` exists: False); no `models/frozen`; `src/models/catboost_model.py` never references holdout; runner rejects holdout IDs. Residual: see 2.3 (notebook parsed holdout rows into memory). | High | A/D | PASS (audit code never loads holdout rows) |
| 6.1 | Required figures present | FAIL | Existing `figure_01…10` cover importance, PR, confusion, ROC, probability, reliability, lift, learning curves, fold scores, SHAP. Missing: target/fold balance, nested-CV layout, bootstrap Δ, candidate comparison, tree-count ablation, single-feature signal, permutation test, subgroups, per-fold strip with baselines, permutation importance, ROC bands. Existing SHAP/importance figures describe a 2-tree model. | Medium | D | PASS (17 figures; `final/figure_index.json`) |
| 6.2 | No hard-coded metrics in the notebook | FAIL | 51 typed 4+-decimal numbers in markdown cells of the previous notebook (e.g. conclusions in cells 32 and 35), plus code cells printing literal reference values. | Medium | D | PASS (new notebook computes narrative numbers) |
| 6.3 | Summary JSON consistent with per-fold evidence | FAIL | `catboost_summary_metrics.json` reports one `selected_candidate_id` (C03) and `selected_tree_count` for the whole run, while folds selected ['C03', 'C04', 'C05', 'C07']. | Medium | D | PASS (per-fold selection saved in `final/fold_selection_summary.csv`) |
| 7.1 | Feature-contract change control | WARN | `catboost_leakage_safe_experiments.csv` Exp 7 evaluated '6 engineered features'. Adding predictors is a prohibited *silent* change (`change_control.prohibited_silent_changes`). It was not adopted; no new feature is used by this audit. | Low | C | n/a (unchanged) |
| 7.2 | Test-suite baseline in this environment | WARN | Full suite before any change: see `improvement/phase1/pytest_baseline_summary.txt` — failures/errors come from missing `torch`, `interpret`, `xgboost` packages (other members' modules) and an FT notebook cell-count test, none from CatBoost files. | Low | C | n/a (unchanged) |

Class: **A** pipeline bug, **B** tuning/training-protocol weakness, **C** feature/data limitation, **D** reporting/visualisation gap.

## 2. Baseline snapshot (before)

| model | PR-AUC | 95% CI | ROC-AUC | 95% CI | Brier | Log loss | Bal. acc @0.5 |
|---|---|---|---|---|---|---|---|
| Dummy prior | 0.10009 | [0.09789, 0.10252] | 0.49992 | [0.49445, 0.50551] | 0.09008 | 0.32530 | 0.5000 |
| Dummy most_frequent | 0.10010 | [0.09811, 0.10220] | 0.50000 | [0.50000, 0.50000] | 0.10010 | 3.60797 | 0.5000 |
| Logistic Regression | 0.09962 | [0.09688, 0.10274] | 0.49842 | [0.49159, 0.50518] | 0.09014 | 0.32561 | 0.5000 |
| CatBoost raw (before) | 0.09935 | [0.09677, 0.10223] | 0.50016 | [0.49401, 0.50688] | 0.21075 | 0.60521 | 0.4980 |
| CatBoost calibrated (before) | 0.09961 | [0.09687, 0.10266] | 0.50003 | [0.49353, 0.50677] | 0.09014 | 0.32565 | 0.5000 |

Positive prevalence = 0.10010. `Dummy most_frequent` predicts a constant class, so its ranking metrics are degenerate (AUC = 0.5 by construction) and its log loss is huge because it outputs probabilities of exactly 0 or 1; use Brier, log loss and balanced accuracy for it.

Candidate pool (8) and per-fold winners are stored in `improvement/baseline_snapshot.json`.

## 3. After the improvement run

| model | PR-AUC | 95% CI | ROC-AUC | 95% CI | Brier | Log loss |
|---|---|---|---|---|---|---|
| Dummy prior | 0.10009 | [0.09789, 0.10252] | 0.49992 | [0.49445, 0.50551] | 0.09008 | 0.32530 |
| Dummy most_frequent | 0.10010 | [0.09811, 0.10220] | 0.50000 | [0.50000, 0.50000] | 0.10010 | 3.60797 |
| Logistic Regression | 0.09962 | [0.09688, 0.10274] | 0.49842 | [0.49159, 0.50518] | 0.09014 | 0.32561 |
| CatBoost (baseline raw) | 0.09935 | [0.09677, 0.10223] | 0.50016 | [0.49401, 0.50688] | 0.21075 | 0.60521 |
| CatBoost (raw) | 0.10225 | [0.09931, 0.10566] | 0.50410 | [0.49733, 0.51083] | 0.09008 | 0.32529 |
| CatBoost (platt) | 0.10226 | [0.09932, 0.10566] | 0.50413 | [0.49733, 0.51091] | 0.09008 | 0.32529 |
| CatBoost (isotonic) | 0.10155 | [0.09889, 0.10457] | 0.50498 | [0.49853, 0.51115] | 0.09014 | 0.34497 |

## 4. Label-shuffle test (final protocol)

10 permutations of the development labels, each through the complete nested protocol. Real PR-AUC 0.10225 vs shuffled mean 0.09902 (sd 0.00087); permutation p = 0.091. Real ROC-AUC 0.50410 vs shuffled mean 0.49543 (sd 0.00347); p = 0.091.

## 5. Experiment log

| id | change | PR-AUC | ROC-AUC | Brier | Log loss | decision |
|---|---|---|---|---|---|---|
| E0_baseline | current repo protocol | 0.09935 | 0.50016 | 0.21075 | 0.60521 | baseline |
| E1 | boost_from_average=True (start Logloss at the prior, not p=0.5) | 0.10004 | 0.49746 | 0.18593 | 0.54582 | reverted |
| E2 | early stopping + candidate selection on inner Log Loss, patience 200 | 0.10017 | 0.50015 | 0.09012 | 0.32550 | kept |
| E1b | boost_from_average=True re-tested on top of the current best (plan Amendment 1) | 0.09837 | 0.49308 | 0.09010 | 0.32543 | kept |
| E3a | class weights: none (3 shapes) | 0.09837 | 0.49308 | 0.09010 | 0.32543 | reverted |
| E3b | class weights: Balanced (3 shapes) | 0.10069 | 0.49835 | 0.24923 | 0.69160 | reverted |
| E3c | class weights: SqrtBalanced (3 shapes) | 0.10028 | 0.50070 | 0.11249 | 0.39744 | reverted |
| E3d | class weights: scale_pos_weight=3 (3 shapes) | 0.10006 | 0.49930 | 0.11253 | 0.39755 | reverted |
| E4 | random search, 20 trials, 3 repeated inner splits | 0.10225 | 0.50410 | 0.09008 | 0.32529 | kept |
