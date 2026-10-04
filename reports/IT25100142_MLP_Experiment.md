# Supplementary MLP experiment — IT25100142

## Purpose
Test whether a simple feed-forward neural network ranks resignation cases differently from FT-Transformer. The supplementary notebook is stored alongside FT-Transformer in notebooks/team_contributions/models/member_1_ft_transformer/. The six-model analysis registry is unchanged.

## Dataset and preprocessing
The supplied source has 100,000 rows and 20 columns. The frozen registry selects 80,000 development rows; no rows were deleted for missing values or outliers. After feature selection the predictor matrix is (80,000, 17), with the target stored separately. Development resignation prevalence is 10.01%. Original employer provenance, salary currency and prediction horizon are unverified.

Use the 13 approved numerical/ordinal features and four categorical features. Employee_ID is used only for alignment, Hire_Date is excluded, and Resigned is the target. StandardScaler learns training means and standard deviations. OneHotEncoder learns training categories and ignores unseen validation categories. The 17 original predictors expand to 36 numerical input columns in this run; these are encodings of the same features, not new source attributes. No PCA, resampling or imputation was performed; unexpected missing values stop the run for review.

## Implementation and varieties
Scikit-learn MLPClassifier uses ReLU hidden layers and a sigmoid binary output, optimized with Adam. It learns interactions through dense layers rather than attention or embeddings. Batch size is 512, seed 42, and the maximum is 100 epochs. L2 alpha penalizes large weights; dropout is not used.

| candidate_id | hidden_layers | learning_rate | alpha | weighting |
| --- | --- | --- | --- | --- |
| M01 | 64 | 0.001000 | 0.000100 | none |
| M02 | 64 | 0.001000 | 0.000100 | balanced |
| M03 | 128 -> 64 | 0.001000 | 0.000100 | none |
| M04 | 128 -> 64 | 0.001000 | 0.000100 | balanced |
| M05 | 128 -> 64 | 0.000100 | 0.000100 | none |
| M06 | 128 -> 64 | 0.000100 | 0.000100 | balanced |
| M07 | 128 -> 64 | 0.001000 | 0.001000 | none |
| M08 | 128 -> 64 | 0.001000 | 0.001000 | balanced |

Balanced sample weights are computed from the applicable training labels: n/(2*n_class). They average one and give resignation errors about nine times the relative weight. The unweighted versions use ordinary loss. Validation AP is always evaluated without class weighting, at the original class distribution.

## Nested evaluation
Five saved outer folds each reserve 16,000 records. The other 64,000 split into 54,400 inner-training and 9,600 inner-validation records with stratification and seed 42. Each candidate fits a new preprocessor on inner training. One partial_fit call makes one training pass (epoch). AP improvement greater than 0.000001 updates the selected epoch. Fifteen consecutive epochs without that improvement stop selection; accuracy is not the stopping metric. Candidate ties select the earlier declared candidate.

After selection, a fresh preprocessor and MLP fit all 64,000 outer-training records for the selected number of epochs, without validation feedback. There are 40 candidate fits and five refits, or 45 fits in total. Every development employee receives one OOF score. Holdout features and labels are not parsed for modelling; source bytes are read only to verify identity. No final all-development deployment model or holdout evaluation is claimed.

## Recorded winners
| fold | selected_candidate | selected_epoch_count | inner_ap | average_precision |
| --- | --- | --- | --- | --- |
| 1 | M04 | 15 | 0.112762 | 0.101603 |
| 2 | M02 | 48 | 0.108484 | 0.101475 |
| 3 | M08 | 11 | 0.112011 | 0.101214 |
| 4 | M07 | 1 | 0.104140 | 0.097849 |
| 5 | M08 | 3 | 0.103904 | 0.099786 |

Inner scores support selection; outer scores measure the selected procedure. They are different evaluations. Winner learning curves show the inner phase, not the fresh refit's validation performance. Refit loss histories are saved separately. Loss scales may differ between weighted and unweighted candidates.

## Results
Pooled AP: **0.100422**; mean fold AP: **0.100385 ± 0.001593**; ROC-AUC: **0.499555**; Brier: **0.204095**. Ranking was close to chance in this evaluation, and probability error was worse than the constant-prevalence reference. The MLP did not demonstrate useful ranking performance on these inputs under this protocol.

At the fixed 0.50 threshold, precision is 0.098572, recall 0.283591, F1 0.146294, and accuracy 0.668687. Runtime for selection and refits was 10.38 minutes on this device, with numerical-library threads limited to one. Runtime is not directly comparable with previous runs on other hardware.

## Current comparison
Metrics below were recomputed from the current registered OOF prediction files on the same 80,000 development identities. Their tuning protocols differ, so this is context rather than a controlled architecture-only experiment. CatBoost uses the current promoted evidence, which differs from older comparison exports.

| Model | average_precision | roc_auc | brier_score |
| --- | --- | --- | --- |
| CatBoost | 0.102248 | 0.504103 | 0.090077 |
| XGBoost | 0.101181 | 0.502471 | 0.175329 |
| MLP (supplementary) | 0.100422 | 0.499555 | 0.204095 |
| LightGBM | 0.100153 | 0.498740 | 0.180215 |
| Constant prevalence reference | 0.100100 | 0.500000 | 0.090080 |
| TabNet | 0.099965 | 0.499317 | 0.186177 |
| EBM | 0.099000 | 0.495976 | 0.090109 |
| FT-Transformer | 0.098648 | 0.497980 | 0.153086 |

The prevalence reference is AP 0.1001, ROC-AUC 0.5 and Brier 0.09007999. Majority-only accuracy is 89.99%, with zero resignation recall. Compare ranking and probability error against these references, not accuracy alone. Small AP differences do not establish meaningful superiority. Fold SD is descriptive, not a confidence interval. No significance test has been performed for the MLP comparisons.

## Figures and evidence
Five figures show PR/ROC curves, a fixed-threshold confusion matrix, fold AP and selected epochs, measured winner training curves, and probability diagnostics. Reliability uses ten quantile bins. Weighting can change probability calibration; sigmoid scores are not automatically calibrated probabilities.

Results directory: `results/models/mlp/individual_IT25100142/run_20261004T174618Z_365bfea0`. It contains candidate_selection.csv (40 rows), fold_metrics.csv (5 rows), summary_metrics.json, training_history.json, refit_training_loss.json, model_comparison.csv, manifest.json, and local-only OOF predictions. The manifest records library version, configuration and source hashes. No employee rows or weights are embedded in the notebook.

## Limitations and improvements
This is one seed and a bounded eight-candidate search. It does not test every architecture or establish that the dataset cannot contain useful signal. Salary redundancy, possible construction patterns and unverified provenance limit interpretation. Age and gender are sensitive predictors; this experiment does not establish fairness. Better outcome definitions, verified predictors, repeated inner splits, wider regularization studies and training-only calibration are future options. Improvements must preserve validation boundaries.

## Viva explanation
“I trained eight MLP settings inside each of five rounds. Inner validation selected the setting and epoch count. I then trained a fresh winner on the full training portion and evaluated it on the reserved fold. Numerical features were standardized and categories were one-hot encoded using training data only. I compared weighted and unweighted loss, then assessed ranking, probability error and detection at a fixed threshold.”

## Source
[Scikit-learn MLPClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). The local implementation uses scikit-learn 1.9.0; weighted partial_fit needs version 1.7 or later. Shared feature and split rules come from configs/feature_contract.yaml.
