# Six-member model comparison and performance discussion

## Comparison of the best results found by each member
All six headline models have completed bounded nested-development evaluations in the canonical analysis registry. Each uses the same 80,000 development IDs and five frozen outer folds. Candidate selection, preprocessing and early stopping occur inside outer training. The formal 20,000-row holdout remains reserved.

“Optimum” here means the best configuration found inside each member's declared bounded search, not a mathematically proven global optimum. Fold winners may differ, so the primary score evaluates the full tuning procedure rather than a single universal configuration. EBM's standardized variants are a sensitivity experiment; its three raw candidates define its primary nested selection.

This table ranks **mean fold Average Precision**; main's group notebook retains its declared pooled-AP primary ranking, with XGBoost first. Both aggregation rules are explicitly reported. Pooled AP is retained separately and can give a different ranking because probabilities from independently trained folds may have different scales. Fold SD describes variation and is not a confidence interval.

| Rank by mean AP | Member | Model | Mean fold AP ± SD | Pooled AP | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| 1 | 6 | TabNet | 0.101700 ± 0.003876 | 0.099965 | 0.499317 | 0.186177 |
| 2 | 3 | XGBoost | 0.101643 ± 0.000990 | 0.101181 | 0.502471 | 0.175329 |
| 3 | 2 | CatBoost | 0.100690 ± 0.002784 | 0.099346 | 0.500155 | 0.210753 |
| 4 | 4 | LightGBM | 0.100202 ± 0.002446 | 0.100153 | 0.498740 | 0.180215 |
| 5 | 5 | EBM | 0.099558 ± 0.002073 | 0.099000 | 0.495976 | 0.090109 |
| 6 | 1 | FT-Transformer | 0.098065 ± 0.002638 | 0.098648 | 0.497980 | 0.153086 |

The numerical leader is **TabNet**, with mean fold AP **0.101700**. The positive prevalence/chance AP reference is approximately **.1001**, and chance ROC-AUC is **.5**. A numerical lead does not by itself demonstrate useful prediction or a meaningful difference.

## Challenges, observed performance and expected behavior

- **Class imbalance:** approximately 90% non-resigned employees make accuracy a weak standalone measure. A majority-only classifier can achieve about 90% accuracy while missing every resignation. AP, recall and F1 expose that failure.
- **Weak predictive signal:** the models must be judged against chance references. Increasing model complexity cannot create information absent from the available predictors. These results do not establish which workplace factors cause resignation.
- **FT-Transformer:** numerical feature tokens and categorical embeddings permit nonlinear interactions. Its eight-candidate search explores depth, width, heads, learning rate and class weighting, with inner AP selecting both configuration and epoch count. It adds neural training cost and hardware-dependent variation; loss weighting can distort raw probability calibration.
- **CatBoost:** native categorical handling avoids unnecessary one-hot encoding. Its boosted trees can capture nonlinear interactions, but categorical handling and extra capacity do not guarantee performance above chance.
- **XGBoost:** boosted trees provide flexible nonlinear decision boundaries. The canonical primary representation and preprocessing sensitivity experiments must stay distinct; the primary group score uses its registered nested result.
- **LightGBM:** efficient boosting with fold-local categorical handling offers a strong tabular comparator. Tree complexity and class weighting must be selected inside training rather than optimized against outer scores.
- **EBM:** additive feature functions and limited interactions support readable model explanations. Three parameter settings were paired with raw and standardized numerical inputs; the scaling pairs produced identical OOF predictions. EBM's primary nested estimate is near chance. Explanations describe the fitted model, not causal effects or a guarantee of useful prediction.
- **TabNet:** attentive feature selection allows nonlinear tabular modeling, but small metric differences do not prove the learned attention is a meaningful resignation explanation. Neural seed and tuning variability should be acknowledged.
- **Fold score scales:** independently selected weighted and unweighted winners can produce different probability scales. This affects pooled AP and pooled risk rankings; mean fold AP avoids comparing scores across folds for the primary metric. Fold-aware calibration requires a separate development-only design.
- **Calibration and thresholds:** Brier measures probability error, while AP/ROC-AUC measure ranking. Better probability error can result from simply predicting the low base rate. The fixed .50 threshold supports comparison; maximizing F1 on pooled OOF is exploratory and should not be reported as independent threshold-validation evidence.
- **Unequal search budgets:** the same outer folds allow fair evaluation of each declared procedure, but different candidate counts, regularization and training costs prevent claiming identical search effort. Report each search budget and the saved selection evidence.

## Conclusion for the group assessment
Present the table, explain the primary ranking, compare it against the chance baseline, then discuss minority-class detection and probability quality. Prefer a supported conclusion about the observed limitations over claiming a useful HR predictor solely because one model ranks first. The group must still decide whether any model has sufficient practical value, finalize calibration and threshold choices, and freeze the whole predictive pipeline before a single final holdout corroboration.

## Evidence and reproduction
`configs/analysis_registry.yaml` identifies all six formal OOF sources. The comparison is executed in `notebooks/group/02_model_comparison.ipynb`; calibration, fairness and prioritization diagnostics are in the subsequent group notebooks. FT-Transformer's new formal evidence is in `results/models/ft_transformer/strict_nested/`, including 40 candidate selections, five outer-fold selections, runtime/input provenance, and 80,000 OOF predictions. The loaders verify portable checksums for FT and EBM and reject historical fallback.
