# Individual model variation evidence for the final viva

This sheet connects each member's model, input preparation, varied parameters, measured performance, validation design and conclusion. It distinguishes the **formal five-fold development evidence** used in the six-model comparison from an **exploratory paired exercise**. The exploratory exercise is reproducible with `python scripts/run_preprocessing_sensitivity.py`; its raw 16-row table and protocol are in `results/models/preprocessing_sensitivity/`. It cannot replace full development OOF estimates or select a new group winner.

The paired exercise used the same stratified sample of **5,000 training rows from frozen folds 2-5** and **2,000 untouched validation rows from frozen fold 1** for every setting. No formal holdout IDs were used. `full_17` retained all approved predictors. `salary_train_median` replaced `Monthly_Salary` with the median computed **only on the 5,000 training rows**, then applied that value to both sets. This masks the feature while leaving each model's input schema unchanged. It is a simple feature-selection sensitivity check motivated by the salary/performance redundancy noted in EDA. The candidates were predeclared; validation results were not used to change the canonical pipelines. FT-Transformer and TabNet were capped at five epochs, and CatBoost/LightGBM at 120 trees, so their values are **not comparable** with the full-run headline values. The positive rate was **0.1002** in training and **0.1000** in validation.

| Member | Formal group input | Formal pooled AP | Formal mean fold AP | Individual preprocessing evidence | Main limitation |
|---|---|---:|---:|---|---|
| 1: FT-Transformer | GPU fixed FT02 OOF; inner epoch selection, fresh inner-training fit | 0.099842 | 0.100396 | Standard scaling/category mappings, plus paired salary masking below | Historical screening is descriptive; optional eight-candidate search not executed; original fold diagnostics unavailable. |
| 2: CatBoost | Nested OOF | 0.099346 | 0.100690 | Native categories, plus paired salary masking below | Per-fold candidate-choice log is not separately exported. |
| 3: XGBoost | Member 3 P1 one-hot nested OOF | 0.101181 | 0.101643 | P1/P2/P3 full nested preprocessing variants | Differences remain tiny relative to no-skill. |
| 4: LightGBM | Strict nested OOF | 0.100153 | 0.100202 | Fold-fitted categorical handling, plus paired salary masking below | Historic pooled candidate selection is separate from formal nested evidence. |
| 5: EBM | Individual IT25100429 nested raw OOF | 0.099000 | 0.099558 | Three raw/standardized candidate pairs | Standardization produced no change in saved predictions. |
| 6: TabNet | Strict nested OOF | 0.099965 | 0.101700 | Fold-fitted categorical codes/scaling, plus paired salary masking below | The paired check has only one small validation sample. |

**Member 1, FT-Transformer.** The canonical GPU FT02 OOF uses inner-training numerical standardization, categorical mappings and dynamic weighting; inner validation selects epochs, then a fresh model fits those inner-training rows. Pooled AP is **0.099842**, ROC-AUC **0.498905**, Brier **0.247200** and runtime **8.52 minutes**. The historical unweighted fixed baseline had pooled AP **0.101132** and Brier **0.090651**; the weighted fixed baseline had AP **0.098820** and Brier **0.251497**; the historical weighted early-stopping result had AP **0.099979** and Brier **0.247413**. Those are different training/weighting protocols, so the paired input experiment below isolates a simpler preprocessing change on a common sample. `FT01` and `FT02` share three blocks, width 64, four heads and learning rate 0.001; `FT02` changes the class-loss weight from none to the training-fold negative/positive ratio.

| Exploratory profile | Input preprocessing | AP | ROC-AUC | Brier |
|---|---|---:|---:|---:|
| FT01, unweighted | Full 17 | 0.101889 | 0.501499 | 0.090237 |
| FT01, unweighted | Salary masked | 0.096129 | 0.488929 | 0.090096 |
| FT02, dynamically weighted | Full 17 | 0.102254 | 0.495831 | 0.279441 |
| FT02, dynamically weighted | Salary masked | 0.101808 | 0.479750 | 0.209849 |

On this small sample, salary masking did not reliably improve ranking; dynamic weighting strongly changed probability error. The group input is the completed GPU run `FT02_20261003T025004Z`, which the notebook loads read-only by default. The completed scope is fixed FT02 with epoch selection; its optional eight-candidate extension was not run. Original per-fold adapter diagnostics were not exported and cannot be reconstructed as measured evidence. Evidence: `notebooks/team_contributions/models/member_1_ft_transformer/05_m1_ft_transformer.ipynb`, `configs/models/ft_transformer.yaml`, the canonical GPU run directory and the three preserved historical FT result directories.

**Member 2, CatBoost.** The formal model handles four categorical features natively and examines eight configured candidates inside development folds. The paired exercise compares `C01` and `C02` at a shared 120-tree cap. Both use depth 6, learning rate 0.05 and L2 3; `C02` changes `auto_class_weights` from none to Balanced.

| Exploratory profile | Input preprocessing | AP | ROC-AUC | Brier |
|---|---|---:|---:|---:|
| C01, unweighted | Full 17 | 0.112278 | 0.529394 | 0.089940 |
| C01, unweighted | Salary masked | 0.104993 | 0.504589 | 0.090101 |
| C02, Balanced | Full 17 | 0.111948 | 0.518097 | 0.236119 |
| C02, Balanced | Salary masked | 0.104978 | 0.509853 | 0.235698 |

Salary masking lowered AP on this one sample; balancing substantially increased Brier error. These short-run AP values are not the formal eight-candidate nested estimate. The eight-candidate demonstration and formal OOF appear in `notebooks/team_contributions/models/member_2_catboost/06_m2_catboost.ipynb`; the missing all-fold candidate log remains an evidence limitation.

**Member 3, XGBoost.** The active notebook runs all eight `X01`-`X08` candidate settings separately within each outer training fold. They vary depth/learning rate/regularization and weighted versus unweighted training. The following three *full five-fold nested* runs compare preprocessing while keeping the predeclared primary P1 pipeline as the group input. P2/P3 outer results are descriptive and did not replace P1 after inspection.

| Preprocessing variety | Candidate selection | Pooled AP | Mean fold AP | ROC-AUC | Brier |
|---|---|---:|---:|---:|---:|
| P1: fold-local one-hot, all 17 predictors | X05,X08,X05,X06,X06 | 0.101181 | 0.101643 | 0.502471 | 0.175329 |
| P2: native categorical splits | X08,X08,X06,X06,X08 | 0.101333 | 0.102083 | 0.502234 | 0.242242 |
| P3: one-hot, salary excluded | X05,X02,X06,X01,X05 | 0.100068 | 0.100092 | 0.499302 | 0.153232 |

P2 has a slightly higher descriptive AP but markedly worse Brier than P1; the differences are tiny and all AP results sit near prevalence **0.1001**. The group registry deliberately uses P1. Evidence: `notebooks/team_contributions/models/member_3_xgboost/07_m3_xgboost.ipynb`, `configs/models/xgboost.yaml`, and `results/models/xgboost/preprocessing_comparison.csv`.

**Member 4, LightGBM.** The formal model builds categorical mappings inside each training fold and selects among eight `LGBM_F01`-`LGBM_F08` candidates using inner AP. The paired exercise uses the F01 and F02 parameter templates at 120 trees. Both use learning rate 0.05, 31 leaves, depth 6 and L2 1; F02 changes `class_weight` from none to balanced.

| Exploratory profile | Input preprocessing | AP | ROC-AUC | Brier |
|---|---|---:|---:|---:|
| F01, unweighted | Full 17 | 0.097906 | 0.487425 | 0.092691 |
| F01, unweighted | Salary masked | 0.097761 | 0.480206 | 0.092620 |
| F02, balanced | Full 17 | 0.090886 | 0.463419 | 0.174084 |
| F02, balanced | Salary masked | 0.100045 | 0.469414 | 0.175612 |

Effects depend on the weighting profile and the sample is small; no preprocessing choice is made from it. Use `results/models/lightgbm/strict_nested/` for formal five-fold performance and `notebooks/team_contributions/models/member_4_lightgbm/08_m4_lightgbm.ipynb` for the historical/formal distinction.

**Member 5, EBM (IT25100429).** The formal raw-input model selects among E01/E02/E03 with an inner AP rule on every outer fold. It also evaluates the same three parameter settings after fold-local numerical standardization as a sensitivity check. The following parameter table is generated from `configs/models/ebm.yaml`; E02 uses two leaves and E03 uses three.

| Candidate | Preprocessing | Learning rate | Max leaves | Min samples/leaf | Interactions |
| --- | --- | --- | --- | --- | --- |
| E01_raw | raw | 0.015 | 2 | 10 | 0 |
| E01_standardized | standardized | 0.015 | 2 | 10 | 0 |
| E02_raw | raw | 0.03 | 2 | 4 | 5 |
| E02_standardized | standardized | 0.03 | 2 | 4 | 5 |
| E03_raw | raw | 0.005 | 3 | 2 | 10 |
| E03_standardized | standardized | 0.005 | 3 | 2 | 10 |

| Candidate/preprocessing | Pooled AP | Mean fold AP | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| E01 raw | 0.098714 | 0.099570 | 0.495909 | 0.090093 |
| E01 standardized | 0.098714 | 0.099570 | 0.495909 | 0.090093 |
| E02 raw | 0.097735 | 0.097884 | 0.492520 | 0.090129 |
| E02 standardized | 0.097735 | 0.097884 | 0.492520 | 0.090129 |
| E03 raw | 0.098626 | 0.099097 | 0.494396 | 0.090120 |
| E03 standardized | 0.098626 | 0.099097 | 0.494396 | 0.090120 |

Standardization changed none of the saved paired OOF probabilities. The formal inner-selected raw result (AP **0.099000**) is different from any one fixed-candidate row; do not select a candidate by its outer OOF score. Evidence: `reports/IT25100429_Individual_Contribution.md`, `notebooks/team_contributions/models/member_5_ebm/09_m5_ebm.ipynb`, and `results/models/ebm/individual_IT25100429/candidate_comparison.csv`.

**Member 6, TabNet.** The formal adapter fits categorical codes, median imputation and numerical standardization inside each training fold and searches eight configurations. The paired exercise uses T01/T02 with the same architecture (widths 8, three steps, learning rate 0.02) and changes `class_balance` from false to true. Each fit was capped at five epochs.

| Exploratory profile | Input preprocessing | AP | ROC-AUC | Brier |
|---|---|---:|---:|---:|
| T01, unweighted | Full 17 | 0.098768 | 0.477739 | 0.094554 |
| T01, unweighted | Salary masked | 0.109992 | 0.496021 | 0.095717 |
| T02, class balanced | Full 17 | 0.105518 | 0.507144 | 0.220316 |
| T02, class balanced | Salary masked | 0.108103 | 0.539664 | 0.225330 |

The one-sample response to salary masking differs by class-balance profile, and weighted scores have higher Brier error. This does not supersede the full nested five-fold TabNet estimate. Evidence: `notebooks/team_contributions/models/member_6_tabnet/10_m6_tabnet.ipynb`, `configs/models/tabnet.yaml`, and `results/models/tabnet/candidate_selection.csv`.

Across the six formal group inputs, model-family differences are minute relative to the no-skill prevalence benchmark. The paired exercise helps each student explain an input-preparation choice and its limitations. It does not establish that the salary feature helps or hurts reliably, nor that a short-run configuration is the best model. The official name/IT-number correspondence beyond the verified Member 5 record still requires group confirmation before submission.
