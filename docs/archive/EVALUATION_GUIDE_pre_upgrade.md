# Current Evaluation Guide

This guide describes the current refactored repository. Historical evaluation notes are preserved in `docs/archive/EVALUATION_GUIDE_pre_refactor.md`.

## Assessment-facing evidence map

### Shared foundations

- `notebooks/01_data_audit.ipynb`
- `notebooks/02_create_split_registry.ipynb`
- `notebooks/03_eda.ipynb`
- `notebooks/04_baselines.ipynb`
- `notebooks/04a_pca_analysis.ipynb`
- `notebooks/group/01_group_pipeline.ipynb`

### Individual headline models

| Member | Model | Notebook | Current evidence status |
|---|---|---|---|
| 1 | FT-Transformer | `notebooks/team_contributions/models/member_1_ft_transformer/05_m1_ft_transformer.ipynb` | 5-fold results + tuning artifacts present; protocol/config consistency needs cleanup |
| 2 | CatBoost | `notebooks/team_contributions/models/member_2_catboost/06_m2_catboost.ipynb` | 5-fold OOF + candidate tuning evidence present |
| 3 | XGBoost | `notebooks/team_contributions/models/member_3_xgboost/07_m3_xgboost.ipynb` | 5-fold OOF present; genuine multi-configuration tuning still required |
| 4 | LightGBM | `notebooks/team_contributions/models/member_4_lightgbm/08_m4_lightgbm.ipynb` | strict-nested artifacts present; reviewed sklearn `fit()` API mismatch repaired and focused tests pass |
| 5 | EBM | `notebooks/team_contributions/models/member_5_ebm/09_m5_ebm.ipynb` | 5-fold OOF present; genuine multi-configuration tuning still required |
| 6 | TabNet | `notebooks/team_contributions/models/member_6_tabnet/10_m6_tabnet.ipynb` | strict-nested 5-fold OOF + candidate-selection evidence present |

### Group evaluation

- `notebooks/group/02_model_comparison.ipynb` — must become the authoritative comparison of the six final individual model results.
- `notebooks/group/03_calibration.ipynb` — optional/supporting probability calibration work as justified.
- `notebooks/group/04_fairness_analysis.ipynb` — supports ethical considerations/bias analysis.
- `notebooks/group/05_retention_prioritization.ipynb` — optional downstream simulation; lower priority than required assessment work.
- `notebooks/group/06_final_holdout_evaluation.ipynb` — keep locked until the final development pipeline is frozen.

## Viva checklist per member

Each member should be able to explain:

1. why the model suits the dataset/problem;
2. libraries/classes/functions used;
3. preprocessing required specifically by that model;
4. hyperparameters and how variants were selected;
5. how many configurations/variants were trained;
6. validation method and leakage controls;
7. evaluation metrics and why they are appropriate;
8. comparison across model variants;
9. limitations and possible improvements;
10. live code/output evidence.

## Group checklist

Before the final viva/report:

1. finish XGBoost tuning;
2. finish EBM tuning;
3. retain the repaired LightGBM source and re-run the formal workflow only when you intentionally regenerate results;
4. clean FT notebook/config consistency;
5. finish the six-model comparison notebook using aligned OOF predictions;
6. finish a defensible fairness/ethics analysis;
7. prepare the final report in `reports/`;
8. update presentation assets/deck in `presentations/`;
9. only then freeze the pipeline and decide whether to execute the final holdout once.
