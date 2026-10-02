# Employee Resignation Prediction — IT2011 AI/ML Project

This repository contains the group AI/ML project for employee-resignation prediction. The project uses a frozen development/holdout split, five development CV folds, six individually owned headline models, and shared group evaluation infrastructure.

## Repository structure

```text
Employee_Resignation_Prediction/
├── README.md
├── requirements.txt
├── configs/                 # feature contract, split rules, model configs
├── data/
│   ├── raw/                 # assigned source dataset
│   └── processed/           # frozen split registry / processed artifacts
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_create_split_registry.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_baselines.ipynb
│   ├── 04a_pca_analysis.ipynb
│   ├── team_contributions/  # individually attributable work
│   │   ├── preprocessing/   # preprocessing contributions named by IT number
│   │   └── models/          # six member-owned headline model notebooks
│   └── group/               # integrated/group-level evaluation notebooks
├── src/                     # reusable project implementation
├── results/                 # generated evidence and model outputs
├── models/                  # serialized final model artifacts when produced
├── reports/                 # final report deliverables
├── presentations/           # viva/presentation deliverables
├── docs/                    # methodology, context, architecture, review notes
└── tests/                   # automated validation/tests
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for design rules and [`notebooks/team_contributions/README.md`](notebooks/team_contributions/README.md) for the individual-contribution map.

## Headline model ownership

| Member | Model | Notebook |
|---|---|---|
| Member 1 | FT-Transformer | `notebooks/team_contributions/models/member_1_ft_transformer/05_m1_ft_transformer.ipynb` |
| Member 2 | CatBoost | `notebooks/team_contributions/models/member_2_catboost/06_m2_catboost.ipynb` |
| Member 3 | XGBoost | `notebooks/team_contributions/models/member_3_xgboost/07_m3_xgboost.ipynb` |
| Member 4 | LightGBM | `notebooks/team_contributions/models/member_4_lightgbm/08_m4_lightgbm.ipynb` |
| Member 5 | Explainable Boosting Machine (EBM) | `notebooks/team_contributions/models/member_5_ebm/09_m5_ebm.ipynb` |
| Member 6 | TabNet | `notebooks/team_contributions/models/member_6_tabnet/10_m6_tabnet.ipynb` |

The repository does **not** establish which preprocessing IT number maps to which Member 1–6 role, so that mapping is intentionally not inferred here.

## Current verified implementation state

The upgraded package contains executed development OOF evidence for all six headline models and completed group-stage development analysis without accessing the locked formal holdout.

- FT-Transformer: saved five-fold OOF evidence and controlled screening are present; the final strict-nested eight-candidate workflow has **not** been independently reproduced in this package.
- CatBoost: five-fold OOF evaluation and multi-candidate nested tuning evidence are present.
- XGBoost: an upgraded **8-candidate × 5-outer-fold strict nested search** is present under `results/models/xgboost/strict_nested/`.
- LightGBM: strict-nested evaluation artifacts are present; the current adapter uses the supported `eval_set` API.
- EBM: completed six-variety evaluation with nested selection among three raw candidates and three paired standardization sensitivities. Canonical formal results: `results/models/ebm/individual_IT25100429/`. Mean fold AP .099558; pooled AP .099000; ROC-AUC .495976. Top-level EBM files are historical defaults.
- TabNet: strict-nested five-fold OOF evaluation and candidate-selection evidence are present.
- Group model comparison: completed in `notebooks/group/02_model_comparison.ipynb` and `results/comparison/`.
- Calibration diagnostics: completed development-only in `notebooks/group/03_calibration.ipynb`.
- Fairness/subgroup diagnostics: completed development-only in `notebooks/group/04_fairness_analysis.ipynb`.
- Retention prioritization: completed only as an exploratory development simulation; it is explicitly non-operational.
- Formal holdout: still locked because no practically useful final model has been frozen.
- Final report: evidence-backed draft content is in `reports/FINAL_REPORT_DRAFT.md`; the official Group ID is not available in the repository, so the required final Group-ID-named PDF cannot be generated accurately yet.

The strongest pooled development Average Precision is the upgraded nested XGBoost result (~0.1011) versus a no-skill prevalence baseline of ~0.1001. The package therefore records a **near-no-signal negative result** rather than claiming useful resignation prediction.

See `docs/ASSIGNMENT_ALIGNMENT.md` and `docs/FINAL_VALIDATION_STATUS.md` for the current assessment/readiness state.

## Recommended notebook order

1. `01_data_audit.ipynb`
2. `02_create_split_registry.ipynb`
3. `03_eda.ipynb`
4. `04_baselines.ipynb`
5. `04a_pca_analysis.ipynb`
6. Individual work under `notebooks/team_contributions/`
7. `notebooks/group/01_group_pipeline.ipynb`
8. `notebooks/group/02_model_comparison.ipynb`
9. `notebooks/group/03_calibration.ipynb` as required
10. `notebooks/group/04_fairness_analysis.ipynb`
11. `notebooks/group/05_retention_prioritization.ipynb` if retained for the project scope
12. `notebooks/group/06_final_holdout_evaluation.ipynb` only after formal freeze

## Source-of-truth rule

- **Model configuration:** `configs/models/*.yaml`
- **Reusable implementation:** `src/`
- **Individual explanation/demonstration:** `notebooks/team_contributions/`
- **Group integration/comparison:** `notebooks/group/`
- **Evidence/results:** `results/`

Notebooks should orchestrate and explain the implementation rather than become a second independent implementation source.

## Historical documentation

Documents under `docs/context/` and `docs/archive/` preserve earlier project states. They may contain pre-refactor paths and older status statements. Use this README and `docs/ARCHITECTURE.md` for the current repository structure.

### IT25100429 contribution on the refactored branch

The completed notebooks use `notebooks/team_contributions/preprocessing/IT25100429_Standardization.ipynb` and `notebooks/team_contributions/models/member_5_ebm/09_m5_ebm.ipynb`. The experiment reads `configs/models/ebm.yaml`; `scripts/run_ebm_nested.py` invokes that same six-variety workflow. Group analysis uses the canonical manifest-validated EBM OOF via `configs/analysis_registry.yaml`. The Windows CRLF fix and manual Windows CI workflow are restored; native Windows verification remains pending. Assessment and viva guidance is in `reports/IT25100429_Individual_Contribution.md`.
