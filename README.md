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

- **FT-Transformer:** completed strict eight-candidate nested evaluation: 40 inner evaluations, five full outer-training winner refits, and 80,000 OOF predictions. Canonical evidence: `results/models/ft_transformer/strict_nested/`. Pooled AP 0.098648; mean fold AP 0.098065. Earlier fixed FT02 GPU results are historical.
- CatBoost: five-fold OOF evaluation and multi-candidate nested tuning evidence are present.
- XGBoost: Member 3's primary **P1 one-hot, 8-candidate × 5-outer-fold strict nested search** is the group input at `results/models/xgboost/oof_predictions.csv`. P2/P3 preprocessing comparisons are sensitivity evidence; the older independent result under `strict_nested/` is historical.
- LightGBM: strict-nested evaluation artifacts are present; the current adapter uses the supported `eval_set` API.
- EBM: completed nested five-fold development evaluation with three raw candidates and three standardized sensitivity variants; formal inputs are verified against their manifest under `results/models/ebm/individual_IT25100429/`.
- TabNet: strict-nested five-fold OOF evaluation and candidate-selection evidence are present.
- Group model comparison: completed in `notebooks/group/02_model_comparison.ipynb` and `results/comparison/`.
- Calibration diagnostics: completed development-only in `notebooks/group/03_calibration.ipynb`.
- Fairness/subgroup diagnostics: completed development-only in `notebooks/group/04_fairness_analysis.ipynb`.
- Retention prioritization: completed only as an exploratory development simulation; it is explicitly non-operational.
- Formal holdout: still locked because no practically useful final model has been frozen.
- Final report: the requested Word deliverable is `reports/FINAL_REPORT.docx`, with editable source in `reports/FINAL_REPORT_DRAFT.md`. Group ID, member identities and group confirmation of AI usage remain explicit placeholders. See `reports/README.md` for rebuilding and submission preparation.

The strongest pooled development Average Precision is the current Member 3 XGBoost primary result (**0.101181**) versus a no-skill prevalence baseline of **0.1001**. The package therefore records a **near-no-signal negative result** rather than claiming useful resignation prediction. The six-model comparison ranks pooled AP; mean outer-fold AP is also reported and has a different numerical leader.

The four models that previously lacked an explicit input-representation comparison now have a bounded, paired training exercise in `scripts/run_preprocessing_sensitivity.py`. It contrasts the full predictor set with training-median masking of `Monthly_Salary` on a 5,000/2,000 development sample. The script verifies frozen input hashes and excludes holdout rows before parsing their values. Deliberate repeats write to new `results/models/preprocessing_sensitivity/reruns/<UTC timestamp>/` directories, preserving the original evidence. These short fits are explanatory evidence only; see `reports/INDIVIDUAL_MODEL_EVIDENCE.md` for all six members' variation tables and limitations.

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

Documents under `docs/context/` preserve earlier project states. They may contain pre-refactor paths and older status statements. Use this README and `docs/ARCHITECTURE.md` for the current repository structure.

## EBM formal input reconciliation — IT25100429

The six executed varieties are now registered in `configs/models/ebm.yaml`: E01/E02/E03 with raw and standardized numerical representations. Three raw candidates participate in nested inner-AP selection; three standardized pairs are sensitivity experiments. All fixed parameters and the .001 practical AP margin match the executed experiment. The previous unexecuted search-space ranges are superseded.

The group's canonical formal EBM OOF is `results/models/ebm/individual_IT25100429/oof_predictions.csv`, paired with `fold_metrics.csv` and `summary_metrics.json` in that directory. Mean fold AP .099558; pooled AP .099000; pooled ROC-AUC .495976. This is a completed nested development evaluation, not final holdout evaluation or a frozen deployment model. Earlier EBM status tables are superseded by this update.

Top-level EBM outputs are historical default-only results and must not be loaded as the group's formal input. The shared `load_group_inputs` loader checks every model against the frozen development IDs, folds, original development targets and probability rules. It also retains EBM's config-bound artifact manifest verification, with no historical fallback. Group comparison, calibration, fairness and prioritization all use this loader.

CatBoost, XGBoost and EBM notebooks default to loading completed evidence. To deliberately repeat a workflow, change its setup flag (`RUN_CATBOOST_TRAINING`, `RUN_XGBOOST_TRAINING`, or `RUN_EBM_TRAINING`) to `True`, then restart and run all. New evidence and serialized models go inside a unique `results/models/<model>/reruns/<UTC timestamp>/` directory. The canonical registry is unchanged until a new run is explicitly adopted. Restore the flag to `False` after training for saved review.

The command-line equivalents `python scripts/run_xgboost_nested.py` and `python scripts/run_ebm_nested.py` also default to saved review; add `--train` to deliberately create a new run. The EBM script performs 60 experimental fits; its notebook additionally performs the explanation fit. The prior eight-candidate EBM launcher is superseded.

The TabNet and strict LightGBM inventories describe files available in this checkout. TabNet's 18 absent serialized artifacts are marked unavailable, with blank current hashes and sizes; its saved predictions remain usable. These inventories do not certify that unavailable model weights can be reloaded.

Completed formal FT update: see `reports/Six_Model_Comparison.md` for the current six-member discussion. The group notebook retains main's declared pooled-AP ranking and separately reports mean fold AP. The existing Word export predates this update; rebuild it from the updated Markdown before submission.
