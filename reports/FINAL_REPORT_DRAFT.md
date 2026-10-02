# IT2011 — Artificial Intelligence and Machine Learning
## Final Group Report Draft: Employee Resignation Prediction

**Group ID:** NEEDS VERIFICATION — the supplied repository does not establish the official group ID. Replace this line and use the Group ID as the final PDF filename before submission.

## 1. Introduction and Problem Statement

This project investigates whether the supplied employee-performance and productivity dataset contains enough information to predict employee resignation reliably. The primary academic objective is to compare multiple AI/ML approaches under a common leakage-controlled evaluation design rather than to force a deployable result. A secondary development-only simulation explores how risk scores might behave under constrained retention-review capacity; it is explicitly non-causal and non-operational.

The classification target is `Resigned`. Because only about 10.01% of records are positive, Average Precision (AP) is treated as the primary ranking metric, supported by ROC-AUC, Brier score, threshold metrics, fold stability and subgroup diagnostics.

## 2. Dataset Description

The canonical raw file is `data/raw/Extended_Employee_Performance_and_Productivity_Data.csv`.

- Rows: **100,000**
- Columns: **20**
- Positive resignation rate: **10.01%**
- Missing values: **0**
- Duplicate full rows: **0**
- Identifier: `Employee_ID`
- Categorical predictors: `Department`, `Gender`, `Job_Title`, `Education_Level`
- Numerical predictors: 13 approved numerical features defined in `configs/feature_contract.yaml`
- Excluded from model inputs: `Employee_ID` and raw `Hire_Date`

The project records the supplied dataset source as the Kaggle listing `mexwell/employee-performance-and-productivity-data`. The repository does not establish original organizational provenance, so the data must not be represented as records from a verified real employer.

## 3. Preprocessing and EDA

The repository contains individually attributable preprocessing notebooks under `notebooks/team_contributions/preprocessing/` and an integrated group pipeline under `notebooks/group/01_group_pipeline.ipynb`.

The shared audit verifies schema, row count, target distribution, missingness, duplicates and identifier integrity. Outliers are reviewed with the IQR rule rather than automatically deleted. Nominal variables are encoded in model-appropriate ways, scaling is applied where model families require it, and feature redundancy is reviewed before modeling. PCA is investigated separately on standardized numerical predictors and compared with a Logistic Regression benchmark; PCA is not forced onto models whose native representation does not require it.

EDA indicates that the available predictors have weak direct relationships with resignation. This becomes important when interpreting the near-baseline model results: increasing model complexity does not create information absent from the available predictors.

## 4. Model Design and Implementation

Six headline models are individually owned:

| Member role | Headline model | Main representation / implementation |
|---|---|---|
| Member 1 | FT-Transformer | Deep tabular transformer with numerical and categorical tokenization |
| Member 2 | CatBoost | Native categorical gradient boosting |
| Member 3 | XGBoost | Fold-local one-hot encoding with histogram tree boosting |
| Member 4 | LightGBM | Fold-local categorical mapping with leaf-wise gradient boosting |
| Member 5 | Explainable Boosting Machine (EBM) | Interpretable additive boosting with interactions |
| Member 6 | TabNet | Attention-based deep tabular network with fold-local preprocessing |

Shared baselines include Dummy classifiers, and PCA is benchmarked with Logistic Regression. The frozen split registry reserves **80,000** development rows and **20,000** formal holdout rows. The development partition is divided into five saved outer folds.

### Hyperparameter tuning status

- **CatBoost:** multi-candidate nested development selection evidenced.
- **XGBoost:** upgraded eight-candidate nested development selection evidenced in `results/models/xgboost/strict_nested/`.
- **LightGBM:** strict nested candidate selection evidenced.
- **TabNet:** strict nested candidate selection evidenced.
- **FT-Transformer:** saved five-fold OOF results and earlier controlled screening are present, but the final strict-nested eight-candidate adapter run has not been independently reproduced in this package.
- **EBM:** completed six-variety evaluation; primary nested selection uses three raw candidates, with three standardized sensitivity pairs. Formal inputs are in `results/models/ebm/individual_IT25100429/`.

## 5. Evaluation and Comparison

All six canonical development OOF files cover the same 80,000 development employees with no formal-holdout overlap. The current comparison is:

| Model | Pooled AP | ROC-AUC | Brier | CV AP mean ± SD |
|---|---:|---:|---:|---:|
| XGBoost | 0.101115 | 0.504539 | 0.239092 | 0.101350 ± 0.001942 |
| LightGBM | 0.100153 | 0.498740 | 0.180215 | 0.100202 ± 0.002446 |
| FT-Transformer | 0.099979 | 0.497254 | 0.247413 | 0.099696 ± 0.002237 |
| TabNet | 0.099965 | 0.499317 | 0.186177 | 0.101700 ± 0.003876 |
| CatBoost | 0.099346 | 0.500155 | 0.210753 | 0.100690 ± 0.002784 |
| EBM | 0.099000 | 0.495976 | 0.090109 | 0.099558 ± 0.002073 |

The development prevalence/no-skill AP is approximately **0.1001**. XGBoost is the numerical pooled-AP leader after the upgraded nested search, but its AP uplift above prevalence is only about **0.0010**. This difference is too small to support a claim of meaningful predictive usefulness, especially while FT-Transformer still requires its final strict-nested rerun.

Calibration diagnostics are saved in `results/calibration/`. EBM has the smallest raw Brier score because its probabilities stay close to the low base rate; this is a calibration property and does not make it the strongest discriminator.

The project therefore records a **negative empirical result**: across six substantially different model families, development discrimination is effectively near chance/no-skill. No final predictive model is frozen and the 20,000-row formal holdout remains locked.

## 6. Ethical Considerations and Bias Mitigation

This work is intended only for academic model comparison and simulated decision analysis. It must not be used for real hiring, firing, promotion, surveillance, discipline or automated retention decisions.

Key risks include false positives that could stigmatize employees, false negatives that could miss employees needing support, direct or proxy discrimination through sensitive attributes, false precision from weak probabilities, and causal over-interpretation of feature importance. Dataset realism and provenance are also uncertain.

Development-only subgroup diagnostics are provided for `Gender` and age bands `22-29`, `30-39`, `40-49`, and `50-60` in `results/fairness/`. Because overall prediction is near no-skill, these subgroup statistics are descriptive and do not establish that the system is fair or suitable for operational use.

An optional 10%/20%/30% retention-capacity simulation is preserved in `results/prioritization/`. At the primary 20% capacity, risk-only selection has lift close to 1.0, which is consistent with the weak predictive signal. The value proxy is an academic synthetic index and must not be interpreted as employee worth.

## 7. AI Tool Usage Declaration and Transparency

Project records document AI assistance including ChatGPT and Antigravity for requirement interpretation, architecture planning, code review/debugging, model-methodology discussion, documentation support, artifact packaging and review. AI-assisted outputs were subject to repository tests, source/result cross-checking and user/group ownership. The exact final declaration should be reviewed by the group against the full AI-usage log in `docs/context/Employee_Resignation_Prediction_AI_Project_Context_v4_6_2.md` before submission.

The group remains responsible for understanding, verifying and defending all submitted code, results and explanations during the viva.

## 8. Reflections and Lessons Learned

The most important lesson is that model sophistication cannot compensate for weak predictive information. Tree boosting, interpretable additive modeling and deep tabular architectures all converge on approximately no-skill discrimination. This is a valid result when supported by consistent splits, OOF evaluation and honest comparison.

The project also demonstrates why leakage control and reproducibility matter: a high-looking score would be misleading if produced by holdout reuse, target leakage or inconsistent rows. The frozen split registry, common OOF contracts and development-only tuning help prevent those errors.

Remaining limitations are explicit: FT-Transformer still requires a final strict-nested rerun in a compatible environment, EBM remains near chance despite completed tuning, and the final report identity fields must be supplied by the group. Until those points are resolved, the formal holdout should remain untouched.

## 9. References

1. Faculty of Computing. *IT2011 — Artificial Intelligence and Machine Learning: Group Assignment Specification*. Year 2 Semester 1, 2026.
2. Faculty of Computing. *IT2011 — Artificial Intelligence and Machine Learning: Final Evaluation — Implementation*. Year 2 Semester 1, 2026.
3. Faculty of Computing. *IT2011 — Artificial Intelligence and Machine Learning: Final Evaluation — Documentation*. Year 2 Semester 1, 2026.
4. Kaggle dataset listing: `mexwell/employee-performance-and-productivity-data` (repository-recorded source pointer; original organizational provenance not independently established by this project package).
5. Official documentation for scikit-learn, XGBoost, LightGBM, CatBoost, InterpretML, PyTorch/RTDL and pytorch-tabnet, as applicable to the implementations used.
