# Employee Resignation Prediction — External Evaluation Guide

This guide is intended for an AI reviewer, project evaluator, or technical auditor examining the **Employee Resignation Prediction** repository.

The repository contains the current project codebase, configuration files, analytical notebooks, tests, documentation, and the experimental artifacts generated **so far**.

The project is **not yet complete**. Reviewers must distinguish between:

* implemented code,
* notebook scaffolding,
* smoke-tested components,
* formally executed experiments,
* saved development results,
* and planned downstream stages.

The presence of a notebook, model adapter, YAML configuration, or result directory does **not** by itself mean that the corresponding experiment has been completed.

---

## 1. Project Overview & Objective

The project investigates whether employee resignation (`Resigned`) can be predicted from a structured HR dataset containing 100,000 employee records.

The intended experimental programme benchmarks six advanced tabular machine learning architectures alongside reference baselines.

Later project stages are designed to evaluate:

* predictive performance,
* probability calibration,
* subgroup fairness,
* resource-constrained retention prioritization,
* and final generalization on a quarantined formal holdout.

### Core Problem Specifications

* **Target Variable:** `Resigned` (`True` / `False`; positive class = resignation).
* **Dataset Size:** 100,000 records.
* **Positive-Class Prevalence:** **10.01%**.
* **Predictor Set:** 17 validated predictors:

  * 4 categorical,
  * 13 numerical/ordinal.
* **Feature Contract:** `configs/feature_contract.yaml`.
* **Primary Predictive Metric:** **Average Precision (AP / PR-AUC)**.
* **Secondary Metrics:** ROC-AUC, Brier Score, Balanced Accuracy, Recall, and F1.
* **Development Partition:** 80,000 records.
* **Formal Holdout:** 20,000 records.
* **Development CV:** five fixed stratified folds of 16,000 records each.

The project treats development cross-validation as the primary evidence for model comparison.

---

## 2. Formal Holdout Policy

The formal split is defined by:

`data/processed/split_registry.csv`

Current structure:

```text
Full Dataset (100,000)
├── Development (80,000)
│   ├── Fold 1: 16,000
│   ├── Fold 2: 16,000
│   ├── Fold 3: 16,000
│   ├── Fold 4: 16,000
│   └── Fold 5: 16,000
└── Formal Holdout (20,000)
```

Since the formal protocol was frozen, the holdout must not influence:

* feature engineering,
* preprocessing decisions,
* PCA decisions,
* hyperparameter selection,
* early stopping,
* model selection,
* calibration selection,
* threshold selection,
* fairness-design decisions,
* prioritization strategy design,
* or any other development decision.

The intended final evaluation is:

`notebooks/15_final_holdout_evaluation.ipynb`

That notebook remains locked.

### Historical Exposure Caveat

The canonical methodology records a preliminary context-framing diagnostic performed before the final protocol was established.

Therefore, evaluators should **not describe the holdout as guaranteed pristine or completely untouched from all historical exploratory exposure**.

The correct interpretation is:

> The formal holdout has been quarantined from the subsequent formal model-development process, but the project acknowledges limited preliminary historical exposure before the final protocol was frozen.

---

## 3. Six Headline Model Architectures

The planned headline models are:

1. **Member 1 — FT-Transformer**

   * Feature-tokenizer transformer for tabular data.
   * PyTorch / RTDL-style implementation.
   * A full five-fold development run currently exists.
   * Saved weighted/early-stopping results are available.
   * The experimental protocol is still under methodology review before it should be regarded as final/frozen.

2. **Member 2 — CatBoost**

   * Gradient-boosted decision trees with native categorical support.
   * A basic adapter, configuration file, and notebook scaffold exist.
   * The formal tuning/CV workflow has **not yet been implemented and executed completely**.
   * No formal OOF result set currently exists.

3. **Member 3 — XGBoost**

   * Gradient-boosted tree architecture.
   * A basic adapter, configuration file, and notebook scaffold exist.
   * Formal tuning/CV execution is still pending.
   * Do not assume `scale_pos_weight`, early stopping, or the final methodology-defined tuning procedure is already implemented unless verified directly in the current source.

4. **Member 4 — LightGBM**

   * Leaf-wise gradient-boosted decision-tree architecture.
   * A basic adapter, configuration file, and notebook scaffold exist.
   * Formal tuning/CV execution is still pending.
   * Intended imbalance and early-stopping strategies should be verified against actual source rather than inferred from documentation.

5. **Member 5 — Explainable Boosting Machine / EBM**

   * Interpretable generalized additive boosting model via InterpretML.
   * A basic adapter, configuration file, and notebook scaffold exist.
   * Formal tuning/CV execution and OOF result generation remain pending.

6. **Member 6 — TabNet**

   * Attention-based tabular deep-learning architecture.
   * Adapter and preprocessing components exist.
   * A development smoke test has been executed.
   * Formal five-fold CV has **not** yet been run.

---

## 4. Baselines & PCA Benchmark

The project includes the following reference models:

### Dummy Prior

`DummyClassifier(strategy="prior")`

Purpose:

* establish the no-skill probability-ranking reference,
* provide a baseline approximately equal to positive-class prevalence.

Verified development result:

* **AP / PR-AUC:** `0.100085`
* **ROC-AUC:** `0.499917`

---

### Dummy Most Frequent

`DummyClassifier(strategy="most_frequent")`

Purpose:

* predict the majority class,
* demonstrate why raw accuracy is misleading under approximately 10% positive prevalence.

Verified development result:

* **AP / PR-AUC:** `0.100100`
* **ROC-AUC:** `0.500000`

---

### Logistic Regression — No PCA

Pipeline:

```text
StandardScaler
→ LogisticRegression
```

Verified development result:

* **AP / PR-AUC:** `0.100270`
* **ROC-AUC:** `0.499881`

---

### PCA Logistic Regression Benchmark

Pipeline:

```text
StandardScaler
→ PCA (>=95% variance)
→ LogisticRegression
```

Verified development result:

* **AP / PR-AUC:** `0.099721`
* **ROC-AUC:** `0.499724`

The PCA benchmark is a required analytical comparison and is **not** treated as a seventh advanced model.

The current PCA analysis reports:

* 10 principal components for at least 80% variance,
* 12 principal components for at least 90% variance,
* 12 principal components for at least 95% variance.

The PCA results have been generated, but the notebook still requires final verification/clean execution before being treated as fully frozen.

---

## 5. Current FT-Transformer Result

The currently saved weighted early-stopping FT-Transformer development result is approximately:

* **Pooled AP / PR-AUC:** `0.099979`
* **Pooled ROC-AUC:** `0.497254`

This performance is close to the no-skill baseline.

However:

> The current FT-Transformer experiment should not yet be treated as the final frozen headline result because the project's tuning / outer-CV methodology is still being reviewed for complete independence between hyperparameter selection and outer-fold evaluation.

An evaluator should inspect the actual notebook, source implementation, and tuning procedure rather than assuming the current experiment is the project's final protocol.

---

## 6. Current Predictive Evidence

The experiments completed so far are clustered near the no-skill AP baseline of approximately `0.10`.

Current evidence therefore suggests **very weak predictive signal**.

This is **not yet a final project-wide no-signal conclusion**, because:

* CatBoost has not completed formal CV,
* XGBoost has not completed formal CV,
* LightGBM has not completed formal CV,
* EBM has not completed formal CV,
* TabNet has not completed formal CV.

The project's canonical methodology explicitly allows a valid no-signal outcome.

The purpose of the project is not to manufacture predictive performance. A rigorous finding that the available features contain little useful resignation signal is academically valid if supported by properly controlled experiments.

---

## 7. Notebook Status

| Notebook                                 | Purpose                                         | Current Status                                         |
| :--------------------------------------- | :---------------------------------------------- | :----------------------------------------------------- |
| `01_data_audit.ipynb`                    | Dataset audit and integrity checks              | **COMPLETED**                                          |
| `02_create_split_registry.ipynb`         | Frozen development/holdout + five-fold registry | **COMPLETED**                                          |
| `03_eda.ipynb`                           | Development-only exploratory analysis           | **COMPLETED**                                          |
| `04_baselines.ipynb`                     | Dummy baselines                                 | **COMPLETED**                                          |
| `04a_pca_analysis.ipynb`                 | PCA + Logistic benchmark                        | **RESULTS GENERATED — FINAL VERIFICATION PENDING**     |
| `05_member_1/05_m1_ft_transformer.ipynb` | FT-Transformer                                  | **5-FOLD RESULTS AVAILABLE — PROTOCOL REVIEW PENDING** |
| `06_member_2/06_m2_catboost.ipynb`       | CatBoost                                        | **BASIC SCAFFOLD / ADAPTER — FORMAL WORKFLOW PENDING** |
| `07_member_3/07_m3_xgboost.ipynb`        | XGBoost                                         | **BASIC SCAFFOLD / ADAPTER — FORMAL WORKFLOW PENDING** |
| `08_member_4/08_m4_lightgbm.ipynb`       | LightGBM                                        | **BASIC SCAFFOLD / ADAPTER — FORMAL WORKFLOW PENDING** |
| `09_member_5/09_m5_ebm.ipynb`            | EBM                                             | **BASIC SCAFFOLD / ADAPTER — FORMAL WORKFLOW PENDING** |
| `10_member_6/10_m6_tabnet.ipynb`         | TabNet                                          | **SMOKE-TESTED — FORMAL 5-FOLD CV PENDING**            |
| `11_model_comparison.ipynb`              | Cross-model comparison                          | **SCAFFOLD — NOT EXECUTED**                            |
| `12_calibration.ipynb`                   | Probability calibration                         | **SCAFFOLD — NOT EXECUTED**                            |
| `13_fairness_analysis.ipynb`             | Subgroup fairness analysis                      | **SCAFFOLD — NOT EXECUTED**                            |
| `14_retention_prioritization.ipynb`      | Resource-constrained prioritization             | **SCAFFOLD — NOT EXECUTED**                            |
| `15_final_holdout_evaluation.ipynb`      | Formal final corroboration                      | **LOCKED**                                             |

---

## 8. Directory Structure & Current Implementation State

```text
Employee_Resignation_Prediction/
├── configs/
│   ├── feature_contract.yaml
│   ├── split_config.yaml
│   └── models/
│       ├── catboost.yaml
│       ├── ebm.yaml
│       ├── ft_transformer.yaml
│       ├── lightgbm.yaml
│       ├── tabnet.yaml
│       └── xgboost.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│       └── split_registry.csv
│
├── docs/
│   └── context/
│       └── Employee_Resignation_Prediction_AI_Project_Context_v4_4_0.md
│
├── models/
│   └── Reserved for final serialized model artefacts
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_create_split_registry.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_baselines.ipynb
│   ├── 04a_pca_analysis.ipynb
│   ├── 05_member_1/
│   ├── 06_member_2/
│   ├── 07_member_3/
│   ├── 08_member_4/
│   ├── 09_member_5/
│   ├── 10_member_6/
│   ├── 11_model_comparison.ipynb
│   ├── 12_calibration.ipynb
│   ├── 13_fairness_analysis.ipynb
│   ├── 14_retention_prioritization.ipynb
│   └── 15_final_holdout_evaluation.ipynb
│
├── presentations/
│   └── assets/
│
├── reports/
│
├── results/
│   ├── audit/               # Populated
│   ├── baselines/           # Populated
│   ├── eda/                 # Populated
│   ├── pca/                 # Populated
│   ├── models/
│   │   ├── ft_transformer/  # Current formal development results available
│   │   ├── catboost/        # Formal results pending
│   │   ├── ebm/             # Formal results pending
│   │   ├── lightgbm/        # Formal results pending
│   │   ├── tabnet/          # Formal results pending
│   │   └── xgboost/         # Formal results pending
│   ├── comparison/          # Pending
│   ├── calibration/         # Pending
│   ├── fairness/            # Pending
│   ├── prioritization/      # Pending
│   └── final_holdout/       # Locked / pending
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── prioritization/
│
├── tests/
├── AGENTS.md
├── EVALUATION_GUIDE.md
├── README.md
└── requirements.txt
```

### Important Implementation Note

Do not assume every `src/` package is complete.

Current repository state includes:

* `src/evaluation/metrics.py` — implemented,
* `src/evaluation/oof_validation.py` — implemented,
* several other evaluation modules — still placeholders,
* `src/data/` — currently placeholder modules,
* `src/prioritization/` — currently placeholder modules,
* `src/models/` — contains the model-adapter work completed so far.

Similarly:

* the existence of a YAML model config does not mean its tuning space is finalized,
* several model configurations still require methodology alignment,
* the existence of a notebook scaffold does not mean the workflow has been implemented.

---

## 9. Model Configuration Caution

The files under:

`configs/models/`

should be treated as **current configuration drafts**, not automatically as frozen final experiment contracts.

Several configurations still require alignment with the canonical methodology.

An evaluator should verify:

* whether every intended hyperparameter is represented,
* whether search spaces match the methodology,
* whether early stopping is actually implemented,
* whether class-imbalance handling is actually implemented,
* whether inner-validation logic exists,
* whether the model adapter really consumes the configuration as documented.

Do not infer implementation from YAML comments alone.

---

## 10. Testing Status

The project uses `pytest`.

Recommended invocation:

```bash
python -m pytest
```

Latest verified live-repository result on **2026-09-11**:

```text
149 passed
1 failed
150 total
```

The remaining failure is:

`tests/test_pca_pipeline.py::test_pca_notebook_scaffold_integrity`

That test still assumes the PCA notebook must remain an unexecuted scaffold, while the PCA notebook now contains executed analysis results.

Therefore, the failure currently appears to represent an **outdated test expectation**, rather than a failure of the PCA analysis itself.

The test should be updated to validate the completed PCA workflow and then the full suite should be rerun before claiming a fully green test suite.

---

## 11. Key Files for an External Reviewer

A reviewer should prioritize these sources:

### Canonical Methodology

`docs/context/Employee_Resignation_Prediction_AI_Project_Context_v4_4_0.md`

This is the primary project methodology and governance document.

---

### Project Overview

`README.md`

Use this for the current high-level project state.

---

### Feature Contract

`configs/feature_contract.yaml`

Use this to verify:

* feature membership,
* types,
* allowed roles,
* target definition,
* sensitive attributes.

---

### Split Registry

`data/processed/split_registry.csv`

Use this to verify:

* development membership,
* holdout membership,
* fixed CV folds,
* Employee_ID uniqueness,
* holdout isolation.

---

### Shared Metrics

`src/evaluation/metrics.py`

Use this to audit metric implementations.

---

### OOF Validation

`src/evaluation/oof_validation.py`

Use this to audit out-of-fold integrity and leakage protections.

---

### Model Adapters

`src/models/`

Inspect these directly rather than relying on documentation descriptions.

Particular attention should be given to whether each adapter actually implements:

* preprocessing,
* class-imbalance handling,
* early stopping,
* inner validation,
* serialization,
* deterministic seeding,
* intended hyperparameters.

---

### Saved Results

Relevant currently populated locations include:

```text
results/audit/
results/eda/
results/baselines/
results/pca/
results/models/ft_transformer/
```

Most downstream and remaining-model result directories do not yet contain formal experimental outputs.

---

## 12. Recommended Evaluation Questions

When reviewing this project, focus on evidence rather than file existence.

### A. Data & Split Integrity

Verify:

* whether Employee_ID values are unique,
* whether the frozen 80/20 split is reproducible,
* whether the five development folds are exactly preserved,
* whether formal holdout IDs are absent from development OOF predictions.

---

### B. Leakage Prevention

Verify:

* preprocessing is fitted only on training portions,
* PCA is fitted fold-locally,
* model preprocessing does not use outer validation rows improperly,
* formal holdout information never influences development decisions,
* inner-validation logic is correctly separated from outer validation.

---

### C. Hyperparameter Selection

Verify:

* tuning uses training-side evidence only,
* outer validation remains independent from hyperparameter selection,
* early stopping does not use the outer validation fold,
* final model-comparison OOF predictions are genuinely out-of-sample with respect to both fitting and model-selection decisions.

This point is especially important for the current FT-Transformer workflow.

---

### D. Model Implementation Accuracy

For every model, compare:

```text
Canonical methodology
↕
Model YAML configuration
↕
Source adapter
↕
Notebook workflow
↕
Saved result artifacts
```

Report any disagreement.

Do not assume documentation reflects current source behavior.

---

### E. Reproducibility

Verify:

* random seeds,
* fold membership,
* dependency specifications,
* output manifests,
* notebook execution state,
* model serialization,
* artifact paths.

---

### F. Imbalanced Classification

Because positive prevalence is approximately 10%, determine whether:

* Average Precision is interpreted correctly,
* raw accuracy is not overemphasized,
* class-weighting decisions are development-only,
* threshold-dependent metrics are distinguished from ranking metrics.

---

### G. Calibration

Calibration is currently planned but not yet formally executed.

When implemented, verify:

* the calibrator is selected using development evidence only,
* calibration does not access the formal holdout,
* Brier Score and reliability diagnostics are interpreted appropriately.

---

### H. Fairness

Fairness analysis is currently planned but not yet formally executed.

When implemented, verify:

* subgroup definitions match the canonical methodology,
* Gender and Age sensitivity analyses are handled consistently,
* fairness results are interpreted cautiously,
* protected characteristics are not treated as causal explanations.

---

### I. Retention Prioritization

Prioritization is currently planned but not yet formally executed.

When implemented, verify:

* intervention capacity is applied consistently,
* the value proxy is fitted only using permitted development information,
* risk-only, value-only, random, and combined strategies are compared,
* sensitivity analysis is performed,
* the proxy is not presented as a literal measure of employee worth.

---

### J. No-Signal Interpretation

If all advanced models remain near the no-skill AP baseline:

* do not treat this automatically as model failure,
* determine whether the evaluation was methodologically sound,
* assess whether the predictors genuinely contain insufficient resignation signal,
* verify that the project does not manipulate the evaluation procedure merely to produce stronger scores.

---

## 13. Suggested Evaluation Prompts

### Prompt 1 — Comprehensive Repository Audit

> Review this Employee Resignation Prediction repository as a technical and machine-learning auditor. Distinguish carefully between implemented code, notebook scaffolding, executed experiments, and saved results. Identify incomplete components, contradictions between documentation/configs/source/notebooks, reproducibility issues, and unsupported claims. Do not assume that the existence of a file means its intended workflow is implemented.

---

### Prompt 2 — Machine Learning Methodology & Leakage Audit

> Audit the project's machine-learning methodology with particular attention to the frozen 80/20 split, five-fold development CV, fold-local preprocessing, out-of-fold predictions, hyperparameter selection, inner validation, early stopping, formal-holdout quarantine, and leakage prevention. Verify behavior against the actual source and notebooks rather than documentation claims.

---

### Prompt 3 — FT-Transformer Review

> Inspect `src/models/ft_transformer.py`, `notebooks/05_member_1/05_m1_ft_transformer.ipynb`, the FT configuration, and saved FT result artifacts. Evaluate preprocessing, feature tokenization, class weighting, inner-validation early stopping, hyperparameter screening, OOF prediction integrity, reproducibility, and whether the current tuning procedure leaves the reported outer-CV performance fully independent of hyperparameter selection.

---

### Prompt 4 — Remaining Headline Models

> Review the CatBoost, XGBoost, LightGBM, EBM, and TabNet adapters, configs, and notebooks. Determine exactly what is already implemented versus merely planned. Identify missing tuning logic, early stopping, imbalance handling, preprocessing, OOF generation, serialization, tests, and methodology alignment. Do not infer features from comments or YAML files unless the source actually implements them.

---

### Prompt 5 — Current Results Interpretation

> Independently verify the saved baseline, Logistic Regression, PCA, and FT-Transformer metrics. Compare them with the approximately 10.01% positive prevalence and explain whether the completed experiments demonstrate meaningful ranking signal. Do not make a final dataset-wide no-signal conclusion until the remaining headline models are formally evaluated.

---

### Prompt 6 — Downstream Pipeline Readiness

> Review notebooks 11–15 and associated `src/` modules. Determine what still needs to be implemented before model comparison, calibration, fairness analysis, retention prioritization, pipeline freezing, and final holdout corroboration can be performed safely. Confirm that Notebook 15 should remain locked until all development decisions are frozen.

---

## 14. Evaluation Principles

An evaluator should follow these principles:

1. **Repository evidence overrides documentation claims when they conflict.**
2. **Saved outputs must be traceable to reproducible code.**
3. **A notebook scaffold is not an implemented experiment.**
4. **A model adapter is not evidence that formal CV has been completed.**
5. **A YAML search space is not evidence that the search was executed.**
6. **The formal holdout must not drive development decisions.**
7. **Outer-validation predictions should remain independent of fitting and hyperparameter selection.**
8. **Near-random predictive performance is a valid scientific result if produced by a rigorous evaluation.**
9. **Do not exaggerate project completeness.**
10. **Do not invent missing implementation details.**

---

## 15. Current Overall Status

The project currently has a strong experimental foundation:

* dataset integrity audit completed,
* deterministic split registry created,
* development-only EDA completed,
* dummy baselines completed,
* PCA / Logistic benchmark results generated,
* reusable metric and OOF-validation utilities implemented,
* FT-Transformer full development CV results available,
* model adapters/configuration scaffolding created for the remaining architectures,
* formal holdout still locked.

However, the project is **not yet finished**.

Major remaining work includes:

* finalizing one consistent tuning/evaluation protocol,
* aligning model configs and source implementations with that protocol,
* completing CatBoost formal experiments,
* completing XGBoost formal experiments,
* completing LightGBM formal experiments,
* completing EBM formal experiments,
* completing TabNet formal five-fold CV,
* performing model comparison,
* performing calibration,
* performing fairness analysis,
* performing retention prioritization,
* freezing the final predictive pipeline,
* performing the one-time final holdout corroboration,
* and producing the final academic report and presentation.

The evaluator should judge the project based on the **actual current repository state**, not the intended end state.
