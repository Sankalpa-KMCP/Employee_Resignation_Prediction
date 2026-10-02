# Employee Resignation Prediction

A rigorous, leakage-controlled machine learning benchmark for predicting employee attrition (`Resigned`) and evaluating resource-constrained retention prioritization on tabular HR data (100,000 employee records).

---

## 1. Project Overview & Objective

The objective of this project is to develop and benchmark six advanced tabular machine learning models alongside reference baselines for employee-resignation prediction. The completed predictive experiments are evaluated primarily with Average Precision (AP / PR-AUC), and the later project stages are designed to assess calibration, subgroup fairness, and simulated resource-constrained retention prioritization.

### Key Facts & Contract Specifications

* **Target Variable:** `Resigned` (Boolean: `True` / `False`; positive class = `True` / `1`).
* **Target Prevalence:** **10.01%** positive class across the full 100,000-record dataset and the stratified development partition.
* **Predictor Set:** 17 validated predictors defined in `configs/feature_contract.yaml` (4 categorical, 13 numerical/ordinal attributes).
* **Primary Performance Metric:** **Average Precision (AP / PR-AUC)**. Secondary metrics include ROC-AUC, Brier Score, Balanced Accuracy, Recall, and F1.
* **Planned Resource Simulation:** Retention prioritization at 20% organizational intervention capacity, with sensitivity analysis around the primary capacity and comparison of random, risk-only, value-only, and risk-times-value-proxy strategies.
* **Current Evidence:** Completed development experiments so far are close to the no-skill AP baseline (~0.10). Some headline models still require formal evaluation, so this is **not yet a final no-signal conclusion**.

---

## 2. Experimental Methodology & Holdout Policy

```text
Full Dataset (100,000 rows)
├── Development Partition (80,000 rows, 80%)
│   ├── Fold 1 (16,000 rows)
│   ├── Fold 2 (16,000 rows)
│   ├── Fold 3 (16,000 rows)
│   ├── Fold 4 (16,000 rows)
│   └── Fold 5 (16,000 rows)
└── Formal Holdout Partition (20,000 rows, 20%)
    └── Quarantined from the formal modelling workflow until final post-freeze corroboration
```

> **Formal holdout rule:** Since the formal split protocol was frozen, the 20,000-row holdout must not be used for preprocessing design, feature engineering decisions, model selection, hyperparameter tuning, early stopping, calibration selection, threshold selection, prioritization design, or any other development decision. It is reserved for one final post-freeze corroboration in `notebooks/15_final_holdout_evaluation.ipynb`.
>
> **Historical exposure caveat:** The canonical methodology records a preliminary context-framing diagnostic performed before the formal protocol was established. Therefore, the holdout is **not described as guaranteed pristine or completely untouched from all historical exploratory exposure**. The five-fold development CV remains the primary model-comparison evidence.

### Six Headline Model Architectures

1. **Member 1:** FT-Transformer (Feature Tokenizer Transformer via RTDL / PyTorch)
2. **Member 2:** CatBoost (Gradient Boosted Decision Trees with native categorical handling)
3. **Member 3:** XGBoost (Extreme Gradient Boosting)
4. **Member 4:** LightGBM (Light Gradient Boosting Machine)
5. **Member 5:** Explainable Boosting Machine / EBM (InterpretML generalized additive model with interactions)
6. **Member 6:** TabNet (Attentive Interpretable Tabular Deep Learning via PyTorch)

### Required Baselines & PCA Benchmark

* `DummyClassifier(strategy="prior")` — no-skill probability reference.
* `DummyClassifier(strategy="most_frequent")` — majority-class baseline demonstrating the accuracy paradox.
* `StandardScaler` → `LogisticRegression` — linear comparison without PCA.
* `StandardScaler` → `PCA (>=95% variance)` → `LogisticRegression` — required PCA demonstration benchmark; not a seventh advanced model.

Model-specific preprocessing is fitted only on the appropriate development-training portion. The formal holdout is excluded from all model-development decisions.

---

## 3. End-to-End Workflow & Lifecycle Status

```text
01 Data Audit [COMPLETED]
   ↓
02 Frozen Split + CV Registry [COMPLETED]
   ↓
03 Exploratory Data Analysis (EDA) [COMPLETED]
   ↓
04 Dummy Baselines [COMPLETED]
   ↓
04a PCA Analysis & Linear Benchmark [RESULTS GENERATED — FINAL VERIFICATION PENDING]
    ├─ 13 Numerical/Ordinal Predictors
    ├─ Variance Retained: 10 PCs for >=80%, 12 PCs for >=90%, 12 PCs for >=95%
    ├─ StandardScaler → LogisticRegression
    └─ StandardScaler → PCA → LogisticRegression
   ↓
05–10 Six Headline Advanced Models [IN PROGRESS]
    ├─ 05 FT-Transformer [5-FOLD CV RESULTS AVAILABLE — PROTOCOL REVIEW PENDING]
    ├─ 06 CatBoost [BASIC ADAPTER/CONFIG/SCAFFOLD PRESENT — FORMAL WORKFLOW PENDING]
    ├─ 07 XGBoost [BASIC ADAPTER/CONFIG/SCAFFOLD PRESENT — FORMAL WORKFLOW PENDING]
    ├─ 08 LightGBM [BASIC ADAPTER/CONFIG/SCAFFOLD PRESENT — FORMAL WORKFLOW PENDING]
    ├─ 09 EBM [NESTED DEVELOPMENT EVALUATION COMPLETE — CANONICAL INPUT REGISTERED]
    └─ 10 TabNet [STRICT-NESTED 5-FOLD DEVELOPMENT OOF COMPLETE — HOLDOUT UNTOUCHED]
   ↓
11 Multi-Model Comparison & Error Analysis [SCAFFOLD — NOT EXECUTED]
   ↓
12 Probability Calibration [SCAFFOLD — NOT EXECUTED]
   ↓
13 Subgroup Fairness Analysis [SCAFFOLD — NOT EXECUTED]
   ↓
14 Resource-Constrained Retention Prioritization [SCAFFOLD — NOT EXECUTED]
   ↓
Freeze Final Predictive Pipeline [PENDING]
   ↓
15 Final Holdout Evaluation & Generalization Corroboration [LOCKED]
   ↓
Final Academic Report + Presentation Deck + Viva [PENDING]
```

### Notebook Sequence & Status Matrix

| Notebook                                 | Name                                    | Stage               | Verified Current Status                                      |
| :--------------------------------------- | :-------------------------------------- | :------------------ | :----------------------------------------------------------- |
| `01_data_audit.ipynb`                    | Raw Data Audit & Integrity              | Foundations         | **COMPLETED**                                                |
| `02_create_split_registry.ipynb`         | Stratified 80/20 & 5-Fold CV Generator  | Foundations         | **COMPLETED**                                                |
| `03_eda.ipynb`                           | Development Exploratory Data Analysis   | Foundations         | **COMPLETED**                                                |
| `04_baselines.ipynb`                     | Dummy Prior & Most Frequent Baselines   | Baselines           | **COMPLETED**                                                |
| `04a_pca_analysis.ipynb`                 | Unsupervised PCA & Linear Benchmark     | Benchmark           | **RESULTS GENERATED — FINAL VERIFICATION PENDING**           |
| `05_member_1/05_m1_ft_transformer.ipynb` | FT-Transformer (Member 1)               | Headline Models     | **5-FOLD CV RESULTS AVAILABLE — PROTOCOL REVIEW PENDING**    |
| `06_member_2/06_m2_catboost.ipynb`       | CatBoost (Member 2)                     | Headline Models     | **BASIC SCAFFOLD/ADAPTER PRESENT — FORMAL WORKFLOW PENDING** |
| `07_member_3/07_m3_xgboost.ipynb`        | XGBoost (Member 3)                      | Headline Models     | **BASIC SCAFFOLD/ADAPTER PRESENT — FORMAL WORKFLOW PENDING** |
| `08_member_4/08_m4_lightgbm.ipynb`       | LightGBM (Member 4)                     | Headline Models     | **BASIC SCAFFOLD/ADAPTER PRESENT — FORMAL WORKFLOW PENDING** |
| `09_member_5/09_m5_ebm.ipynb`            | Explainable Boosting Machine (Member 5) | Headline Models     | **NESTED DEVELOPMENT EVALUATION COMPLETE** |
| `10_member_6/10_m6_tabnet.ipynb`         | TabNet (Member 6)                       | Headline Models     | **STRICT-NESTED 5-FOLD DEVELOPMENT OOF COMPLETE**            |
| `11_model_comparison.ipynb`              | Multi-Model Comparison & Error Analysis | Evaluation          | **SCAFFOLD — NOT EXECUTED**                                  |
| `12_calibration.ipynb`                   | Probability Calibration & Reliability   | Evaluation          | **SCAFFOLD — NOT EXECUTED**                                  |
| `13_fairness_analysis.ipynb`             | Subgroup Fairness & Disparity Audit     | Evaluation          | **SCAFFOLD — NOT EXECUTED**                                  |
| `14_retention_prioritization.ipynb`      | Constrained Retention Simulation        | Simulation          | **SCAFFOLD — NOT EXECUTED**                                  |
| `15_final_holdout_evaluation.ipynb`      | Final Holdout Corroboration             | Final Corroboration | **LOCKED**                                                   |

### Current Development Results

These are completed development-stage results only; they are **not** final holdout results.

| Model / Benchmark                                    | AP / PR-AUC |  ROC-AUC | Status                                               |
| :--------------------------------------------------- | ----------: | -------: | :--------------------------------------------------- |
| Dummy Most Frequent                                  |    0.100100 | 0.500000 | Completed                                            |
| Dummy Prior                                          |    0.100085 | 0.499917 | Completed                                            |
| Logistic Regression — No PCA                         |    0.100270 | 0.499881 | Completed benchmark                                  |
| Logistic Regression — PCA 95%                        |    0.099721 | 0.499724 | Completed benchmark                                  |
| FT-Transformer — current weighted early-stopping run |    0.099979 | 0.497254 | 5-fold OOF result available; protocol review pending |
| EBM — raw-candidate nested development OOF | 0.099000 | 0.495976 | Complete; canonical individual-folder artifacts |
| TabNet — formal strict-nested development OOF        |    0.099965 | 0.499317 | Complete: 5 outer folds, 40 candidate evaluations   |

The formal TabNet result is also clustered around the no-skill reference. It contains 80,000 canonical OOF predictions from five 16,000-row outer folds and 40 inner candidate evaluations; the formal holdout was not used. The earlier T01 pooled result (`0.101706` AP / `0.504556` ROC-AUC) remains historical/exploratory evidence and is not the headline estimate. No final 80,000-row TabNet refit has been performed because model comparison must occur first. The remaining protocol-incomplete headline models must still be formally evaluated before a final project-wide conclusion about predictive signal is made.

---

## 4. Project Directory Structure

```text
Employee_Resignation_Prediction/
├── configs/
│   ├── feature_contract.yaml            # Authoritative feature contract
│   ├── split_config.yaml                # Frozen split / CV configuration
│   └── models/                          # Model configuration/search-space files; not all are final/frozen
│       ├── catboost.yaml
│       ├── ebm.yaml
│       ├── ft_transformer.yaml
│       ├── lightgbm.yaml
│       ├── tabnet.yaml
│       └── xgboost.yaml
├── data/
│   ├── raw/                             # Raw assigned dataset
│   └── processed/
│       └── split_registry.csv           # Frozen row/fold membership registry
├── docs/
│   └── context/
│       └── Employee_Resignation_Prediction_AI_Project_Context_v4_6_2.md
├── models/                              # Reserved for serialized final model artefacts; currently no final checkpoints
├── notebooks/                           # Numbered analysis/modeling notebooks (01–15 plus 04a)
├── presentations/
│   └── assets/                          # Presentation figures/assets
├── reports/                             # Final report artefacts not yet produced
├── results/
│   ├── audit/                           # Populated
│   ├── baselines/                       # Populated
│   ├── eda/                             # Populated
│   ├── pca/                             # Populated
│   ├── models/
│   │   ├── ft_transformer/              # Populated with current FT experiment results
│   │   ├── catboost/                    # Formal results pending
│   │   ├── ebm/individual_IT25100429/    # Canonical nested EBM results
│   │   ├── lightgbm/                    # Formal results pending
│   │   ├── tabnet/                      # Formal strict-nested development results available
│   │   └── xgboost/                     # Formal results pending
│   ├── comparison/                      # Pending
│   ├── calibration/                     # Pending
│   ├── fairness/                        # Pending
│   ├── prioritization/                  # Pending
│   └── final_holdout/                   # Locked / pending
├── src/
│   ├── data/                            # Placeholder modules at the current project stage
│   ├── evaluation/                      # Metrics + OOF validation implemented; other utilities still scaffolded
│   ├── models/                          # Model adapters / model-specific implementation work
│   └── prioritization/                  # Placeholder modules pending the prioritization stage
└── tests/                               # Unit, contract, leakage-integrity, config, adapter and pipeline tests
```

The directory tree distinguishes **existing scaffolding** from **completed functionality**. The existence of a model adapter, YAML configuration, notebook, or result directory does not by itself mean that the corresponding formal experiment has been completed.

---

## 5. Environment Setup & Installation

### Environment Contract

* **Formal project target:** Python **3.11.x**
* **Verified development environment recorded by the project:** Python **3.14.0 on Windows x64**
* Use the repository dependency specification in `requirements.txt`.

### 1. Create and Activate a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Dataset & Split Registry Verification

The assigned dataset and frozen split registry are identified by the following SHA-256 hashes:

| File               | Relative Path                                                      | Expected SHA-256 Hash                                              |
| :----------------- | :----------------------------------------------------------------- | :----------------------------------------------------------------- |
| **Raw Dataset**    | `data/raw/Extended_Employee_Performance_and_Productivity_Data.csv` | `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb` |
| **Split Registry** | `data/processed/split_registry.csv`                                | `87edc10ac38f2435c49beddbf412f9de293e3e0622558b27ffbdf31fadd7ea41` |

The split registry defines the fixed 80,000-row development partition, 20,000-row formal holdout partition, and five 16,000-row development folds.

---

## 7. Automated Test Suite

The repository includes `pytest` coverage for dataset and feature contracts, split integrity, metric implementations, OOF validation, model configuration/adapters, PCA constraints, and formal holdout guards.

Run the suite from the repository root:

```bash
python -m pytest
```

**Latest verified live-repository result (2026-09-11):** `149 passed, 1 failed` (`150` total).

The remaining failure is the existing PCA scaffold-integrity assertion, which still expects `04a_pca_analysis.ipynb` to be unexecuted even though the PCA notebook now contains executed analysis results. The test should be updated to validate the completed PCA workflow and the full suite re-run before reporting a fully green status.

---

## 8. Current Project State

The project foundation is established: dataset auditing, frozen row/fold assignment, development EDA, dummy baselines, PCA benchmarking, shared evaluation utilities, a full FT-Transformer development CV run, and a formal strict-nested TabNet development OOF run are present.

The project is **not yet complete**. The next formal work is to align and complete the remaining protocol-incomplete headline-model experiments, perform model comparison/calibration/fairness/prioritization, freeze the final predictive pipeline, and only then run the formal holdout corroboration. The formal holdout remains untouched by TabNet.

For full methodology and governance details, use:

`docs/context/Employee_Resignation_Prediction_AI_Project_Context_v4_6_2.md`

## EBM formal input reconciliation — IT25100429

The six executed varieties are now registered in `configs/models/ebm.yaml`: E01/E02/E03 with raw and standardized numerical representations. Three raw candidates participate in nested inner-AP selection; three standardized pairs are sensitivity experiments. All fixed parameters and the .001 practical AP margin match the executed experiment. The previous unexecuted search-space ranges are superseded.

The group's canonical formal EBM OOF is `results/models/ebm/individual_IT25100429/oof_predictions.csv`, paired with `fold_metrics.csv` and `summary_metrics.json` in that directory. Mean fold AP .099558; pooled AP .099000; pooled ROC-AUC .495976. This is a completed nested development evaluation, not final holdout evaluation or a frozen deployment model. Earlier EBM status tables are superseded by this update.

Top-level EBM outputs are historical default-only results and must not be loaded as the group's formal input. `load_formal_ebm_input` checks the configured paths, artifact manifest hashes and exact frozen development membership; it has no historical fallback. Notebook 11 explicitly uses this loader.
