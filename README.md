# Employee Resignation Prediction

A rigorous, leak-free machine learning benchmark predicting employee attrition (`Resigned`) and evaluating resource-constrained retention prioritization on tabular HR data (100,000 employee records).

---

## 1. Project Overview & Objective

The primary objective of this project is to develop and benchmark six advanced tabular machine learning models alongside reference baselines to predict employee resignation. Beyond point predictions, the project evaluates probability calibration, fairness across demographic subgroups, and simulated resource-constrained retention prioritization strategies (top-20% capacity simulation).

### Key Facts & Contract Specifications
- **Target Variable:** `Resigned` (Boolean: `True` / `False`; positive class = `True` / `1`).
- **Target Prevalence:** ~10.02% positive class across the 100,000-record dataset (10.01% in development).
- **Predictor Set:** 17 strictly validated features defined in `configs/feature_contract.yaml` (numerical, ordinal, and categorical attributes).
- **Primary Performance Metric:** **Average Precision (AP / PR-AUC)**, chosen specifically for positive-class sensitivity under severe class imbalance (~10% positive prevalence). Secondary metrics include ROC-AUC, Brier Score, Balanced Accuracy, Recall, and F1.
- **Resource Simulation:** Retention prioritization at 20% organizational intervention capacity comparing random, risk-only, value-only, and risk-times-value-proxy strategies.

---

## 2. Experimental Methodology & Holdout Policy

```
Full Dataset (100,000 rows)
├── Development Partition (80,000 rows, 80%)  <-- Used for all EDA, training, tuning & OOF evaluation
│   ├── Fold 0 (16,000 rows)
│   ├── Fold 1 (16,000 rows)
│   ├── Fold 2 (16,000 rows)
│   ├── Fold 3 (16,000 rows)
│   └── Fold 4 (16,000 rows)
└── Formal Holdout Partition (20,000 rows, 20%)  <-- STRICTLY ISOLATED; NEVER used for training or tuning
```

> **CRITICAL HOLDOUT ISOLATION RULE:**
> The 20,000-row formal holdout set (`holdout_df`) is strictly isolated in `data/processed/split_registry.csv`. It must **NEVER** be used during exploratory data analysis, feature engineering, model selection, hyperparameter tuning, or threshold setting. It will be evaluated exactly once as a final post-freeze corroboration.

### Six Headline Model Architectures
1. **Member 1:** FT-Transformer (Deep Learning Tabular Transformer via RTDL / PyTorch)
2. **Member 2:** CatBoost (Gradient Boosted Decision Trees with native categorical handling)
3. **Member 3:** XGBoost (Extreme Gradient Boosting with histogram binning)
4. **Member 4:** LightGBM (Light Gradient Boosting Machine)
5. **Member 5:** Explainable Boosting Machine / EBM (InterpretML Generalized Additive Model with pairwise interactions)
6. **Member 6:** TabNet (Attentive Interpretable Tabular Deep Learning via PyTorch)

**Reference Baselines:**
- `DummyClassifier(strategy="prior")` (No-skill probability reference)
- `DummyClassifier(strategy="most_frequent")` (Majority-class baseline demonstrating accuracy paradox)
- Optional `LogisticRegression` (Fold-local preprocessed linear benchmark)

---

## 3. Project Directory Structure

```
Employee_Resignation_Prediction/
├── configs/
│   ├── feature_contract.yaml            # Authoritative feature contract (17 predictors)
│   ├── split_config.yaml                # 80/20 partition and 5-fold CV configuration
│   └── models/                          # Model hyperparameter tuning search spaces
│       ├── catboost.yaml
│       ├── ebm.yaml
│       ├── ft_transformer.yaml
│       ├── lightgbm.yaml
│       ├── tabnet.yaml
│       └── xgboost.yaml
├── data/
│   ├── raw/                             # Raw immutable CSV dataset
│   │   └── Extended_Employee_Performance_and_Productivity_Data.csv
│   └── processed/                       # Frozen split registry
│       └── split_registry.csv
├── docs/
│   └── context/                         # Canonical project context specification (v4.3.0)
│       ├── Employee_Resignation_Prediction_AI_Project_Context_v4_3_0.md
│       └── Employee_Resignation_Prediction_AI_Project_Context_v4_3_0.docx
├── models/                              # Serialized model checkpoint directories
├── notebooks/                           # Numbered reproducible analysis & modeling notebooks
│   ├── 01_data_audit.ipynb              # Raw data audit and integrity verification
│   ├── 02_create_split_registry.ipynb   # Stratified 80/20 holdout and 5-fold CV generator
│   ├── 03_eda.ipynb                     # Development-only exploratory data analysis
│   ├── 04_baselines.ipynb               # Dummy Prior & Dummy Most Frequent baseline OOF models
│   ├── 11_model_comparison.ipynb        # Six-model comparison on development OOF predictions
│   ├── 12_calibration.ipynb             # Probability calibration & reliability curves
│   ├── 13_fairness_analysis.ipynb       # Demographic parity & subgroup disparity audit
│   ├── 14_retention_prioritization.ipynb# Resource-constrained retention simulation
│   └── models/                          # Individual model tuning and OOF execution notebooks
│       ├── 05_m1_ft_transformer.ipynb
│       ├── 06_m2_catboost.ipynb
│       ├── 07_m3_xgboost.ipynb
│       ├── 08_m4_lightgbm.ipynb
│       ├── 09_m5_ebm.ipynb
│       └── 10_m6_tabnet.ipynb
├── presentations/
│   └── assets/                          # Presentation figures and assets
├── results/                             # Generated experimental outputs, tables, and figures
│   ├── audit/                           # Audit tables from Notebook 01
│   ├── baselines/                       # Baseline OOF prediction CSVs and confusion matrices
│   ├── eda/                             # Canonical EDA figures (figure_01 to 07) & manifest
│   └── ...
├── src/                                 # Shared reusable Python modules
│   ├── data/                            # Dataset and split loading utilities
│   ├── evaluation/                      # Metrics calculation and OOF validation assertions
│   ├── models/                          # Model wrapper adapters (M1 to M6)
│   └── prioritization/                  # Retention ranking algorithms & value proxy definitions
└── tests/                               # Automated unit & contract integrity tests
```

---

## 4. Environment Setup & Installation

### Prerequisites
- Python 3.10 to 3.14
- Virtual environment manager (`venv`)

### 1. Create and Activate Virtual Environment
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

## 5. Dataset & Split Registry Verification

The dataset and split registry must match their exact cryptographic hashes:

| File | Relative Path | Expected SHA-256 Hash |
| :--- | :--- | :--- |
| **Raw Dataset** | `data/raw/Extended_Employee_Performance_and_Productivity_Data.csv` | `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb` |
| **Split Registry** | `data/processed/split_registry.csv` | `6eefc43cbb03023ffbf45df27d53086eb2ca73a0c49a37eecb8fcf70e4277717` |

---

## 6. Execution Workflow & Notebook Order

Execute notebooks strictly in sequential order to maintain data integrity and output dependencies:

1. **`notebooks/01_data_audit.ipynb`** — Verifies raw dataset integrity, absence of missing values, and column schemas.
2. **`notebooks/02_create_split_registry.ipynb`** — Creates the frozen stratified 80/20 split and 5 development folds.
3. **`notebooks/03_eda.ipynb`** — Conducts development-only EDA, producing 20 canonical figures and tables.
4. **`notebooks/04_baselines.ipynb`** — Generates out-of-fold predictions and baseline interpretation metrics for Dummy models.
5. **`notebooks/models/05_m1_ft_transformer.ipynb`** through **`10_m6_tabnet.ipynb`** — Implements hyperparameter tuning and 5-fold OOF training for each headline model.
6. **`notebooks/11_model_comparison.ipynb`** — Multi-model performance ranking on development OOF predictions.
7. **`notebooks/12_calibration.ipynb`** — Probability calibration analysis and Platt scaling / isotonic regression.
8. **`notebooks/13_fairness_analysis.ipynb`** — Subgroup fairness evaluation across sensitive attributes (`Gender`, `Age`, `Overtime`).
9. **`notebooks/14_retention_prioritization.ipynb`** — Simulated resource-constrained retention prioritization.

---

## 7. Automated Test Suite

The repository includes a comprehensive `pytest` test suite enforcing dataset contracts, feature types, metric implementations, out-of-fold validation integrity, and split isolation.

Run the test suite from the repository root:
```bash
python -m pytest
```

**Current Verified Test Status:** `79 passed (100% passing)`

---

## 8. Current Project Status & Next Milestone

- **Completed Foundations:**
  - Raw dataset integrity audit (`01_data_audit.ipynb`).
  - Feature contract validation (`configs/feature_contract.yaml`).
  - Frozen split registry with 80k development / 20k holdout rows (`data/processed/split_registry.csv`).
  - Development-only EDA with 20 canonical outputs (`03_eda.ipynb`).
  - Shared baseline modeling with Dummy Prior & Dummy Most Frequent (`04_baselines.ipynb`).
  - Unit and contract test suite (79 passing tests).
  - Repository and notebook hygiene cleanup.
- **Current Milestone (In Progress):**
  - Implementing model adapters in `src/models/` for all six headline models (FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, TabNet) under strict row-parity and OOF-output contracts.
  - Environment smoke testing and hyperparameter search space verification.
