# Employee Resignation Prediction

A rigorous, leak-free machine learning benchmark predicting employee attrition (`Resigned`) and evaluating resource-constrained retention prioritization on tabular HR data (100,000 employee records).

---

## 1. Project Overview & Objective

The primary objective of this project is to develop and benchmark six advanced tabular machine learning models alongside reference baselines to predict employee resignation. Beyond point predictions, the project evaluates probability calibration, fairness across demographic subgroups, and simulated resource-constrained retention prioritization strategies (top-20% capacity simulation).

### Key Facts & Contract Specifications
- **Target Variable:** `Resigned` (Boolean: `True` / `False`; positive class = `True` / `1`).
- **Target Prevalence:** ~10.02% positive class across the 100,000-record dataset (10.01% in development partition).
- **Predictor Set:** 17 strictly validated features defined in `configs/feature_contract.yaml` (4 categorical, 13 numerical/ordinal attributes).
- **Primary Performance Metric:** **Average Precision (AP / PR-AUC)**, chosen specifically for positive-class sensitivity under severe class imbalance (~10% positive prevalence). Secondary metrics include ROC-AUC, Brier Score, Balanced Accuracy, Recall, and F1.
- **Resource Simulation:** Retention prioritization at 20% organizational intervention capacity comparing random, risk-only, value-only, and risk-times-value-proxy strategies.

---

## 2. Experimental Methodology & Holdout Policy

```
Full Dataset (100,000 rows)
├── Development Partition (80,000 rows, 80%)  <-- Used for all EDA, PCA, training, tuning & OOF evaluation
│   ├── Fold 1 (16,000 rows)
│   ├── Fold 2 (16,000 rows)
│   ├── Fold 3 (16,000 rows)
│   ├── Fold 4 (16,000 rows)
│   └── Fold 5 (16,000 rows)
└── Formal Holdout Partition (20,000 rows, 20%)  <-- STRICTLY ISOLATED; NEVER used for training or tuning
```

> **CRITICAL HOLDOUT ISOLATION RULE:**
> The 20,000-row formal holdout set (`holdout_df`) is strictly quarantined in `data/processed/split_registry.csv`. It must **NEVER** be accessed or used during exploratory data analysis, PCA decisions, feature engineering, model selection, hyperparameter tuning, calibration selection, or prioritization strategy design. It will be evaluated exactly once as a final post-freeze corroboration in `notebooks/15_final_holdout_evaluation.ipynb`.

### Six Headline Model Architectures
1. **Member 1:** FT-Transformer (Feature Tokenizer Transformer via RTDL / PyTorch)
2. **Member 2:** CatBoost (Gradient Boosted Decision Trees with native categorical handling)
3. **Member 3:** XGBoost (Extreme Gradient Boosting with histogram binning)
4. **Member 4:** LightGBM (Light Gradient Boosting Machine)
5. **Member 5:** Explainable Boosting Machine / EBM (InterpretML Generalized Additive Model with pairwise interactions)
6. **Member 6:** TabNet (Attentive Interpretable Tabular Deep Learning via PyTorch)

**Mandatory Baseline & Benchmark Models:**
- `DummyClassifier(strategy="prior")` (No-skill probability reference)
- `DummyClassifier(strategy="most_frequent")` (Majority-class baseline demonstrating accuracy paradox)
- `StandardScaler` $\rightarrow$ `LogisticRegression` (Reference linear baseline without PCA)
- `StandardScaler` $\rightarrow$ `PCA (>=95% variance)` $\rightarrow$ `LogisticRegression` (Mandatory PCA linear benchmark; supporting analysis, not a 7th advanced model)

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
04a PCA Analysis & Linear Benchmark [CURRENT]
    ├─ 13 Numerical/Ordinal Predictors (80,000 dev rows, holdout isolated)
    ├─ Variance Retained: 10 PCs for 80%, 12 PCs for 90%, 12 PCs for 95%
    ├─ Benchmark Pipeline A: StandardScaler → LogisticRegression (No PCA)
    └─ Benchmark Pipeline B: StandardScaler → PCA (>=95% variance) → LogisticRegression
   ↓
05–10 Six Headline Advanced Models [PLANNED]
    ├─ 05 Member 1: FT-Transformer
    ├─ 06 Member 2: CatBoost
    ├─ 07 Member 3: XGBoost
    ├─ 08 Member 4: LightGBM
    ├─ 09 Member 5: EBM
    └─ 10 Member 6: TabNet
   ↓
11 Multi-Model Comparison & Error Analysis [PLANNED]
   ↓
12 Probability Calibration [PLANNED]
   ↓
13 Subgroup Fairness Analysis [PLANNED]
   ↓
14 Resource-Constrained Retention Prioritization [PLANNED]
   ↓
Freeze Final Predictive Pipeline [PLANNED]
   ↓
15 Final Holdout Evaluation & Generalization Corroboration [LOCKED]
   ↓
Final Academic Report + Presentation Deck + Viva [PLANNED]
```

### Notebook Sequence & Status Matrix

| Notebook | Name | Stage | Status |
| :--- | :--- | :--- | :--- |
| `01_data_audit.ipynb` | Raw Data Audit & Integrity | Foundations | **COMPLETED** |
| `02_create_split_registry.ipynb` | Stratified 80/20 & 5-Fold CV Generator | Foundations | **COMPLETED** |
| `03_eda.ipynb` | Development Exploratory Data Analysis | Foundations | **COMPLETED** |
| `04_baselines.ipynb` | Dummy Prior & Most Frequent Baselines | Baselines | **COMPLETED** |
| `04a_pca_analysis.ipynb` | Unsupervised PCA & Linear Benchmark | Benchmark | **CURRENT** |
| `05_member_1/05_m1_ft_transformer.ipynb` | FT-Transformer Deep Learning Model (Member 1) | Headline Models | **PLANNED** |
| `06_member_2/06_m2_catboost.ipynb` | CatBoost GBDT Model (Member 2) | Headline Models | **PLANNED** |
| `07_member_3/07_m3_xgboost.ipynb` | XGBoost GBDT Model (Member 3) | Headline Models | **PLANNED** |
| `08_member_4/08_m4_lightgbm.ipynb` | LightGBM GBDT Model (Member 4) | Headline Models | **PLANNED** |
| `09_member_5/09_m5_ebm.ipynb` | Explainable Boosting Machine ($GA^2M$, Member 5) | Headline Models | **PLANNED** |
| `10_member_6/10_m6_tabnet.ipynb` | TabNet Deep Learning Model (Member 6) | Headline Models | **PLANNED** |
| `11_model_comparison.ipynb` | Multi-Model Comparison & Error Analysis | Evaluation | **PLANNED** |
| `12_calibration.ipynb` | Probability Calibration & Reliability | Evaluation | **PLANNED** |
| `13_fairness_analysis.ipynb` | Subgroup Fairness & Disparity Audit | Evaluation | **PLANNED** |
| `14_retention_prioritization.ipynb` | Constrained Retention Simulation (20%) | Simulation | **PLANNED** |
| `15_final_holdout_evaluation.ipynb` | Final Isolated Holdout Assessment | Final Corroboration | **LOCKED** |

---

## 4. Project Directory Structure

```text
Employee_Resignation_Prediction/
├── configs/
│   ├── feature_contract.yaml            # Authoritative feature contract (17 predictors)
│   ├── split_config.yaml                # 80/20 partition and 5-fold CV configuration
│   └── models/                          # Model hyperparameter search spaces and schemas
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
│   └── context/                         # Canonical project context specification
│       ├── Employee_Resignation_Prediction_AI_Project_Context_v4_4_0.md
│       └── Employee_Resignation_Prediction_AI_Project_Context_v4_4_0.docx (pending regeneration)
├── models/                              # Serialized model checkpoint directories
├── notebooks/                           # Numbered reproducible analysis & modeling notebooks (01 to 15 + 04a)
├── presentations/
│   └── assets/                          # Presentation figures and assets
├── results/                             # Generated experimental outputs, tables, and figures
│   ├── audit/                           # Audit tables from Notebook 01
│   ├── baselines/                       # Baseline OOF predictions and confusion matrices
│   ├── calibration/                     # Post-calibration reliability curves and artifacts
│   ├── comparison/                      # Multi-model comparative metrics and rankings
│   ├── eda/                             # Canonical EDA figures and manifest
│   ├── fairness/                        # Subgroup disparity and fairness audit tables
│   ├── final_holdout/                   # Final isolated holdout evaluation outputs (locked)
│   ├── models/                          # Individual model OOF prediction CSVs and fold metrics
│   │   ├── catboost/
│   │   ├── ebm/
│   │   ├── ft_transformer/
│   │   ├── lightgbm/
│   │   ├── tabnet/
│   │   └── xgboost/
│   ├── pca/                             # PCA explained variance, loadings, and benchmark outputs
│   └── prioritization/                  # Retention allocation curves and sensitivity tables
├── src/                                 # Shared reusable Python modules
│   ├── data/                            # Dataset and split loading utilities
│   ├── evaluation/                      # Metrics calculation and OOF validation assertions
│   ├── models/                          # Model adapter implementations (M1 to M6 + Logistic Regression)
│   └── prioritization/                  # Retention ranking algorithms & value proxy definitions
└── tests/                               # Automated unit, contract, and leakage integrity tests
```

---

## 5. Environment Setup & Installation

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

## 6. Dataset & Split Registry Verification

The dataset and split registry must match their exact cryptographic hashes:

| File | Relative Path | Expected SHA-256 Hash |
| :--- | :--- | :--- |
| **Raw Dataset** | `data/raw/Extended_Employee_Performance_and_Productivity_Data.csv` | `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb` |
| **Split Registry** | `data/processed/split_registry.csv` | `87edc10ac38f2435c49beddbf412f9de293e3e0622558b27ffbdf31fadd7ea41` |

---

## 7. Automated Test Suite

The repository includes an extensive `pytest` test suite enforcing dataset contracts, feature types, metric implementations, out-of-fold validation integrity, model configurations, model adapter protocol compliance, PCA pipeline constraints, and formal holdout isolation guards.

Run the test suite from the repository root:
```bash
python -m pytest
```

**Current Verified Test Status:** `143 passed (100% passing)`
