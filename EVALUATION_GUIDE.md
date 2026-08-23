# Employee Resignation Prediction — ChatGPT Evaluation Guide

Welcome! This archive contains the complete codebase, configuration contracts, analytical notebooks, documentation, and experimental results for the **Employee Resignation Prediction** project.

---

## 1. Project Overview & Objective

The objective of this project is to develop, benchmark, and evaluate six tabular machine learning architectures alongside reference baselines to predict employee attrition (`Resigned`) on a 100,000-record enterprise HR dataset. Beyond raw classification accuracy, the project evaluates **probability calibration**, **subgroup algorithmic fairness**, and **resource-constrained retention prioritization** under a simulated 20% organizational intervention budget.

### Core Problem Specifications
- **Target Variable:** `Resigned` (Binary boolean: `True` / `False`; positive class = `1` / resignation).
- **Class Imbalance:** ~10.02% positive prevalence across the 100,000 records.
- **Predictor Set:** 17 strictly validated features (4 categorical, 13 numerical/ordinal) defined in `configs/feature_contract.yaml`.
- **Primary Optimization Metric:** **PR-AUC / Average Precision (AP)** for positive-class sensitivity under severe imbalance. Secondary metrics: ROC-AUC, Brier Score, Balanced Accuracy, Recall, F1.
- **Experimental Integrity:** Strict 80/20 split (80,000 development records with 5-fold stratified cross-validation; 20,000 formal holdout records permanently quarantined until final post-freeze evaluation).

---

## 2. Six Benchmark Model Architectures

1. **Member 1 (FT-Transformer):** Feature Tokenizer Transformer for tabular data via PyTorch / `rtdl-revisiting-models` with inner-validation early stopping on PR-AUC, fold-local class weighting, and feature scaling.
2. **Member 2 (CatBoost):** Gradient Boosted Decision Trees with native categorical handling and symmetric tree structures.
3. **Member 3 (XGBoost):** Histogram-based gradient boosting with scale-pos-weight handling and tree regularization.
4. **Member 4 (LightGBM):** Fast gradient boosting with leaf-wise splitting and exclusive feature bundling.
5. **Member 5 (Explainable Boosting Machine / EBM):** Generalized Additive Model with pairwise interaction terms ($GA^2M$) via InterpretML for glassbox interpretability.
6. **Member 6 (TabNet):** Sequential attention-based tabular deep learning via PyTorch TabNet.

### Reference & Benchmark Pipelines
- **Dummy Prior:** Predicts training class prior probabilities ($AP \approx 0.1001$).
- **Dummy Most Frequent:** Predicts majority class (`False`), demonstrating the accuracy paradox.
- **Standard Logistic Regression:** Standardized linear baseline ($AP \approx 0.1601$).
- **PCA Benchmark:** $StandardScaler \to PCA (\ge 95\% \text{ variance retained}) \to LogisticRegression$ ($AP \approx 0.1603$).

---

## 3. Directory Structure & Key Files

```text
Employee_Resignation_Prediction/
├── configs/
│   ├── feature_contract.yaml            # Authoritative feature contract (types, ranges, constraints)
│   ├── split_config.yaml                # Stratification and 5-fold CV definitions
│   └── models/                          # YAML hyperparameter grids & schemas (M1 to M6)
├── docs/
│   └── context/                         # Comprehensive project specification & research context
│       └── Employee_Resignation_Prediction_AI_Project_Context_v4_4_0.md
├── notebooks/                           # Reproducible Jupyter analysis & modeling sequence
│   ├── 01_data_audit.ipynb              # Schema validation, missingness, target distribution
│   ├── 02_create_split_registry.ipynb   # Deterministic split registry generator (80/20 + 5 folds)
│   ├── 03_eda.ipynb                     # Exploratory analysis, feature correlations, distributions
│   ├── 04_baselines.ipynb               # Dummy prior and most-frequent benchmarks
│   ├── 04a_pca_analysis.ipynb           # Unsupervised PCA & linear benchmark comparison
│   ├── 05_member_1/05_m1_ft_transformer.ipynb # Member 1 FT-Transformer training & evaluation
│   ├── 06_member_2/06_m2_catboost.ipynb # Member 2 CatBoost model
│   ├── 07_member_3/07_m3_xgboost.ipynb  # Member 3 XGBoost model
│   ├── 08_member_4/08_m4_lightgbm.ipynb # Member 4 LightGBM model
│   ├── 09_member_5/09_m5_ebm.ipynb      # Member 5 EBM model
│   ├── 10_member_6/10_m6_tabnet.ipynb   # Member 6 TabNet model
│   ├── 11_model_comparison.ipynb        # Multi-model benchmarking & error analysis
│   ├── 12_calibration.ipynb             # Isotonic & Platt probability calibration
│   ├── 13_fairness_analysis.ipynb       # Subgroup demographic parity & equalized odds
│   ├── 14_retention_prioritization.ipynb # Resource-constrained retention simulation (top 20%)
│   └── 15_final_holdout_evaluation.ipynb # Final locked holdout verification
├── presentations/
│   └── assets/                          # Presentation figures & assets
├── results/                             # Evaluation outputs, metrics JSON/CSV, figures, tuning tables
│   ├── audit/                           # Data audit summaries
│   ├── baselines/                       # Baseline fold metrics, PR curves, confusion matrices
│   ├── eda/                             # Canonical EDA visualization PNGs and findings
│   ├── models/ft_transformer/           # FT-Transformer metrics, tuning grid, PR curves
│   └── pca/                             # PCA loadings, explained variance, benchmark metrics
├── src/                                 # Shared, modular Python library
│   ├── data/                            # Feature contract, split registry loader, validation
│   ├── evaluation/                      # Metrics (PR-AUC, Brier, ROC-AUC), OOF validation engine
│   ├── models/                          # Model adapters (FT-Transformer, GBDT, EBM, TabNet, Logistic)
│   └── prioritization/                  # Retention ranking algorithms & value proxy strategies
├── tests/                               # 140+ automated pytest suites (contracts, leak-guards, OOF)
├── AGENTS.md                            # Development & engineering conventions
├── README.md                            # Project overview & quickstart
└── requirements.txt                     # Project dependencies
```

---

## 4. Suggested Evaluation Prompts for ChatGPT

To evaluate this project thoroughly, you can use any of the following prompts with ChatGPT:

### Prompt 1: Comprehensive Code Quality & Architectural Review
> "Please review the attached Employee Resignation Prediction project repository. Evaluate the codebase for architectural design, code modularity, adherence to software engineering best practices, type safety, error handling, and test coverage across the `src/`, `configs/`, `notebooks/`, and `tests/` directories."

### Prompt 2: Machine Learning Rigor & Leakage Prevention Audit
> "Conduct a rigorous machine learning audit of this project. Evaluate whether the cross-validation strategy, out-of-fold validation runner (`src/evaluation/oof_validation.py`), fold-local preprocessing, class-weight handling, and formal holdout quarantine (`data/processed/split_registry.csv`) prevent data leakage and ensure reproducible benchmarks."

### Prompt 3: Deep Learning / FT-Transformer Implementation Assessment
> "Inspect the FT-Transformer model adapter (`src/models/ft_transformer.py`) and notebook (`notebooks/05_member_1/05_m1_ft_transformer.ipynb`). Evaluate the implementation of feature tokenization, inner-validation early stopping on PR-AUC, learning rate scheduling, class imbalance mitigation, and serialization safety."

### Prompt 4: Evaluation, Fairness & Retention Prioritization Evaluation
> "Evaluate the downstream analytical pipeline of this project: the metric choices (Average Precision / PR-AUC under 10% imbalance), probability calibration design, demographic subgroup fairness analysis, and resource-constrained retention prioritization simulation (20% intervention capacity)."
