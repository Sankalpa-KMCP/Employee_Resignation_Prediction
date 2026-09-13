---
title: "Design and Evaluation of Machine Learning Models for Employee Resignation Risk and Resource-Constrained Retention Prioritization"
subtitle: "Canonical project specification, live state, and AI handoff memory"
author: "IT2011 Project Group"
date: "2026-09-12"
version: "4.6.2"
---

# AI entry point - read this first

This project context is intentionally maintained as one canonical Markdown file for AI compatibility, project continuity, and auditability.

```yaml
ai_handoff_entry:
  canonical_file: Employee_Resignation_Prediction_AI_Project_Context_v4_6_2.md
  canonical_status: Markdown v4.6.2 is ACTIVE and CANONICAL. It is the sole persistent project-context artifact.
  canonical_rule: Read this Markdown file as the authoritative single source of truth for persistent project context.
  conflict_rule: If repository code, results, or secondary documentation conflict with this file, apply the authority hierarchy, report the mismatch, and update this canonical file only through the logged decision/update protocol.
  repository_url: https://github.com/Sankalpa-KMCP/Employee_Resignation_Prediction
  active_target: Resigned
  positive_class: true
  active_model_registry:
    mandatory_shared_baselines:
      - DummyClassifier(strategy="prior")
      - DummyClassifier(strategy="most_frequent")
    logistic_regression_benchmark:
      - Logistic Regression (required for the PCA demonstration; otherwise optional)
    required_advanced_models:
      - Member 1: FT-Transformer (rtdl_revisiting_models.FTTransformer with PyTorch)
      - Member 2: CatBoost Classifier
      - Member 3: XGBoost Classifier (xgboost.XGBClassifier)
      - Member 4: LightGBM Classifier (lightgbm.LGBMClassifier)
      - Member 5: Explainable Boosting Machine (ExplainableBoostingClassifier)
      - Member 6: TabNet Classifier (pytorch_tabnet.tab_model.TabNetClassifier)
    individual_ownership_rule: Each member owns at least one required advanced model and must be able to explain, train, tune, evaluate, and defend that model.
    shared_experiment_rule: All six headline models use the same saved development/holdout rows and the same five outer stratified folds, with leakage-safe model-specific preprocessing.
    superseded_core_models:
      - Random Forest Classifier
      - Decision Tree Classifier
      - RBF-SVM
  primary_prediction_metric: Average Precision / PR-AUC
  primary_capacity: 0.20
  formal_holdout_rule: From the formal protocol freeze onward, never use the formal holdout for preprocessing design, tuning, early stopping, calibration choice, threshold choice, proxy choice, capacity choice, strategy choice, or model selection. Preserve the historical preliminary-audit exposure caveat.
  execution_context:
    conceptual_questions: This canonical Markdown file is sufficient for persistent project-context understanding.
    data_dependent_work: Also require the exact assigned CSV and checksum verification.
    implementation_work: Use the canonical GitHub repository and inspect the relevant current files before making code-dependent claims.
    generated_outputs: Experiment outputs are repository artefacts created during implementation; they are not missing context inputs and must never be invented.
  current_methodology_change: The strict nested model-selection methodology introduced and user-approved in v4.5.0 remains active in v4.6.2 for final headline-model evaluation. For every outer fold, all hyperparameter selection, class/loss-weight selection, preprocessing design, and early stopping must use only that outer fold's training partition. The outer-validation fold remains untouched until the fold-specific configuration is selected. The six headline models remain FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, and TabNet. Dummy baselines remain mandatory; Logistic Regression remains required only for the PCA demonstration benchmark and otherwise optional.
```

**Fast operational summary:** Build and compare six required advanced models under the same locked development/holdout rows and five outer stratified folds: FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, and TabNet. Each member owns one model. Use model-specific, fold-fitted preprocessing; never force all six models onto one transformed matrix. Conduct mandatory PCA (Principal Component Analysis) dimensionality-reduction analysis on the 13 approved numerical predictors with scree/variance plots, component loadings inspection, and a required PCA demonstration benchmark using Logistic Regression (comparing StandardScaler -> LogisticRegression vs StandardScaler -> PCA(>=95% variance) -> LogisticRegression). Broader standalone Logistic Regression exploration remains optional; Logistic Regression is not a seventh headline model. Headline tree/additive/deep models retain their approved non-PCA representations. Retain two DummyClassifier strategies as mandatory no-skill references. Select by development evidence, not model reputation or complexity. Assess calibration before using probabilities in the retention-prioritization score. Do not claim real employee worth, causal resignation drivers, intervention success, a real monetary budget, operational readiness, legal fairness, or suitability for actual employment decisions.

# Document map

- **AI entry point** - canonical-file rule, active model registry, individual ownership, and artefact requirements.
- **0. AI execution contract** - authority, mandatory behaviour, startup check, and anti-hallucination rules.
- **1. Machine-readable project memory** - structured project, dataset, methodology, live state, requirements, and update rules.
- **2. AI quick reference** - one-minute summary, terminology, and prohibited assumptions.
- **3. Current operating state** - milestone, completed work, blockers, risks, and next actions.
- **4-7. AI operating guidance** - task routing, canonical-file manifest, dependencies, maintenance, and update protocol.
- **8. Full canonical project specification** - requirements, dataset evidence, methodology, models, evaluation, ethics, deliverables, team plan, results templates, and logs.
- **9. Embedded normalized requirement extracts** - project-relevant requirements from the supplied formal PDFs, excluding nonessential administrative timing and submission logistics.
- **10. Reusable AI update templates** - live-state, experiment, AI-use, and decision-change records.


# 0. AI execution contract

## 0.1 Document purpose

This file is the persistent AI-first project memory for **Design and Evaluation of Machine Learning Models for Employee Resignation Risk and Resource-Constrained Retention Prioritization**. It is intended to let a new AI assistant, team member, reviewer, or future project session understand the project without relying on conversational memory.

**VERIFIED** - The project is an IT2011 group assignment for six members. It requires an end-to-end AI/ML solution, at least two suitable algorithms, Python implementation, hyperparameter tuning, pipelines, k-fold cross-validation, model comparison, ethical evaluation, a 15-minute proposal presentation, and a final report of no more than 15 pages.

**DECIDED** - The project retains `Resigned` as the binary target and now has two linked analytical stages:

1. estimate whether the available employee-related variables contain usable out-of-sample signal for `Resigned=True`; and
2. simulate how limited HR retention capacity could be prioritized when only a fixed share of records can be reviewed or supported.

**DECIDED** - The primary capacity scenario is the top 20% of records. The project compares random selection, risk-only ranking, value-proxy-only ranking, and a combined risk-plus-retention-value-proxy ranking. Capacity sensitivity at 10% and 30% is reported when feasible.

**DECIDED** - The retention-value measure is explicitly a **proxy**, not actual employee worth. The primary proxy is an equal-weight average of development-fitted min-max normalized `Performance_Score`, `Monthly_Salary`, and `Years_At_Company`. A no-salary proxy and a normalized multiplicative proxy are sensitivity analyses because salary is deterministically related to job title and performance in this dataset.

**DECIDED** - The six required advanced models are FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, and TabNet, with one headline model owned by each member. Member 1 owns FT-Transformer; Member 2 CatBoost; Member 3 XGBoost; Member 4 LightGBM; Member 5 EBM; Member 6 TabNet. DummyClassifier remains a mandatory shared no-skill reference. Logistic Regression is required for the PCA demonstration benchmark and otherwise optional; it is not a headline model.

**DECIDED** - The assessed build remains an academic, reproducible Python study. It does not estimate whether a real retention intervention would succeed, does not perform genuine monetary budget optimization, and must not be used for real employment actions. Academic record-level ranking is permitted only inside the controlled simulation on the assigned dataset.

## 0.2 Canonical-source rule

1. This Markdown file is the **sole canonical persistent project-context artifact**.
2. Do not create or maintain a parallel mirrored context file in another document format.
3. Code, notebooks, results, reports, and slides must not silently override a **DECIDED** item in this file. A change must be logged first.
4. The assigned CSV is identified by filename and SHA-256. Never substitute a similarly named dataset without an explicit, logged decision and checksum update.
5. Secondary project documents such as `README.md` and `EVALUATION_GUIDE.md` may summarize this context but must not become competing sources of truth.

## 0.3 Authority hierarchy

Apply sources in this order:

1. Formal assignment specification, marking rubric, or official lecturer/module instruction.
2. Verified contents of the exact uploaded project files and assigned dataset.
3. Explicit user or group decisions recorded as **DECIDED**.
4. Existing approved decision records.
5. Evidence-based recommendations.
6. General conventions.

A lower-authority source must not silently override a higher-authority source.

## 0.4 Mandatory AI behaviour

An AI assistant working on this project must:

- read Sections 0-7 and the normalized source policy in Section 9 before doing project-dependent work;
- complete the startup acknowledgement and identify which source/data/repository files are required for the requested task;
- verify the dataset checksum before data-dependent work;
- preserve the raw CSV unchanged;
- distinguish **VERIFIED**, **DECIDED**, **RECOMMENDED**, **PLACEHOLDER - HUMAN OR EXTERNAL INPUT REQUIRED**, **TO BE GENERATED**, **LIVE RECORD**, and **SUPERSEDED** information;
- use strict nested development evaluation for final headline-model evidence: within each outer fold, all tuning, class/loss-weight selection, preprocessing choices, and early stopping use only that outer fold's training partition; the outer-validation fold is prediction-only after fold-specific selection;
- avoid claiming causality, operational readiness, legal compliance, or real-world employee provenance;
- state when a requested conclusion is unsupported by the data;
- update the live state, result record, AI-use record, and decision log after material work;
- report checksum mismatches, schema changes, requirement conflicts, or material divergence between this canonical context and repository evidence before proceeding.

## 0.5 First-pass validation checklist

Before substantive work, check:

- [ ] The requested task is within the approved academic scope.
- [ ] The normalized requirement extracts in Section 9 are available when written requirements matter.
- [ ] Original PDFs are requested only when exact visual appearance, annotations, or a newer/differently hashed revision must be verified.
- [ ] The dataset file is exactly `Extended_Employee_Performance_and_Productivity_Data.csv`.
- [ ] The dataset SHA-256 is `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb`.
- [ ] The dataset has 100,000 rows and 20 columns before transformation.
- [ ] The target is `Resigned`, with `True` as the positive class.
- [ ] The two-stage scope is preserved: resignation-risk evaluation followed by resource-constrained prioritization simulation.
- [ ] The active model registry is mandatory Dummy baselines, a required Logistic Regression PCA demonstration benchmark (with broader standalone Logistic Regression otherwise optional), and six required advanced models: FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, and TabNet.
- [ ] Random Forest, Decision Tree, and RBF-SVM appear only as historical references unless a new logged decision reinstates them.
- [ ] All six required models use identical saved rows/folds but model-specific leakage-safe representations; final headline-model evaluation uses strict nested outer-CV/inner-selection so each outer-validation fold remains independent of tuning. Logistic Regression is required only for the PCA benchmark and otherwise optional.
- [ ] The primary capacity is 20%; 10% and 30% are sensitivity scenarios, not separate tuned targets.
- [ ] The retention-value measure is called a proxy and is not described as actual employee worth, replacement cost, or financial contribution.
- [ ] Proxy normalization parameters are fitted on development-training data only and then applied to validation/formal-holdout records.
- [ ] From the formal protocol freeze onward, the formal 20% holdout is unavailable to tuning, calibration choice, threshold selection, proxy-definition selection, capacity selection, strategy selection, or model selection; the historical preliminary-audit exposure caveat remains disclosed.
- [ ] Any holdout claim preserves the historical preliminary-audit exposure limitation; it is not described as guaranteed pristine.
- [ ] Any result being discussed is either mechanically reproduced or labelled **TO BE GENERATED**.
- [ ] The current administrative placeholders have not been guessed.
- [ ] AI-assisted work is being recorded in the actual-use live record.

## 0.6 Conflict-resolution protocol

When two sources conflict:

1. Record the conflicting statements and source IDs.
2. Apply the authority hierarchy.
3. Preserve the rejected value in the change log as **SUPERSEDED** when it was previously active.
4. Explain the effect on code, data, results, report, slides, and timeline.
5. If the conflict requires lecturer or supervisor authority, create a precise **PLACEHOLDER - HUMAN OR EXTERNAL INPUT REQUIRED** and continue only with work that is not blocked.

## 0.7 Anti-hallucination rules

Do not invent:

- team-member names, availability, ownership, or contribution percentages;
- dataset units, original organization, collection procedure, sampling frame, population, consent process, or verified real-world status;
- model scores, selected hyperparameters, calibration results, ranking results, confidence intervals, fairness findings, feature rankings, or conclusions before execution;
- claims that the model is suitable for employment decisions;
- claims that the retention-value proxy measures actual employee worth, revenue contribution, replacement cost, or intervention benefit;
- claims that the top 20% represents a real organization's budget rather than a hypothetical equal-cost capacity constraint;
- claims that selecting a record would prevent resignation, because the dataset contains no intervention-outcome evidence;
- claims that the combined ranking is mathematically or financially optimal;
- a licence different from the licence displayed by the Kaggle listing;
- missing-value or outlier treatments when the verified file does not require them.

## 0.8 Status-label definitions

| Label | Meaning and handling |
| --- | --- |
| VERIFIED | Directly confirmed from a formal source, authoritative external source, or mechanical inspection. |
| DECIDED | Approved project decision. Follow it until explicitly changed and logged. |
| RECOMMENDED | Evidence-based proposal not yet approved. Do not present it as fixed. |
| PLACEHOLDER - HUMAN OR EXTERNAL INPUT REQUIRED | Unresolved and unsafe to invent. State how to obtain the answer and whether it blocks work. |
| TO BE GENERATED | Requires future implementation, analysis, testing, or experimentation. Never populate with illustrative numbers. |
| LIVE RECORD | Operational information that must be updated as the project changes. |
| SUPERSEDED | Retained for audit history but no longer active. |

## 0.9 Preservation and scope-change rules

- Preserve every **DECIDED** item unless a higher-authority source requires a change.
- The 2026-08-01 expansion from resignation prediction alone to prediction plus constrained retention-prioritization simulation is an approved major scope change and is recorded in Section 8.36.
- FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, and TabNet are the approved required advanced models. DummyClassifier is mandatory. Logistic Regression is required for the PCA demonstration benchmark and otherwise optional; it is not a seventh headline model. Do not add or substitute Random Forest, Decision Tree, RBF-SVM, TabPFN, stacking, additional targets, deployment components, causal intervention models, or monetary optimization engines without a new logged pre-holdout decision.
- A target pivot after viewing formal holdout results is prohibited. A justified pivot before formal holdout evaluation requires a logged group or lecturer decision.
- Any change to the 20% primary capacity, value-proxy definition, ranking strategies, or evaluation metrics must be chosen from development evidence only and logged before formal-holdout access.
- Do not create a second persistent context mirror; keep all canonical context updates in this Markdown file.

## 0.10 Mandatory AI startup acknowledgement

Before performing project-dependent work, an AI must provide a compact **Context acknowledgement** containing:

1. project objective and the two-stage build;
2. current phase and immediate milestone;
3. target column and positive class;
4. active six-model registry, individual member ownership, model-specific representations, and primary prediction metric;
5. primary prioritization capacity, strategies, proxy definition, and ranking metrics;
6. formal holdout restriction and the historical preliminary-audit exposure caveat;
7. principal provenance or data-quality limitation;
8. prohibited real-world operational use and absence of intervention-effect evidence;
9. unresolved placeholders relevant to the requested task;
10. files or evidence currently available; and
11. the next action it will perform.

This acknowledgement is a parsing check, not a request for the user to repeat decisions. Do not ask the user to reconfirm a **DECIDED** item unless higher-authority evidence conflicts with it.

## 0.11 Stable context, live state, and canonical-file handling

- Sections 0, 2, 4-7, 8.1-8.30, 8.38, and 9 are stable project context unless changed through the decision protocol. Sections 8.31-8.37 are live/status records.
- Section 1 `live_state`, Section 3, and the dated status/result/AI-use/change records are live information.
- A newer dated **LIVE RECORD** overrides an older status statement, but it cannot silently override a formal requirement or **DECIDED** item.
- Every live update must include status, `last_updated`, updater/evidence, and owner when known.
- This canonical Markdown file is sufficient for persistent conceptual project understanding. Data-dependent execution still requires the exact CSV; implementation-dependent work must inspect the canonical repository and the specific files relevant to the task.

# 1. Machine-readable project memory

```yaml
document_control:
  context_schema_version: '4.6'
  project_context_version: 4.6.2
  context_release_id: ERPC-20260912-V4.6.2
  generated_at: 2026-09-12 Asia/Colombo
  canonical_file: Employee_Resignation_Prediction_AI_Project_Context_v4_6_2.md
  substantive_content_rule: This Markdown file is the sole canonical persistent project-context artifact.
  stable_context_sections:
  - '0'
  - '2'
  - '4'
  - '5'
  - '6'
  - '7'
  - '8.1-8.30'
  - '8.38'
  - '9'
  live_state_sections:
  - 1.live_state
  - '3'
  - '8.31'
  - '8.32'
  - '8.33'
  - '8.35'
  - '8.36'
  - '8.37'
  status_precedence: A newer dated LIVE RECORD overrides an older status statement, but cannot override formal requirements
    or DECIDED items without a logged change.
  ai_ingestion_priority:
  - AI entry point
  - 0-7 operating contract and live state
  - 1 machine-readable memory
  - 8 task-relevant specification
  - 9 embedded normalized requirement extracts
project:
  name: Design and Evaluation of Machine Learning Models for Employee Resignation Risk and Resource-Constrained
    Retention Prioritization
  slug: Employee_Resignation_Prediction
  repository_url: https://github.com/Sankalpa-KMCP/Employee_Resignation_Prediction
  type: academic two-stage AI/ML classification, ranking, and critical evaluation project
  domain: human resources and workforce analytics
  module:
    code: IT2011
    name: Artificial Intelligence and Machine Learning
    faculty: Faculty of Computing
    academic_period: Year 2 Semester 1 (2026)
  team_size: 6
  primary_objective: Evaluate whether the assigned employee dataset contains usable out-of-sample signal for Resigned=True
    and whether that signal can support a transparent simulation of limited-capacity retention prioritization better
    than simple baselines.
  problem_statement: Use supervised binary classification to estimate resignation risk, then compare resource-constrained
    ranking strategies under a hypothetical equal-cost capacity limit without claiming causality, actual employee
    worth, intervention effectiveness, operational readiness, legal compliance, or verified real-employee provenance.
  build_definition:
    primary_build: End-to-end Python workflow for binary resignation-risk modelling and constrained retention-prioritization
      simulation.
    required_components:
    - immutable raw-data preservation and checksum/schema validation
    - reproducible preprocessing and exploratory data analysis
    - mandatory PCA dimensionality-reduction analysis on standardized approved numerical predictors (variance ratios, scree/projection plots, loadings) and a required PCA demonstration benchmark via Logistic Regression (StandardScaler -> LogisticRegression vs StandardScaler -> PCA(>=95% variance) -> LogisticRegression)
    - shared split/fold registry and model-specific preprocessing adapters
    - DummyClassifier non-learning baselines and a Logistic Regression benchmark required for the PCA demonstration (broader standalone use otherwise optional)
    - FT-Transformer advanced deep tabular pipeline using fold-local numerical normalization, categorical integer
      vocabularies/embeddings, PyTorch training controls, and training-only inner validation
    - CatBoost Classifier advanced gradient-boosted predictive pipeline with native categorical handling
    - XGBoost Classifier advanced gradient-boosted predictive pipeline with fold-local one-hot categorical encoding
      and histogram tree training
    - LightGBM Classifier advanced gradient-boosted predictive pipeline with fold-local categorical handling and
      bounded tree search
    - Explainable Boosting Machine advanced inherently interpretable nonlinear pipeline with explicit feature types
    - TabNet Classifier advanced attentive deep tabular pipeline with fold-local category codes/embeddings, numerical
      scaling, and training-only inner validation
    - strict nested hyperparameter selection within the saved five-fold outer development cross-validation
    - calibration assessment because predicted probabilities feed the prioritization score
    - one-time formal-holdout corroboration after model, calibration, proxy, capacity, and strategy freeze
    - top-20-percent capacity simulation with 10-percent and 30-percent sensitivity
    - random, risk-only, value-only, and risk-times-value-proxy strategy comparison
    - prediction metrics, ranking-at-capacity metrics, error analysis, interpretability, ethics, and subgroup audit
    - saved code, configurations, results, figures, model artefacts, proposal content, report evidence, and viva
      preparation
    explicitly_not_required:
    - website or mobile application
    - dashboard or graphical user interface
    - API, database, cloud service, or production deployment
    - real monetary budget optimization
    - causal estimation of retention intervention effectiveness
    - unapproved additional model families, stacking, or model-family substitution
    - real-world automated or individual employment action
    success_definition: Correct, reproducible, critically evaluated execution satisfies the project even if no model
      materially exceeds the dummy baseline or no prioritization strategy exceeds random selection.
live_state:
  status: active
  current_phase: methodology harmonization and remaining headline-model completion before downstream comparison
  current_milestone: align all model configs and implementations with the approved strict nested outer-CV / inner-selection protocol, complete the remaining headline-model evaluations, then unlock model comparison
  last_updated: '2026-09-11'
  updated_by: ChatGPT-assisted canonical-context maintenance using verified live-repository audit evidence
  evidence:
  - SRC-001
  - SRC-002
  - SRC-004
  - SRC-013
  - SRC-014
  - SRC-015
  - SRC-016
  completed:
  - formal requirement extraction
  - dataset audit and data contract
  - canonical GitHub repository URL recorded
  - six required advanced models selected and assigned one per member
  - frozen stratified 80/20 development/formal-holdout split and five 16,000-row development folds
  - development-only exploratory data analysis with 20 canonical outputs
  - shared Dummy baselines and Logistic/PCA benchmark evidence
  - PCA analysis executed with saved explained-variance, loadings, comparison, and summary artifacts; final verification cell still pending
  - shared evaluation metric utilities and OOF validation utilities implemented
  - all model adapters implemented under src/models/
  - FT-Transformer development 5-fold result artifacts generated
  - XGBoost development 5-fold result artifacts generated using the current fixed configuration
  - TabNet eight-candidate development evaluation completed and T01 selected with seed-sensitivity evidence
  - EBM development results exist upstream on origin/main and require local integration/verification before canonical local completion
  - automated test suite expanded to 186 tests across 14 modules (186/186 passing)
  - single-file Markdown canonical-context policy adopted; prior mirrored-context workflow retired
  active_next_actions:
  - harmonize XGBoost, LightGBM, EBM, and FT-Transformer configs with the approved bounded nested-selection protocol (CatBoost v2 harmonized)
  - decide implementation placement for tree-model inner validation/early stopping without changing the methodological requirement
  - perform methodology-complete nested evaluation for FT-Transformer, XGBoost, and TabNet before treating their current CV scores as final unbiased headline estimates
  - complete LightGBM formal nested evaluation (CatBoost formal development evaluation is complete)
  - integrate and verify EBM upstream results, then rerun under the final protocol if required
  - execute model comparison, calibration, fairness analysis, and retention prioritization only after development-model evidence is protocol-complete
  blockers:
  - team names and final ownership mapping are not supplied in the context
  - real intervention costs, outcomes, replacement costs, and business value are unavailable; monetary optimization claims remain blocked
  technical_work_blocked: false
authority_hierarchy:
- formal assignment specification, marking rubric, or official lecturer/module instruction
- verified contents of the exact uploaded source files and assigned dataset
- explicit user/group decisions recorded as DECIDED
- approved decision log
- evidence-based recommendations
- general conventions
source_files:
- id: SRC-001
  name: Group Assignment Specification.pdf
  sha256: 7bcd43f095c4e76d89d941ee6c28da228260b6cebb70ae56c623e6d55867b481
  authority: formal assignment specification
  embedded_requirement_extract: Section 9.2
- id: SRC-002
  name: Proposal Presentation.pdf
  sha256: 7cbf695c122b194880beb62a672b15eb56eaa470d43bf286a3794e9c10abb4d3
  authority: formal proposal instruction
  embedded_requirement_extract: Section 9.3
- id: SRC-003
  name: archive.zip
  sha256: abe4a9aa869fe553faec813a06ff93432900d72af2036058f8b84e3dd64db375
  authority: assigned uploaded dataset archive
- id: SRC-004
  name: Extended_Employee_Performance_and_Productivity_Data.csv
  sha256: 2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb
  authority: mechanically inspected assigned dataset
- id: SRC-005
  name: Kaggle dataset page
  url: https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data
  authority: third-party source link and licence metadata
- id: SRC-006
  name: User approval in project conversation
  date: '2026-07-28'
  authority: approval of project recommendations and updated context generation
- id: SRC-007
  name: User-approved assignment-aligned project title and context update
  date: '2026-07-29'
  authority: explicit user decision to use the resignation-prediction title and regenerate the project AI handoff
- id: SRC-008
  name: User-approved retention-prioritization scope expansion
  date: '2026-08-01'
  authority: explicit user decision to extend the project while retaining Resigned as the target
- id: SRC-010
  name: User-approved advanced-model and AI-handoff update
  date: '2026-08-05'
  authority: explicit user decision
  statement: Use Dummy and Logistic Regression as references, CatBoost and EBM as required advanced models, and
    update the project AI context.
  status: SUPERSEDED BY SRC-012
- id: SRC-011
  name: User-supplied canonical repository and v4.1.1 cleanup instruction
  date: '2026-08-06'
  authority: explicit user decision
  statement: Record the GitHub repository, remove administrative and generated-output items from unresolved handoff
    requirements, and correct defects in the then-current project handoff package.
  status: HISTORICAL CONTEXT-FILE WORKFLOW SUPERSEDED BY SRC-016; repository and cleanup decisions remain retained.
- id: SRC-012
  name: User-approved three-model and three-pair methodology update
  date: '2026-08-06'
  authority: explicit user decision
  statement: Use CatBoost, Explainable Boosting Machine, and FT-Transformer as the three required advanced models,
    one per two-member pair; retain DummyClassifier as the shared mandatory baseline and Logistic Regression only
    as an optional lightweight benchmark; update the project AI context.
  status: SUPERSEDED BY SRC-013 FOR ACTIVE MODEL REGISTRY AND TEAM OWNERSHIP
- id: SRC-013
  name: User-approved six-model individual-ownership update
  date: '2026-08-07'
  authority: explicit user decision
  statement: 'Use six required advanced models with one headline model per member: Member 1 FT-Transformer, Member
    2 CatBoost, Member 3 XGBoost, Member 4 LightGBM, Member 5 EBM, Member 6 TabNet. Retain DummyClassifier baselines
    and optional Logistic Regression. Keep one shared split/fold/evaluation protocol.'
- id: SRC-014
  name: Lecturer / Course Requirement for Dimensionality Reduction (PCA)
  date: '2026-08-19'
  authority: module lecturer / course instruction
  statement: 'Principal Component Analysis (PCA) dimensionality reduction must be demonstrated in the project on numerical features, including variance retention analysis, component loadings interpretation, and a PCA-augmented benchmark pipeline comparison.'
- id: SRC-015
  name: User-approved strict nested evaluation and 2026-09-11 live-repository audit
  date: '2026-09-11'
  authority: explicit user methodology decision plus verified current repository evidence
  statement: For final headline-model evaluation, every outer fold must keep its outer-validation rows completely uninvolved in hyperparameter selection, class/loss-weight selection, preprocessing decisions, and early stopping. All such decisions must use only the corresponding outer-training partition. Current FT-Transformer, XGBoost, TabNet, and upstream EBM results remain valid development evidence but are not automatically the final unbiased nested-CV headline estimates.
- id: SRC-016
  name: User-approved single-file Markdown canonical-context policy
  date: '2026-09-11'
  authority: explicit user project-workflow decision
  statement: Maintain exactly one persistent canonical project-context file in Markdown. The previous mirrored-document workflow is retired and must not be regenerated or treated as part of the active handoff package.
embedded_source_policy:
  normalized_requirement_extracts_included: true
  complete_verbatim_transcripts_included: false
  extract_use: Sufficient for ordinary project planning and requirement alignment. The original PDFs remain authoritative.
  deliberately_excluded_from_active_handoff:
  - exact administrative deadlines and tentative dates
  - submission logistics not needed for technical execution
  - institution-specific citation-style or AI-policy text not present in the supplied technical requirements
  - unverified lecturer feedback or proposal-approval claims
  original_pdf_needed_when:
  - checking exact wording, layout, signatures, stamps, annotations, or page appearance
  - verifying a newer or differently hashed revision
  - resolving a conflict between the normalized extract and an official source
requirement_register:
- id: REQ-001
  requirement: Use the assigned dataset; dataset selection is not required.
  source: SRC-001 and SRC-002
  status: VERIFIED
- id: REQ-002
  requirement: Identify a real-world problem domain aligned with the dataset.
  source: SRC-001 and SRC-002
  status: VERIFIED
- id: REQ-003
  requirement: Describe dataset source, structure, target, size, quality, and suitability.
  source: SRC-001 and SRC-002
  status: VERIFIED
- id: REQ-004
  requirement: Clean/preprocess data, perform EDA with visualizations, and report insights.
  source: SRC-001
  status: VERIFIED
- id: REQ-005
  requirement: Select and justify at least two suitable AI/ML algorithms.
  source: SRC-001
  status: VERIFIED
- id: REQ-006
  requirement: Implement in Python with hyperparameter tuning and model pipelines.
  source: SRC-001
  status: VERIFIED
- id: REQ-007
  requirement: Use relevant metrics, perform k-fold cross-validation, compare models, and justify the best performer.
  source: SRC-001
  status: VERIFIED
- id: REQ-008
  requirement: Prepare a professional group report of no more than 15 pages in PDF format.
  source: SRC-001
  status: VERIFIED
- id: REQ-009
  requirement: Include the nine specified report sections, ethics/bias mitigation, reflections, references, and
    AI Tool Usage Declaration.
  source: SRC-001
  status: VERIFIED
- id: REQ-010
  requirement: Proposal must be PowerPoint, 15 minutes per group, in class during lab sessions.
  source: SRC-002
  status: VERIFIED
- id: REQ-011
  requirement: Proposal slides must include dataset source/link, problem domain, dataset justification, and planned
    AI use.
  source: SRC-002
  status: VERIFIED
- id: REQ-012
  requirement: Proposal scheduling is an external administrative detail; consult the current official module channel
    when needed.
  source: SRC-002
  status: EXCLUDED FROM ACTIVE TECHNICAL HANDOFF
- id: REQ-013
  requirement: AI disclosure must state tool/version if known, supported tasks, extent, verification, and group
    ownership.
  source: SRC-001
  status: VERIFIED
- id: REQ-014
  requirement: Demonstrate Principal Component Analysis (PCA) dimensionality reduction on the 13 approved numerical predictors, evaluate eigenvalues and cumulative explained variance, generate scree and projection plots, inspect component loadings, and execute a required PCA demonstration benchmark via Logistic Regression (StandardScaler -> LogisticRegression vs StandardScaler -> PCA(>=95% variance) -> LogisticRegression) under 5-fold CV without forcing PCA onto headline models.
  source: SRC-014
  status: VERIFIED requirement — RESULTS GENERATED; FINAL VERIFICATION PENDING (04a_pca_analysis.ipynb executed except final verification cell; results/pca/ populated)
dataset:
  filename: Extended_Employee_Performance_and_Productivity_Data.csv
  sha256: 2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb
  rows: 100000
  columns: 20
  format: CSV; ASCII text and UTF-8 compatible
  delimiter: ','
  header: true
  missing_values: 0
  duplicate_rows: 0
  duplicate_employee_ids: 0
  licence: 'CC0: Public Domain according to the Kaggle listing'
  provenance: Original organization, collection, sampling, and generation process are undocumented. Mechanical patterns
    are consistent with constructed/synthetic-like data, but this is an inference.
  target:
    column: Resigned
    positive_label: true
    negative_label: false
    mapping:
      'False': 0
      'True': 1
    class_counts:
      'False': 89990
      'True': 10010
    positive_prevalence: 0.1001
  ordered_columns:
  - name: Employee_ID
    raw_dtype: integer
    nullable: false
    role: identifier
    model_policy: exclude from every model; retain only for traceability and deterministic tie-breaking
  - name: Department
    raw_dtype: string/category
    nullable: false
    role: predictor
    model_policy: categorical predictor; native categorical for CatBoost; fold-local one-hot for XGBoost and the Logistic Regression benchmark; fold-local pandas/category-code handling for LightGBM; nominal feature type for EBM;
      fold-fitted integer vocabulary/embedding inputs for FT-Transformer and TabNet
  - name: Gender
    raw_dtype: string/category
    nullable: false
    role: sensitive predictor and audit attribute
    model_policy: sensitive categorical predictor and audit attribute; native categorical for CatBoost; fold-local
      one-hot for XGBoost and the Logistic Regression benchmark; fold-local pandas/category-code handling for LightGBM;
      nominal feature type for EBM; fold-fitted integer vocabulary/embedding inputs for FT-Transformer and TabNet;
      exclude in no-Gender sensitivity model
  - name: Age
    raw_dtype: integer
    nullable: false
    role: sensitive predictor and audit attribute
    model_policy: numeric sensitive predictor and audit attribute; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling for the Logistic benchmark where used; fold-local imputation/standardization for FT-Transformer
      and TabNet; exclude in no-Gender/no-Age sensitivity model
  - name: Job_Title
    raw_dtype: string/category
    nullable: false
    role: predictor
    model_policy: categorical predictor; native categorical for CatBoost; fold-local one-hot for XGBoost and the Logistic Regression benchmark; fold-local pandas/category-code handling for LightGBM; nominal feature type for EBM;
      fold-fitted integer vocabulary/embedding inputs for FT-Transformer and TabNet
  - name: Hire_Date
    raw_dtype: date-time text
    nullable: false
    role: audit field
    model_policy: parse for audit; exclude raw value from every primary model
  - name: Years_At_Company
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Education_Level
    raw_dtype: string/category
    nullable: false
    role: predictor
    model_policy: categorical predictor; native categorical for CatBoost; fold-local one-hot for XGBoost and the Logistic Regression benchmark; fold-local pandas/category-code handling for LightGBM; nominal feature type for EBM;
      fold-fitted integer vocabulary/embedding inputs for FT-Transformer and TabNet
  - name: Performance_Score
    raw_dtype: integer/ordinal
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Monthly_Salary
    raw_dtype: integer/numeric
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Work_Hours_Per_Week
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Projects_Handled
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Overtime_Hours
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Sick_Days
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Remote_Work_Frequency
    raw_dtype: integer percentage
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Team_Size
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Training_Hours
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Promotions
    raw_dtype: integer count
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Employee_Satisfaction_Score
    raw_dtype: float/numeric
    nullable: false
    role: predictor
    model_policy: retain as approved numeric/ordinal predictor; native numeric for CatBoost/LightGBM/EBM; fold-local
      numerical matrix for XGBoost; fold-local scaling where required by the Logistic benchmark, FT-Transformer, or TabNet
  - name: Resigned
    raw_dtype: boolean
    nullable: false
    role: target
    model_policy: map True=1 and False=0; never use as a feature or proxy input
  unknown_category_policy:
    validation: Report categories outside the verified register before fitting or inference.
    logistic_regression: OneHotEncoder(handle_unknown="ignore") inside the fold-specific pipeline.
    catboost: Pass named categorical columns as strings through pandas DataFrame/Pool; do not pre-one-hot encode.
    ebm: Pass explicit nominal/continuous feature types and preserve feature names; smoke-test unseen-category behaviour
      for the tested InterpretML environment.
    ft_transformer: Fit categorical vocabularies on each outer training fold only; reserve explicit missing and
      unknown integer codes; export cardinalities and mapping metadata; never infer vocabulary from outer validation
      or formal holdout.
    xgboost: Use fold-local OneHotEncoder(handle_unknown="ignore") in the primary protocol; do not fit categories
      on outer validation or formal holdout.
    lightgbm: Fit categorical dtype/category mapping on each outer training fold only; unseen validation/holdout
      values must map to missing/unknown handling and be reported.
    tabnet: Fit categorical vocabularies on each outer training fold only; reserve explicit missing/unknown integer
      codes and export cat_idxs/cat_dims metadata.
  row_identity_policy: Use Employee_ID only to save split membership and prediction traceability; never as a predictor.
audit_exposure:
  status: HISTORICAL LIMITATION
  purpose: Preliminary context-framing diagnostic only
  reported_metrics_present: true
  split_method: not retained in the prior audit record
  random_seed: not retained
  row_ids_or_indices: not retained
  preprocessing_details: not retained at repository-grade reproducibility level
  implication: A later 20% holdout can be locked against future tuning, but cannot be described as guaranteed pristine
    from all historical audit exposure.
  formal_policy: Use training-only five-fold stratified CV as the primary comparison evidence. Lock the formal 20%
    holdout after repository initialization, use it once after selection, label it formal holdout corroboration,
    and disclose the historical exposure limitation.
  prohibited_claim: Do not describe the formal holdout as guaranteed pristine or completely untouched from all historical exploratory exposure.
methodology:
  pca_protocol:
    purpose: Mandatory course-required unsupervised dimensionality reduction, variance explanation, and linear projection analysis.
    scope: Applied to all 13 approved numerical predictors (Age, Years_At_Company, Performance_Score, Monthly_Salary, Work_Hours_Per_Week, Projects_Handled, Overtime_Hours, Sick_Days, Remote_Work_Frequency, Team_Size, Training_Hours, Promotions, Employee_Satisfaction_Score).
    exclusions: Employee_ID (identifier), raw Hire_Date (unparsed datetime), and Resigned (target).
    scaling: StandardScaler fitted on development-training data only; never fitted across full dataset or formal holdout.
    unsupervised_nature: PCA is strictly unsupervised and does not use Resigned during component computation.
    outputs:
    - eigenvalues and explained variance ratio per component
    - cumulative explained variance curve
    - scree plot and 2D PC1 vs PC2 projection
    - component loadings matrix associating original numeric features to PCs
    selection_rule: For the predictive PCA demonstration benchmark, retain the minimum number of principal components needed to explain at least 95% (>= 0.95) of variance in the applicable outer-training fold. For 2D visualization and interpretation only, PC1 and PC2 are used.
    benchmark_policy:
    - "Logistic Regression is required specifically in a narrow capacity as the PCA demonstration benchmark (Pipeline A: StandardScaler -> LogisticRegression vs Pipeline B: StandardScaler -> PCA(>=95% variance) -> LogisticRegression)."
    - Broader standalone Logistic Regression experimentation remains optional.
    - Logistic Regression is not a seventh headline model; the six headline models remain unchanged.
    model_integration:
    - Primary headline models (CatBoost, XGBoost, LightGBM, EBM, FT-Transformer, TabNet) retain their approved model-specific representations without PCA to preserve categorical structure and interpretability.
    leakage_safeguards: Scaler and PCA transformations are fitted strictly on outer-training folds; formal holdout is never accessed for component selection or variance evaluation. Descriptive development-only PCA may use the full development partition only if clearly labeled descriptive, and must not leak into CV evaluation.
  split:
    development_train: 0.8
    formal_holdout: 0.2
    stratified: true
    random_seed: 42
    holdout_role: one-time corroboration after model/threshold freeze; historical audit-exposure caveat required
  cross_validation:
    type: StratifiedKFold
    n_splits: 5
    shuffle: true
    random_seed: 42
    scope: development partition only
    role: primary model-comparison evidence
    final_headline_evaluation_protocol: strict nested model selection
    nested_selection_rule: For each outer fold, every hyperparameter choice, class/loss-weight choice, preprocessing decision, candidate-model choice, and early-stopping/epoch decision must be made using only that outer fold's training partition. The corresponding outer-validation fold is prediction-only after fold-specific selection and must not influence any training or model-selection choice.
    inner_validation_for_iterative_models: Within each outer training partition, create training-only inner validation or inner cross-validation for tuning and early stopping. Never use the outer-validation fold or formal holdout to control training.
    historical_result_handling: Existing FT-Transformer, XGBoost, TabNet, and upstream EBM development results remain valid exploratory/development evidence. If their configuration selection used information that later participates in another outer fold's validation role, rerun under the strict nested protocol before calling the pooled score the final unbiased headline estimate.
  primary_metric: Average Precision / PR-AUC
  secondary_metrics:
  - ROC-AUC
  - precision
  - recall
  - F1
  - balanced accuracy
  - confusion matrix
  - accuracy with imbalance warning
  dummy_baselines:
  - strategy: prior
    purpose: constant prevalence probability and PR-AUC reference
  - strategy: most_frequent
    purpose: majority-class threshold metrics and accuracy reference
  model_registry:
    non_learning_baselines:
    - model: DummyClassifier
      strategy: prior
      role: prevalence-probability and PR-AUC reference
    - model: DummyClassifier
      strategy: most_frequent
      role: majority-threshold and accuracy reference
    logistic_regression_benchmark:
      model: Logistic Regression
      role: required narrow benchmark for the PCA demonstration; broader standalone use is optional; not a required advanced model and not an individually owned headline workstream
      execution_rule: The StandardScaler -> LogisticRegression and StandardScaler -> PCA(>=95% variance) -> LogisticRegression comparison is required. Additional standalone Logistic Regression experiments are optional and must not delay the six required advanced models.
    required_advanced_models:
    - member: Member 1
      model: FT-Transformer
      implementation: rtdl_revisiting_models.FTTransformer with PyTorch
      role: advanced attention-based deep tabular model using numerical and categorical feature tokens
    - member: Member 2
      model: CatBoost Classifier
      role: advanced gradient-boosted predictive model with native categorical handling
    - member: Member 3
      model: XGBoost Classifier
      implementation: xgboost.XGBClassifier
      role: advanced gradient-boosted tree model using a fold-local encoded matrix
    - member: Member 4
      model: LightGBM Classifier
      implementation: lightgbm.LGBMClassifier
      role: advanced leaf-wise gradient-boosted tree model with efficient tabular training
    - member: Member 5
      model: Explainable Boosting Machine
      implementation: interpret.glassbox.ExplainableBoostingClassifier
      role: advanced inherently interpretable nonlinear additive model
    - member: Member 6
      model: TabNet Classifier
      implementation: pytorch_tabnet.tab_model.TabNetClassifier
      role: advanced attentive deep-learning model for tabular data
    superseded_core_models:
    - Random Forest Classifier
    - Decision Tree Classifier
    - RBF-SVM
    addition_rule: No seventh headline model family, stacking, voting, or selected-model ensemble without a logged
      pre-holdout decision. Dummy baselines and the Logistic Regression PCA benchmark do not count toward the six headline models; broader standalone Logistic Regression remains optional.
  model_specific_preprocessing:
    shared:
    - exclude Employee_ID and raw Hire_Date from primary modelling
    - preserve identical row order, target mapping, development/holdout membership, and outer fold IDs
    - fit every imputer, encoder, scaler, binning rule, category vocabulary, and proxy normalizer only on the applicable
      training data
    - apply no target-derived preprocessing outside training folds
    logistic_regression_benchmark:
    - one-hot encode categorical variables with unknown-category handling
    - standardize approved numerical variables
    - use an sklearn Pipeline/ColumnTransformer fitted inside each outer training fold
    catboost:
    - use a pandas DataFrame or CatBoost Pool
    - pass categorical column names through cat_features
    - do not one-hot encode or standardize solely for CatBoost
    - fit CatBoost categorical statistics only within the outer training fold
    ebm:
    - use a pandas DataFrame with preserved feature names
    - declare nominal and continuous feature types explicitly
    - do not reuse the one-hot matrix or FT-Transformer integer codes
    - keep interactions bounded and report every retained interaction
    ft_transformer:
    - split approved predictors into continuous and categorical inputs
    - fit numerical imputation and standardization on each outer training fold only
    - fit a categorical vocabulary on each outer training fold, reserve explicit missing and unknown codes, and
      export category cardinalities
    - pass continuous features as floating tensors and categorical features as int64 tensors
    - do not one-hot encode categorical inputs for FT-Transformer
    - use a training-only inner validation subset for early stopping and epoch selection
    xgboost:
    - one-hot encode categorical variables inside each outer training fold with unknown-category handling
    - use an sklearn ColumnTransformer or equivalent fold-fitted encoder; do not build one full-data encoded matrix
    - pass numerical values in a dense/sparse matrix as appropriate; scaling is not required solely for tree boosting
    - use training-only inner validation if early stopping is enabled
    lightgbm:
    - use pandas DataFrame/category-aware input or a documented fold-local category-code adapter
    - fit category mappings on each outer training fold and preserve feature names
    - do not standardize solely for LightGBM
    - use training-only inner validation if early stopping is enabled
    tabnet:
    - construct a numeric matrix with continuous values and fold-local categorical integer codes
    - fit continuous imputation/standardization on each outer training fold only
    - fit categorical vocabularies on each outer training fold; reserve missing/unknown codes and export cat_idxs/cat_dims
    - use training-only inner validation for early stopping; never use outer validation/formal holdout to control
      training
  logistic_search:
    method: GridSearchCV or equivalent fold registry
    solver: liblinear unless the tested scikit-learn environment and parameter grid justify another compatible solver
    penalty:
    - l1
    - l2
    C:
    - 0.01
    - 0.1
    - 1
    - 10
    class_weight:
    - null
    - balanced
    max_iter: 2000
    status: required for the PCA demonstration benchmark; broader standalone search optional
    maximum_configurations: 8
  catboost_search:
    method: bounded randomized search or explicit candidate list using the shared five-fold registry
    maximum_configurations: 12
    fixed:
      loss_function: Logloss
      random_seed: 42
      verbose: false
      allow_writing_files: false
    search_space:
      iterations:
      - 300
      - 600
      - 1000
      depth:
      - 4
      - 6
      - 8
      learning_rate:
      - 0.03
      - 0.05
      - 0.1
      l2_leaf_reg:
      - 3
      - 5
      - 10
      random_strength:
      - 0
      - 1
      - 2
      auto_class_weights:
      - null
      - Balanced
    restriction: Do not use the formal holdout or an outer validation fold as an early-stopping control. Use fixed
      iterations during fair outer-fold comparison or create a training-only inner validation subset; document the
      rule consistently.
  ebm_search:
    method: bounded candidate grid or randomized search using the shared five-fold registry
    maximum_configurations: 10
    implementation: interpret.glassbox.ExplainableBoostingClassifier
    fixed:
      objective: log_loss
      random_state: 42
    search_space:
      learning_rate:
      - 0.005
      - 0.015
      - 0.03
      max_leaves:
      - 2
      - 3
      min_samples_leaf:
      - 2
      - 4
      - 10
      interactions:
      - 0
      - 5
      - 10
      max_rounds:
      - 5000
      - 15000
      early_stopping_rounds:
      - 100
      - 200
    interpretability_rule: Prefer fewer interactions when PR-AUC is practically indistinguishable; report retained
      main effects and interactions explicitly. Any internal validation used by EBM must remain inside the outer
      training fold.
  model_selection_rule: Rank all six required advanced models by mean strict nested-development-CV Average Precision using identical outer folds. Inspect fold stability, calibration, runtime, training failures, reproducibility, and explanation
    quality. Use a practical-equivalence rule rather than choosing a numerically higher score automatically. When
    performance is practically equivalent, prefer the model with stronger reproducibility, lower unnecessary complexity,
    and clearer valid interpretation. Freeze the complete model, preprocessing, calibration source, and threshold
    before formal holdout use.
  threshold_policy:
    reference_threshold: 0.5
    secondary_threshold: threshold maximizing F1 on pooled out-of-fold development predictions
    restriction: never select threshold from formal holdout
    reporting: report both thresholds and full precision/recall trade-off
  uncertainty:
    cv: mean, standard deviation, and fold values
    formal_holdout: 95% percentile bootstrap interval with 2000 stratified resamples and seed 42 when metric is
      defined
  calibration_rule: Generate out-of-fold probabilities for all six required advanced models; include the required PCA Logistic benchmark and any broader optional Logistic run as applicable. Compare Brier score and reliability curves. If development evidence shows material miscalibration, compare development-only sigmoid or isotonic calibration using cross-fitted or otherwise out-of-sample calibration assessment; do not fit and evaluate a calibrator on the same targets. Because class/loss weighting can alter probability calibration, assess weighted and unweighted probability behaviour explicitly. Freeze the probability source before formal holdout access.
  usable_signal_rule: Prediction signal is potentially useful only when CV Average Precision exceeds the prior baseline
    consistently, the absolute mean improvement is at least 0.01, uncertainty is not dominated by baseline performance,
    and no serious subgroup failure invalidates the claim. Prioritization is potentially useful only when the risk-based
    strategies also show stable lift over repeated random selection at the same capacity. These are project rules,
    not universal thresholds.
  no_signal_rule: If prediction models remain near baseline or risk-based prioritization does not exceed random
    selection, report a defensible negative result and reject operational use. Do not allow the value proxy alone
    to disguise an uninformative resignation model.
  prioritization:
    purpose: Academic simulation of limited equal-cost HR review/retention capacity; not real budget optimization.
    primary_capacity_fraction: 0.2
    sensitivity_capacity_fractions:
    - 0.1
    - 0.3
    capacity_selection_rule: Pre-registered; never chosen by formal-holdout performance.
    primary_value_proxy:
      name: equal-weight normalized performance-salary-tenure proxy
      components:
      - Performance_Score
      - Monthly_Salary
      - Years_At_Company
      normalization: MinMaxScaler fitted on development-training data only
      formula: mean(normalized Performance_Score, normalized Monthly_Salary, normalized Years_At_Company)
      interpretation: Illustrative retention-value proxy only; not actual employee worth.
    sensitivity_value_proxies:
    - name: no-salary proxy
      formula: mean(normalized Performance_Score, normalized Years_At_Company)
      reason: Tests the effect of deterministic salary redundancy and avoids counting performance indirectly through
        salary.
    - name: normalized multiplicative proxy
      formula: product(epsilon-adjusted normalized Performance_Score, Monthly_Salary, Years_At_Company)
      reason: Tests the user-originated performance-times-salary-times-tenure concept without raw-scale domination
        or automatic zeroing for zero tenure.
    priority_score: calibrated resignation probability multiplied by the selected retention-value proxy
    comparison_strategies:
    - repeated random selection at the same capacity
    - top-k by calibrated resignation risk
    - top-k by retention-value proxy
    - top-k by risk-times-value-proxy priority score
    - reference probability-threshold strategy with its actual selected fraction reported
    primary_metrics:
    - precision_at_capacity
    - recall_at_capacity
    - lift_at_capacity
    - captured_positive_count
    - value_weighted_recall_at_capacity
    restriction: No claim that selected records will be retained, that proxy value will be protected, or that the
      strategy is financially optimal.
  ft_transformer_search:
    method: bounded staged candidate search using the shared five-fold registry; no full Cartesian grid
    maximum_configurations: 12
    implementation: rtdl_revisiting_models.FTTransformer with PyTorch
    fixed:
      output_dimension: 1
      loss: BCEWithLogitsLoss
      optimizer: AdamW using model.make_parameter_groups()
      ffn_d_hidden_multiplier: 1.3333333333333333 # 4/3 multiplier for ReGLU activation (explicit project configuration required by FTTransformerBackbone constructor)
      residual_dropout: 0.0 # explicit project configuration (required by FTTransformerBackbone constructor)
      random_seed: 42
      maximum_epochs: 100
      early_stopping_metric: inner-validation Average Precision
      early_stopping_patience: 12
    search_space:
      n_blocks:
      - 2
      - 3
      d_block:
      - 64
      - 128
      - 192
      attention_n_heads:
      - 4
      - 8
      attention_dropout:
      - 0.1
      - 0.2
      ffn_dropout:
      - 0.1
      - 0.2
      learning_rate:
      - 0.0001
      - 0.0003
      - 0.001
      weight_decay:
      - 1.0e-06
      - 1.0e-05
      - 0.0001
      batch_size:
      - 256
      - 512
      positive_class_weight:
      - null
      - outer-training negative_count / positive_count
    validity_rules:
    - d_block must be divisible by attention_n_heads
    - use one deterministic seed for model comparison and report additional-seed sensitivity for the selected configuration
      when feasible
    - save the selected epoch count and training curves for every outer fold
    - do not claim attention weights are causal explanations or reliable feature importance without a validated
      method
  xgboost_search:
    method: bounded randomized or staged candidate search using the shared five-fold registry
    maximum_configurations: 12
    fixed:
      objective: binary:logistic
      tree_method: hist
      random_state: 42
      n_jobs: recorded environment value
    search_space:
      n_estimators:
      - 300
      - 600
      - 1000
      max_depth:
      - 3
      - 5
      - 7
      learning_rate:
      - 0.03
      - 0.05
      - 0.1
      min_child_weight:
      - 1
      - 5
      - 10
      subsample:
      - 0.8
      - 1.0
      colsample_bytree:
      - 0.8
      - 1.0
      reg_lambda:
      - 1
      - 5
      - 10
      scale_pos_weight:
      - '1'
      - outer-training negative_count / positive_count
    restriction: If early stopping is used, the eval_set must be a training-only inner validation subset, never
      the outer validation fold or formal holdout.
  lightgbm_search:
    method: bounded randomized or staged candidate search using the shared five-fold registry
    maximum_configurations: 12
    fixed:
      objective: binary
      random_state: 42
      verbosity: -1
      subsample_freq: 1 # required for subsample (bagging_fraction) < 1.0 to take effect
    search_space:
      n_estimators:
      - 300
      - 600
      - 1000
      num_leaves:
      - 15
      - 31
      - 63
      max_depth:
      - -1
      - 6
      - 10
      learning_rate:
      - 0.03
      - 0.05
      - 0.1
      min_child_samples:
      - 20
      - 50
      - 100
      subsample:
      - 0.8
      - 1.0
      colsample_bytree:
      - 0.8
      - 1.0
      reg_lambda:
      - 0
      - 1
      - 5
      class_weight:
      - 'null'
      - balanced
    restriction: If early stopping is used, callbacks/evaluation data must come only from a training-only inner
      validation subset.
  tabnet_search:
    method: bounded staged candidate search; no full Cartesian grid
    maximum_configurations: 10
    implementation: pytorch_tabnet.tab_model.TabNetClassifier
    init_parameters:
      fixed:
        seed: 42
        cat_emb_dim: 1
      search_space:
        n_d:
        - 8
        - 16
        - 32
        n_a:
        - 8
        - 16
        - 32
        n_steps:
        - 3
        - 5
        - 7
        gamma:
        - 1.2
        - 1.5
        lambda_sparse:
        - 1.0e-05
        - 0.0001
        - 0.001
        optimizer_params:
        - lr: 0.01
        - lr: 0.02
    fit_parameters:
      fixed:
        max_epochs: 150
        patience: 15
        eval_metric: ['TabNetAveragePrecision'] # custom pytorch_tabnet.metrics.Metric subclass calculating Average Precision on inner validation
        early_stopping_metric: inner-validation Average Precision
      search_space:
        batch_size:
        - 1024
        - 2048
        virtual_batch_size:
        - 128
        - 256
        weights:
        - 0
        - 1
    validity_rules:
    - virtual_batch_size must divide batch_size
    - categorical vocabularies and numerical scaler are fitted on outer training only
    - save training history, selected epoch, cat_idxs/cat_dims, preprocessing state, seed and device
    - do not interpret attentive masks as causal evidence
fairness_protocol:
  gender_groups:
  - Female
  - Male
  - Other
  age_bands:
  - 22-29
  - 30-39
  - 40-49
  - 50-60
  minimum_reporting_denominators:
    total_n: 200
    positive_n: 30
    negative_n: 30
  metrics:
  - sample size
  - positive prevalence
  - precision
  - recall
  - false-positive rate
  - predicted-positive rate
  - selection rate at capacity
  - representation ratio at capacity
  - value-weighted selection share
  insufficient_sample_rule: When thresholds are not met, report counts/prevalence only and label rate estimates
    unstable or not reported.
  sensitivity_models:
  - full model
  - no-Gender model
  - no-Gender/no-Age model if required model work is complete
  interpretation: Subgroup similarity does not prove fairness or legal compliance. Disparities may arise from the
    risk model, the value proxy, or their combination and must be reported separately.
environment_contract:
  python_target: 3.11.x
  tested_development_environment: Python 3.14.0 (Windows x64) smoke-tested successfully; formal project target remains 3.11.x
  packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - catboost
  - interpret
  - torch
  - rtdl_revisiting_models
  - xgboost
  - lightgbm
  - pytorch-tabnet
  - jupyter
  - joblib
  - pytest
  - git
  dependency_specification: Maintain and smoke-test a compatible requirements.txt, environment.yml, or equivalent
    repository specification. Do not hard-code package versions in this context file; record exact tested versions
    in the execution environment artefact.
  compatibility_rule: Record the tested environment identifier with formal execution and verify imports, CPU/GPU
    device selection where relevant, fitting, probability output, explanation methods, checkpoint/model reload,
    and serialization for all six headline models.
  serialization_rule: 'Save family-appropriate artefacts: CatBoost native model plus metadata; XGBoost booster/model
    plus encoder metadata; LightGBM booster/model plus category metadata; verified EBM artefact; FT-Transformer
    PyTorch state_dict plus architecture/vocabularies/scaler/epoch/device-independent reload test; TabNet save_model
    artefact plus preprocessing/category metadata and reload test; Logistic Regression sklearn pipelines for the required PCA benchmark, plus any optional broader standalone Logistic run when executed.'
deliverables:
- name: Proposal presentation
  format: PowerPoint
  duration: 15 minutes per group
  status: VERIFIED requirement
- name: End-to-end Python AI/ML implementation
  format: repository/notebooks/scripts
  status: VERIFIED requirement
- name: Final group report
  format: PDF
  page_limit: 15
  status: VERIFIED requirement
- name: Viva preparation and participation
  format: oral review
  status: VERIFIED requirement
- name: Employee_Resignation_Prediction_AI_Project_Context_v4_6_2.md
  format: Markdown
  status: canonical project context
fixed_decisions:
- Primary target remains Resigned with True as the positive class.
- 'The project has two linked stages: resignation-risk modelling and constrained retention-prioritization simulation.'
- The six required advanced models are FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, and TabNet.
- 'Each member owns one headline model: Member 1 FT-Transformer; Member 2 CatBoost; Member 3 XGBoost; Member 4 LightGBM;
  Member 5 EBM; Member 6 TabNet.'
- All six models share the same source rows, target mapping, 80/20 split, five outer folds, evaluation metrics,
  and holdout restrictions while using model-specific fold-local preprocessing.
- DummyClassifier prior and most_frequent strategies are mandatory shared no-skill references.
- Logistic Regression is required for the PCA demonstration benchmark; broader standalone Logistic Regression remains optional. It is not a required advanced headline model and must not displace work on the six required headline models.
- FT-Transformer uses fold-local continuous normalization, categorical vocabularies with missing/unknown codes,
  embeddings, PyTorch BCE-with-logits training, and training-only inner validation for early stopping.
- Primary capacity is the top 20 percent of records under an equal-cost hypothetical constraint; 10 and 30 percent
  are sensitivity scenarios.
- The retention-value measure is a proxy, not actual employee worth or financial value.
- Primary proxy is the equal-weight mean of development-fitted normalized Performance_Score, Monthly_Salary, and
  Years_At_Company; no-salary and normalized multiplicative proxies are sensitivity analyses.
- Compare repeated random, risk-only, value-only, combined risk-times-value, and reference threshold strategies.
- Assessed build is the end-to-end Python workflow and evaluation package, not a website, dashboard, API, or deployed
  HR product.
- Academic record-level ranking is allowed only inside the controlled assigned-dataset simulation; real-world employment
  use is prohibited.
- Employee_ID and raw Hire_Date are excluded from primary modelling; Years_At_Company is retained.
- Class imbalance is handled primarily through model-appropriate class weighting/loss weighting and threshold analysis;
  no pre-split resampling.
- Five-fold development CV is primary comparison evidence; the formal holdout is corroborative and carries the historical
  audit-exposure caveat.
- Final headline-model comparison uses strict nested outer-CV/inner-selection: all tuning, weighting, preprocessing choices, and early stopping remain inside each outer-training partition; each outer-validation fold is prediction-only after fold-specific selection.
- Calibration must be evaluated because probabilities are used in the priority score.
- A valid negative result is acceptable and must be reported honestly.
- ChatGPT usage is disclosed and maintained as a live record.
- This Markdown file is the sole persistent canonical project-context artifact; no parallel mirrored context file is maintained.
unresolved_placeholders:
- GROUP-NUMBER
- LAB-BATCH-OR-SESSION
- TEAM-MEMBER-NAMES-AND-STUDENT-IDS
- MODULE-COORDINATOR-OR-LECTURER
prohibited_assumptions:
- Do not claim the dataset contains verified real employees.
- Do not infer salary currency or units.
- Do not fabricate model results or team identities.
- Do not treat accuracy as the primary metric.
- Do not tune against the formal holdout after it is locked.
- Do not claim the holdout was historically pristine.
- Do not make causal, deployment, fairness-certification, or legal-compliance claims.
- Do not substitute another similarly named dataset.
- Do not call the retention-value proxy actual employee worth, contribution, replacement cost, or revenue.
- Do not treat the 20 percent capacity as a verified real budget.
- Do not claim that a selected intervention would prevent resignation.
- Do not claim the combined ranking is optimal without real cost, benefit, and intervention-effect data.
- Do not tune capacity, proxy formula, proxy weights, or ranking strategy against the formal holdout.
- Do not one-hot encode categorical variables for CatBoost merely to reuse the Logistic Regression matrix.
- Do not assume CatBoost is the winner because it is more complex.
- Do not call EBM a simple linear model; it is the required advanced interpretable nonlinear model.
- Do not assume FT-Transformer is the winner because it is deep learning or uses attention.
- Do not interpret raw attention weights as causal explanations or validated feature importance.
- Do not let any member create a private train/test split or private outer CV folds for headline comparison.
- Do not force all six required models to reuse one transformed feature matrix when their valid input representations
  differ.
- Do not assume the three boosting libraries or the two deep-learning models will produce materially different results
  simply because their architectures differ.
- Do not interpret TabNet attentive masks as causal explanations.
startup_acknowledgement:
  required_before_project_dependent_work: true
  fields:
  - project objective and two-stage scope
  - current phase
  - target and positive class
  - active six-model registry, individual ownership, model-specific representations, and primary prediction metric
  - six-member model ownership map
  - capacity, proxy definition, comparison strategies, and ranking metrics
  - formal holdout restriction and historical exposure caveat
  - principal provenance/data-quality limitation
  - absence of intervention-effect and true employee-value data
  - prohibited real-world operational use
  - relevant unresolved placeholders
  - files/evidence currently available
  - next action
  reconfirmation_rule: Do not ask the user to reconfirm a DECIDED item unless higher-authority evidence conflicts
    with it.
update_rules:
  live_record_required_fields:
  - status
  - last_updated
  - updated_by
  - evidence
  - owner when known
  material_updates:
  - new official instruction
  - new file or checksum
  - repository/split creation
  - experiment result
  - model freeze or holdout use
  - deliverable completion
  - AI tool use
  - decision change
  - new blocker
  never_overwrite_history: Mark previous active values SUPERSEDED and retain them in the change log.
```

# 2. AI quick reference

## 2.1 One-minute project summary

**What is being created:** An academically defensible two-stage study. Stage 1 evaluates whether the assigned employee variables contain out-of-sample signal for `Resigned=True`. Stage 2 uses the frozen probability output to compare limited-capacity retention-prioritization strategies.

**Six required advanced models and ownership:**

- **Member 1 — FT-Transformer** (`rtdl_revisiting_models.FTTransformer`): attention-based deep tabular learning with numerical and categorical feature tokens.
- **Member 2 — CatBoost Classifier:** gradient boosting with native categorical handling.
- **Member 3 — XGBoost Classifier** (`xgboost.XGBClassifier`): histogram-based boosted trees using a fold-local encoded matrix.
- **Member 4 — LightGBM Classifier** (`lightgbm.LGBMClassifier`): efficient leaf-wise gradient boosting with fold-local categorical handling.
- **Member 5 — Explainable Boosting Machine** (`ExplainableBoostingClassifier`): nonlinear, inherently interpretable additive boosting.
- **Member 6 — TabNet Classifier** (`pytorch_tabnet.tab_model.TabNetClassifier`): attentive deep learning for tabular data.

**References:** `DummyClassifier(strategy="prior")` and `DummyClassifier(strategy="most_frequent")` are mandatory no-skill baselines. Logistic Regression is required for the PCA demonstration benchmark and otherwise optional; it does not count toward the six headline models.

**Shared experimental foundation:** Every member uses the exact same saved 80/20 stratified development/formal-holdout split and the same five outer stratified CV folds. No member may create a private comparison split. Model-specific preprocessing is allowed and required where appropriate, but it must be fitted only inside the relevant training fold.

**Fixed evaluation:** PR-AUC/Average Precision is the primary prediction metric. Secondary metrics include ROC-AUC, precision, recall, F1, balanced accuracy, confusion matrix, calibration/Brier score, fold stability, runtime, reproducibility, and valid interpretability evidence. The formal holdout is used only once after the complete design is frozen.

**Prioritization:** Primary capacity is 20%; 10% and 30% are fixed sensitivities. Compare repeated random selection, risk-only, value-proxy-only, risk-times-value-proxy, and threshold-reference strategies. The retention-value score is a transparent proxy, not actual employee worth.

**Critical limitation:** Preliminary audit-only evidence suggests the target may contain little predictive signal. Six complex models cannot create absent information. A rigorous no-signal result remains valid and must prevent operational claims.

**Current phase:** Harmonize the implemented model adapters/configs with the approved strict nested evaluation protocol, complete protocol-compliant headline-model evidence, then proceed to model comparison, calibration, fairness, and prioritization. PCA results are generated; FT-Transformer, XGBoost, and TabNet have current development results; CatBoost formal strict-nested development evaluation is complete; LightGBM remains pending formal evaluation; EBM results exist upstream and require local integration/verification.

## 2.2 Critical terminology

| Term | Project-specific meaning |
| --- | --- |
| Positive class | `Resigned=True`. |
| Required advanced model | One of the six headline models: FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, or TabNet. Each is individually owned by one member. |
| Shared baseline | Mandatory DummyClassifier reference evaluated under the same row/fold protocol but not presented as an advanced solution. |
| Logistic Regression benchmark | Required for the PCA demonstration; broader standalone Logistic Regression is optional. |
| Outer fold | One of five saved StratifiedKFold development-CV folds. Under the strict nested protocol, its validation rows are prediction-only after fold-specific inner selection, providing selection-independent fold evidence for model comparison. |
| Inner validation | A stratified subset created only from an outer training fold for early stopping, epoch selection, or iterative-model monitoring. |
| PR-AUC / Average Precision | Primary prediction-ranking metric for the imbalanced positive class. |
| Formal holdout | The 20% stratified partition locked against design, tuning, early stopping, calibration, and selection; used once after freeze. |
| Resignation-risk probability | A model output whose calibration must be assessed before use in the priority score. It is not certainty. |
| Capacity | Fraction of records selected in the academic simulation. Primary scenario: 20%. |
| Retention-value proxy | Transparent score constructed from available variables for academic comparison; not actual worth, contribution, or replacement cost. |
| Priority score | Calibrated resignation-risk probability multiplied by the selected retention-value proxy. |
| Precision@20% | Share of selected top-20% records that are actually positive. |
| Recall@20% | Share of all actual positive records captured in the selected top 20%. |
| Lift@20% | Precision@20% divided by overall positive prevalence. |
| Value-weighted recall@20% | Share of total positive-class proxy value contained in the selected top 20%; it does not imply value was protected. |
| Predictive signal | Out-of-sample association materially exceeding a suitable baseline; not causality. |
| Fairness audit | Subgroup prediction and selection analysis; not a declaration of legal fairness or compliance. |

## 2.3 Items an AI must never assume

- that the employees are verified real individuals;
- that `Monthly_Salary` uses any particular currency;
- that high accuracy implies useful resignation prediction;
- that complexity, deep learning, or attention guarantees superior performance;
- that class weighting or resampling creates missing target signal;
- that correlation, EBM effects, CatBoost importance, SHAP values, or attention weights identify why employees resign;
- that the retention-value proxy measures actual worth, contribution, replacement cost, or revenue;
- that the 20% scenario is an actual organization's budget;
- that selecting or supporting a record would prevent resignation;
- that the risk-times-value ranking is financially optimal;
- that a fairness check makes the strategy automatically fair or lawful;
- that a GPU is available until the runtime environment confirms it;
- that every member must speak unless officially confirmed;
- that a model should be deployed or used for real individual HR action.

# 3. Current operating state

## 3.1 Current milestone

**LIVE RECORD - 2026-09-12:** The canonical 100,000-row dataset, frozen 80/20 split, five fixed 16,000-row development folds, EDA, Dummy baselines, Logistic/PCA benchmark artifacts, shared metric/OOF utilities, and all model adapters are present. The current test suite contains 186 tests across 14 modules; the latest verified run is **186 passed (100% passing)** after hardening the PCA notebook integrity assertions.

Current model evidence is mixed by maturity:

- FT-Transformer: current five-fold development result artifacts exist; protocol-complete strict nested reevaluation is still required before calling the pooled score the final unbiased headline estimate.
- XGBoost: current five-fold fixed-configuration development result artifacts exist; methodology-complete tuning/weighting/early-stopping alignment remains pending.
- TabNet: eight declared candidates were evaluated and T01 selected; strict nested final evaluation remains required for an unbiased selected-model headline estimate.
- CatBoost: formal strict-nested 5-fold development evaluation complete; 80,000 development OOF predictions generated; performance is essentially at chance / no-signal baseline (pooled AP 0.099350).
- LightGBM: adapter/config/scaffold exist; formal nested evaluation pending.
- EBM: adapter/config exist; executed results are available upstream on `origin/main` and require local integration/verification before canonical completion.
- Notebooks 11-14 remain scaffolds. Notebook 15 remains locked.

The immediate milestone is to align model configs and implementations to the approved nested outer-CV / inner-selection protocol, complete remaining/revised headline evaluations, then execute Notebook 11 model comparison.

Canonical repository: `https://github.com/Sankalpa-KMCP/Employee_Resignation_Prediction`.

## 3.2 Immediate objectives

1. Maintain this canonical Markdown context as v4.6.2 and keep secondary summaries such as README.md and EVALUATION_GUIDE.md aligned with verified live state.
2. Finalize bounded model configurations for XGBoost, LightGBM, EBM, and FT-Transformer; remove unresolved TODO search-space markers only after the final nested protocol is represented exactly (CatBoost v2 schema and C01-C08 registry finalized).
3. Implement or orchestrate inner validation/inner tuning for XGBoost and LightGBM without allowing any outer-validation fold to control training or selection.
4. Re-evaluate FT-Transformer, XGBoost, and TabNet under strict nested selection where required so final OOF estimates are independent of model-selection decisions.
5. Integrate and verify the upstream EBM work, then determine whether rerun is required under the final protocol.
6. Complete LightGBM nested five-fold evaluation (CatBoost formal development evaluation complete).
7. Only after all six headline models have protocol-complete OOF evidence, execute model comparison, calibration, fairness, and retention-prioritization workflows (`11` through `14`).
8. Keep Notebook 15 locked until model, preprocessing, calibration, threshold, proxy, capacity, and strategy choices are frozen.

## 3.3 Completed work

- **VERIFIED** - Formal assignment requirements, dataset identity, schema, target distribution, and audit limitations are recorded.
- **COMPLETED** - Canonical GitHub repository URL is recorded.
- **COMPLETED** - `01_data_audit.ipynb`, `02_create_split_registry.ipynb`, `03_eda.ipynb`, and `04_baselines.ipynb` are executed with saved outputs.
- **DECIDED 2026-08-01** - Project includes constrained retention-prioritization simulation after resignation-risk modelling.
- **SUPERSEDED 2026-08-06** - Three-model / three-pair registry (CatBoost, EBM, FT-Transformer).
- **DECIDED 2026-08-07** - Six headline models are required with one model per member: FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, TabNet.
- **COMPLETED 2026-08-07** - Context v4.3.0 synchronized the six-model methodology, individual ownership, preprocessing, tuning, evaluation, artefacts, report, and viva rules.
- **RESULTS GENERATED / FINAL VERIFICATION PENDING** - `04a_pca_analysis.ipynb` has 23 of 24 code cells executed and all four canonical PCA CSV artifacts saved; the final verification cell remains unexecuted.
- **DEVELOPMENT EVIDENCE AVAILABLE** - FT-Transformer, XGBoost, and TabNet have saved development result artifacts; these are not automatically the final unbiased nested-CV headline estimates.
- **UPSTREAM DEVELOPMENT EVIDENCE AVAILABLE** - EBM results exist on `origin/main` and require local integration/verification.
- **VERIFIED 2026-09-11** - The automated suite contains 150 tests across 13 modules; latest run: 149 passed, 1 failed due to the obsolete PCA scaffold-integrity expectation.
- **DECIDED 2026-09-11** - Final headline-model evaluation uses strict nested outer-CV / inner-selection.

## 3.4 Work that may proceed now

- synchronize and commit the verified documentation updates when ready;
- finalize bounded search spaces and implementation alignment for the strict nested protocol;
- implement/orchestrate inner tuning and early stopping for XGBoost and LightGBM without using outer-validation rows;
- complete LightGBM formal nested evaluation (CatBoost development evaluation is complete);
- integrate and verify the upstream EBM work;
- rerun FT-Transformer, XGBoost, and TabNet where needed to obtain final nested OOF estimates;
- prepare Notebook 11 model-comparison inputs only after all six headline models have protocol-complete OOF evidence.

## 3.5 Blocked or conditional work

- final six-model comparison is blocked until all six headline models have protocol-complete nested OOF evidence on the common development folds;
- calibration, fairness, and prioritization are blocked from final execution until model-comparison inputs are complete and the probability source is selected using development evidence;
- Notebook 15 formal-holdout execution is blocked until model, preprocessing, calibration, threshold, proxy, capacity, and strategy choices are frozen;
- genuine monetary budget optimization is blocked because intervention costs, effects, replacement costs, and verified business-value data are absent;
- causal claims that a retention action prevents resignation are blocked;
- production use, employment decisions, legal-compliance claims, and fairness certification are outside scope;
- GPU acceleration is conditional on the tested environment; CPU execution remains valid where feasible;
- final numerical conclusions remain conditional on executing the approved nested workflow and must not be invented.

## 3.6 Work not to prioritize now

- dashboard, GUI, website, API, database, cloud deployment, or mobile application;
- model stacking, voting, foundation models, or seventh headline model families;
- unrestricted hyperparameter search;
- synthetic provenance narratives or assumed salary currency;
- automated record-level employment actions.

## 3.7 Current risks

| ID | Risk | Likelihood | Impact | Active mitigation |
| --- | --- | --- | --- | --- |
| R-001 | Target may contain little usable signal. | High | High | Dummy baselines, PR-AUC, uncertainty, no-signal rule. |
| R-002 | Historical preliminary audit exposed aggregate holdout-related diagnostics. | Certain | Medium | Freeze protocol; preserve exposure caveat; use holdout only as corroboration. |
| R-003 | Salary determinism may dominate proxy or model behaviour. | High | High | No-salary sensitivity and explicit redundancy analysis. |
| R-004 | Six-model scope may create duplicated or inconsistent experiments. | Medium | High | One split/fold registry, one evaluator contract, one result schema. |
| R-005 | Model-specific adapters may create row/fold parity errors. | Medium | High | Shared row-ID assertions and OOF-coverage tests. |
| R-006 | Deep models may leak validation information through vocabularies, scaling, or early stopping. | Medium | High | Fit preprocessing on outer training only; inner validation stays inside outer training. |
| R-007 | FT-Transformer and TabNet may vary by seed/device. | Medium | Medium | Seed/device logs and additional-seed sensitivity for selected configurations when feasible. |
| R-008 | Probabilities may be poorly calibrated, especially under class weighting. | Medium | High | OOF reliability curves, Brier score, development-only calibration. |
| R-009 | Six members may accidentally use different metrics or threshold rules. | Medium | High | Shared evaluator module and frozen metric definitions. |
| R-010 | AI-generated code or explanations may be accepted without ownership. | Medium | High | Tests, source verification, manual review, AI-use records, viva preparation. |

## 3.8 Next concrete actions

1. Finalize XGBoost, LightGBM, EBM, and FT-Transformer candidate spaces so configs match the approved bounded nested-selection protocol.
2. Add or orchestrate training-only inner validation/tuning for remaining tree models (XGBoost and LightGBM) without changing the outer folds.
3. Complete LightGBM nested evaluation (CatBoost development evaluation is complete).
4. Integrate and verify the EBM work from `origin/main`; rerun under the final protocol if its current result is not nested-selection independent.
5. Re-evaluate FT-Transformer, XGBoost, and TabNet under the strict nested protocol where required.
6. Validate complete six-model OOF row parity and then execute Notebook 11.
7. Execute calibration, fairness, and prioritization only after Notebook 11 selects the development probability source.
8. Keep Notebook 15 locked until the complete development pipeline is frozen.

# 4. Task router for AI assistants

| Future request | Must read | Must verify | May generate | Must not invent |
| --- | --- | --- | --- | --- |
| Understand the project | Sections 0-3 and 8.1-8.10 | v4.6.2, six-model registry, individual ownership, current status | concise status-aware summary | new objectives or results |
| Inspect data | Sections 5, 8.10-8.15, 8.18-8.19 | exact CSV checksum, shape, schema | audit code/tables/EDA plan | units or provenance |
| Create shared split/folds | Sections 8.16-8.19 | seed 42, stratification, row IDs, holdout lock | split/fold registry and tests | alternate private member splits |
| Generate FT-Transformer code | Sections 8.18-8.23 | fold-local scaler/vocabularies, inner validation, device | M1 pipeline/training/checkpoint code | holdout early stopping or attention-as-causality |
| Generate CatBoost code | Sections 8.18-8.23 | native categoricals, common folds | M2 pipeline/tuning/OOF predictions | full-data categorical statistics |
| Generate XGBoost code | Sections 8.18-8.23 | fold-local encoder, common folds | M3 pipeline/tuning/OOF predictions | full-data one-hot matrix |
| Generate LightGBM code | Sections 8.18-8.23 | fold-local category mapping, common folds | M4 pipeline/tuning/OOF predictions | category mapping learned from holdout |
| Generate EBM code | Sections 8.18-8.23 | explicit feature types/interactions | M5 pipeline/tuning/explanations | causal interpretation |
| Generate TabNet code | Sections 8.18-8.23 | fold-local codes/scaler, inner validation | M6 pipeline/tuning/history | holdout early stopping or mask-as-causality |
| Generate baseline code | Sections 8.20-8.22 | shared rows/folds | Dummy baselines; required PCA Logistic benchmark; optional broader Logistic if desired | counting baseline/benchmark as one of six |
| Tune/evaluate models | Sections 8.21-8.25 and 8.33 | frozen search caps, OOF coverage, identical folds | metrics/calibration/uncertainty tables | holdout-guided tuning |
| Generate prioritization code | Sections 8.16-8.24 | frozen capacity/proxy/probability source | random/top-k comparisons | actual worth or intervention benefit |
| Write report/presentation | Sections 8.6-8.9 and generated evidence | actual results and six-member ownership | evidence-backed prose/figures | fabricated findings |
| Prepare viva | Sections 8.2, 8.13, 8.16-8.25, 8.30, 8.33 | each member's own model plus whole workflow | member-specific and shared Q&A | ungenerated results |
| Update status/decision | Sections 3, 7, 8.35-8.38 | evidence and authority | live-state/change records | silent methodology changes |

# 5. Canonical-file manifest and handoff package

## 5.1 Required files for project-dependent work

| ID | Canonical filename or path | Required for | Integrity rule | Status |
| --- | --- | --- | --- | --- |
| CTX-MD | `Employee_Resignation_Prediction_AI_Project_Context_v4_6_2.md` | All project-dependent work | Canonical; version/date must match. | VERIFIED generated |
| SRC-001 | `Group Assignment Specification.pdf` or Section 9.2 extract | Requirements and marks | Original PDF remains authoritative; checksum recorded. | VERIFIED |
| SRC-002 | `Proposal Presentation.pdf` or Section 9.3 extract | Proposal requirements | Original PDF remains authoritative; checksum recorded. | VERIFIED |
| SRC-003 | `archive.zip` | Original assigned archive | Verify recorded SHA-256. | VERIFIED |
| SRC-004 | `data/raw/Extended_Employee_Performance_and_Productivity_Data.csv` | Data analysis and modelling | Verify SHA-256 `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb`. | VERIFIED source |
| SRC-005 | Kaggle dataset page | Source link and licence metadata | Do not treat listing description as original provenance evidence. | VERIFIED listing |
| CODE | `https://github.com/Sankalpa-KMCP/Employee_Resignation_Prediction` | Implementation | Inspect current branch/files before code-dependent claims. | DECIDED / LIVE SOURCE |

## 5.2 Optional files

- newer lecturer/LMS clarifications with date and source;
- exact tested environment export and hardware/device record;
- slide deck, report source, and final exports;
- model/experiment cards, peer-review notes, and AI prompt/output evidence.

## 5.3 Expected repository structure

```text
Employee_Resignation_Prediction/
├── README.md
├── EVALUATION_GUIDE.md
├── requirements.txt
├── AGENTS.md
├── .gitignore
├── data/
│   ├── raw/
│   │   └── Extended_Employee_Performance_and_Productivity_Data.csv
│   └── processed/
│       └── split_registry.csv
├── configs/
│   ├── split_config.yaml
│   ├── feature_contract.yaml
│   └── models/
│       ├── catboost.yaml                 # schema v2.0.0; frozen C01-C08 candidate registry
│       ├── ebm.yaml                      # preliminary; TODO remains
│       ├── ft_transformer.yaml           # preliminary; TODO remains
│       ├── lightgbm.yaml                 # preliminary; TODO remains
│       ├── tabnet.yaml                   # discrete T01-T08 candidate protocol
│       └── xgboost.yaml                  # preliminary; TODO remains
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_create_split_registry.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_baselines.ipynb
│   ├── 04a_pca_analysis.ipynb
│   ├── 05_member_1/05_m1_ft_transformer.ipynb
│   ├── 06_member_2/06_m2_catboost.ipynb
│   ├── 07_member_3/07_m3_xgboost.ipynb
│   ├── 08_member_4/08_m4_lightgbm.ipynb
│   ├── 09_member_5/09_m5_ebm.ipynb
│   ├── 10_member_6/10_m6_tabnet.ipynb
│   ├── 11_model_comparison.ipynb
│   ├── 12_calibration.ipynb
│   ├── 13_fairness_analysis.ipynb
│   ├── 14_retention_prioritization.ipynb
│   └── 15_final_holdout_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py                  # placeholder
│   │   ├── feature_contract.py          # placeholder
│   │   ├── splits.py                    # placeholder
│   │   └── validation.py                # placeholder
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── runner.py
│   │   ├── logistic_regression.py
│   │   ├── ft_transformer.py
│   │   ├── catboost_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── ebm_model.py
│   │   └── tabnet_model.py
│   ├── evaluation/
│   │   ├── __init__.py                  # placeholder
│   │   ├── metrics.py                   # implemented
│   │   ├── oof_validation.py            # implemented
│   │   ├── calibration.py               # placeholder
│   │   ├── fairness.py                  # placeholder
│   │   └── plots.py                     # placeholder
│   └── prioritization/
│       ├── __init__.py                  # placeholder
│       ├── ranking.py                    # placeholder
│       └── value_proxy.py                # placeholder
├── tests/                                # 14 modules; 186 collected tests as of 2026-09-12 (186/186 passing)
├── models/                               # reserved model artifacts; no final frozen model family set yet
├── results/
│   ├── audit/
│   ├── baselines/
│   ├── eda/
│   ├── pca/
│   ├── models/
│   │   ├── ft_transformer/              # populated current development evidence
│   │   ├── catboost/                    # populated formal development nested results
│   │   ├── xgboost/                     # populated current development evidence
│   │   ├── lightgbm/                    # formal results pending
│   │   ├── ebm/                         # local development results pending; upstream development results exist
│   │   └── tabnet/                      # populated T01-T08 development evidence
│   ├── comparison/
│   ├── calibration/
│   ├── fairness/
│   ├── prioritization/
│   └── final_holdout/
├── reports/
├── presentations/
└── docs/
    └── context/
        └── Employee_Resignation_Prediction_AI_Project_Context_v4_6_2.md
```

The tree deliberately distinguishes implemented modules, placeholders, development evidence, and final frozen artefacts. File existence alone is not evidence that the corresponding workflow has been executed.

## 5.4 Missing or mismatched file protocol

1. Stop data-dependent work if the exact CSV is absent or its checksum differs.
2. Never substitute another copy automatically.
3. Compare filename, archive contents, shape, schema, and checksum.
4. Record a new source ID and decision if the assigned artefact officially changes.
5. Treat the handoff as incomplete if this canonical Markdown file is missing.
6. Treat an untested FT-Transformer package/API as an environment issue, not permission to silently change models.

## 5.5 Canonical-file maintenance rule

- This Markdown file is the only persistent canonical project-context file.
- Make every material context change here first, then increment the version and update the decision/change records.
- Keep `README.md`, `EVALUATION_GUIDE.md`, notebooks, configs, and other secondary artefacts aligned with this file where their scopes overlap.
- Do not maintain a mirrored second-format context file.
- After material edits, validate Markdown structure, embedded YAML, version/source references, active decisions, and stale terminology before adoption.

## 5.6 Handoff package levels

**Conceptual handoff:** This canonical Markdown file is sufficient for ordinary project understanding and planning.

**Data-execution handoff:** Add the exact raw CSV or original archive and verify its checksum.

**Implementation handoff:** Inspect the canonical GitHub repository, exact tested environment, and generated artefacts required by the task.

**Formal-source verification:** Use the original PDFs when exact wording, layout, or a newer revision matters. Section 9 contains normalized project-relevant extracts, not complete verbatim transcripts.

# 6. Dependency map

## 6.1 Technical and procedural dependencies

```text
Formal requirements
    -> approved two-stage framing
        -> exact dataset and checksum
            -> shared feature contract
                -> saved 80/20 development/formal-holdout split
                    -> saved five outer folds
                        -> six model-specific fold-local adapters
                            -> Dummy references + six individually owned headline models
                                -> complete OOF probabilities on identical rows/folds
                                    -> common prediction/calibration/stability comparison
                                        -> model/preprocessing/calibration freeze
                                            -> proxy/capacity/strategy freeze
                                                -> one-time formal-holdout corroboration
                                                    -> prioritization, subgroup, error, and robustness analysis
                                                        -> report, presentation, and viva

Deep models (FT-Transformer and TabNet)
    -> tested PyTorch-related environment
        -> device/seed controls
            -> fold-local vocabularies/scalers
                -> training-only inner validation
                    -> checkpoint/model reload tests

Actual intervention cost/effect evidence
    -> genuine budget optimization
        -> NOT AVAILABLE; blocked and outside scope
```

## 6.2 Blocking table

| Dependency | Blocks | Current status | Continue? |
| --- | --- | --- | --- |
| Exact assigned CSV/checksum | data-dependent work | VERIFIED available | Yes |
| Approved six-model registry/ownership | model work division | DECIDED | Yes |
| Shared split/fold registry | fair headline training/comparison | VERIFIED available (`split_registry.csv`) | Yes |
| Tested CatBoost/XGBoost/LightGBM/InterpretML/PyTorch/RTDL/TabNet environment | model execution | VERIFIED installed & tested | Yes |
| Named team identities | formal accountability | PLACEHOLDER | Numbered member roles are sufficient for now |
| Real intervention/business-value evidence | genuine optimization/causal claims | NOT AVAILABLE | No; retain simulation-only framing |

# 7. Update protocol and handoff rules

## 7.1 Required update sequence

For every material change:

1. Increment semantic version: patch for corrections; minor for methodology/schema/team-workflow changes; major for target/dataset/scope changes.
2. Update date, context release ID, filename, and document-control table.
3. Update machine-readable memory.
4. Update Section 3 live state.
5. Update model registry, individual ownership map, decision records, risks, and placeholders as applicable.
6. Update methodology, result records/templates, references, and AI-use records where affected.
7. Add a change-log entry with evidence and approver/authority.
8. Validate all embedded YAML and Markdown structure.
9. Run stale-reference, version, filename, source-ID, and terminology consistency checks across the complete file.
10. Check overlapping secondary artefacts such as `README.md` and `EVALUATION_GUIDE.md` for material drift before declaring context alignment complete.

## 7.2 Result update rule

A result becomes **VERIFIED** only when its data checksum, split/fold ID, code commit, environment, model configuration, seed/device, and output artefact are identifiable. Unsaved notebook cells, slides, conversations, and preliminary audit diagnostics are insufficient.

## 7.3 Decision update rule

Record decision ID, date, previous value, new value, rationale, authority, affected sections, approver, and reconsideration condition. Never silently substitute a model or expose the formal holdout.

## 7.4 Reusable handoff instruction

> Read `Employee_Resignation_Prediction_AI_Project_Context_v4_6_2.md` first and treat it as the sole persistent canonical project-context file. The six required advanced models are individually owned: M1 FT-Transformer, M2 CatBoost, M3 XGBoost, M4 LightGBM, M5 EBM, M6 TabNet. DummyClassifier baselines are mandatory; Logistic Regression is required for the PCA demonstration benchmark and otherwise optional. All headline models must use identical saved development/holdout rows and outer folds but model-specific fold-local representations. Final headline-model evaluation uses strict nested outer-CV/inner-selection; all tuning, weighting, preprocessing choices, and early stopping stay inside the corresponding outer-training partition. Lock the formal holdout against all preprocessing design, tuning, early stopping, calibration choice, threshold choice, proxy/capacity/strategy choice, and model selection. Verify the exact CSV checksum before data work, inspect current repository files before code-dependent claims, never invent results, and update live state, AI-use records, decisions, and version metadata after material changes.

# 8. Full canonical project specification

## 8.1 Document control

| Field | Value | Status |
| --- | --- | --- |
| Context version | 4.6.2 | ACTIVE |
| Date | 2026-09-12 | VERIFIED |
| Canonical file | `Employee_Resignation_Prediction_AI_Project_Context_v4_6_2.md` | DECIDED |
| Canonical repository | `https://github.com/Sankalpa-KMCP/Employee_Resignation_Prediction` | DECIDED |
| Current phase | Nested-protocol harmonization and remaining headline-model completion | LIVE RECORD |
| Required advanced models | FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, TabNet | DECIDED |
| Mandatory baselines | DummyClassifier `prior` and `most_frequent` | DECIDED |
| Logistic Regression benchmark | Required for PCA demonstration; otherwise optional | DECIDED |
| Individual ownership | M1 FT-Transformer; M2 CatBoost; M3 XGBoost; M4 LightGBM; M5 EBM; M6 TabNet | DECIDED |
| Superseded core models | Random Forest, Decision Tree, RBF-SVM | SUPERSEDED |
| Primary metric | Average Precision / PR-AUC | DECIDED |
| Primary capacity | Top 20% | DECIDED |

## 8.2 Executive project summary

The group is designing, implementing, and critically evaluating a two-stage AI/ML study using the assigned employee dataset.

1. **Prediction stage:** determine whether available employee-related features provide reproducible out-of-sample signal for `Resigned=True`.
2. **Prioritization stage:** use frozen, calibrated probability outputs in a controlled top-k simulation to compare how limited HR review/retention capacity might be allocated.

The six required advanced models are:

1. **FT-Transformer** - attention-based deep tabular learning with continuous and categorical feature tokens;
2. **CatBoost Classifier** - gradient-boosted trees and native categorical handling;
3. **XGBoost Classifier** - histogram-based gradient boosting on a fold-local encoded feature matrix;
4. **LightGBM Classifier** - efficient leaf-wise gradient boosting with fold-local categorical handling;
5. **Explainable Boosting Machine** - nonlinear additive boosting with inherent global/local interpretability;
6. **TabNet Classifier** - attentive deep learning for tabular data.

DummyClassifier provides mandatory no-skill references. Logistic Regression is required for the PCA demonstration benchmark and otherwise optional; it is not a headline advanced model.

The central evaluation is multidimensional: predictive ranking, threshold performance, calibration, stability, runtime, reproducibility, interpretation quality, subgroup behaviour, and prioritization lift. No model is assumed to win because of reputation or complexity.

All record-level prioritization is an academic simulation. The project does not establish causality, intervention effectiveness, actual employee value, financial optimality, legal fairness, or production readiness.

### 8.2.1 Exact assessed build

The assessed build is an end-to-end, reproducible Python workflow containing:

1. immutable raw-data identity and checksum checks;
2. data audit, cleaning decisions, and EDA;
3. saved 80/20 stratified development/formal-holdout split;
4. saved five outer development-CV folds;
5. model-specific leakage-safe adapters;
6. mandatory DummyClassifier references and the required PCA Logistic benchmark, with broader standalone Logistic use optional;
7. bounded tuning for FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, and TabNet;
8. training-only inner validation for iterative-model early stopping;
9. OOF probabilities, common metrics, calibration, and uncertainty;
10. one-time formal-holdout corroboration after complete freeze;
11. retention-value proxy and top-k strategy simulation;
12. explainability, error analysis, subgroup analysis, ethics, and limitations;
13. reproducible artefacts, report evidence, presentation content, and viva preparation.

A GUI, website, API, database, cloud deployment, or real HR decision system is not required.

## 8.3 Formal problem statement

Can the assigned employee dataset support reliable prediction of `Resigned=True`, and can any validated probability signal improve a transparent limited-capacity retention-prioritization simulation over simple random, risk-only, and value-only alternatives?

The project must answer this without claiming that observed associations are causal or that a selected employee can actually be retained.

## 8.4 Objectives and project questions

### 8.4.1 Objectives

- understand and justify the dataset;
- clean, validate, and visualize it reproducibly;
- implement six advanced supervised classifiers spanning boosted-tree, interpretable additive, and deep-tabular approaches;
- use identical saved rows/folds and model-appropriate preprocessing;
- tune within bounded development-only protocols;
- compare prediction, calibration, stability, interpretability, and computational cost;
- simulate top-k prioritization using a clearly labelled retention-value proxy;
- analyse subgroup behaviour and ethical risks;
- produce an honest positive or negative conclusion supported by saved evidence.

### 8.4.2 Project questions

- PQ-01: Does any model consistently exceed the Dummy prior baseline in development PR-AUC?
- PQ-02: How do all six headline models compare under identical outer folds?
- PQ-03: Do the two deep models (FT-Transformer and TabNet) justify their added training complexity relative to boosted-tree and EBM alternatives?
- PQ-04: How do CatBoost, XGBoost, and LightGBM differ in performance, stability, calibration, runtime, and explanation burden?
- PQ-05: Are probability estimates sufficiently calibrated for use in the combined priority score?
- PQ-06: Which model is most reproducible and stable across folds and, where feasible, seeds?
- PQ-07: Which features/terms are associated with predictions, and what explanations are valid for each model family?
- PQ-08: At 20% capacity, do risk-based strategies exceed repeated random selection?
- PQ-09: Does the combined risk-times-value-proxy ranking improve value-weighted capture without hiding weak risk signal?
- PQ-10: Do results change at 10%/30% capacity or under no-salary/multiplicative proxy sensitivities?
- PQ-11: Are subgroup prediction or selection disparities visible by gender and age band?
- PQ-12: If no useful signal exists, what evidence supports rejecting operational use and what additional data would be required?

## 8.5 Intended users, audience, and stakeholders

Primary audience: module evaluators, viva panel, group members, and future AI assistants supporting the academic project.

Hypothetical domain stakeholder: HR analysts considering where limited review/support capacity could be directed. This stakeholder is illustrative only; the dataset's real organizational provenance is unverified.

Affected stakeholders include employees represented by records, especially because false positives, false negatives, sensitive attributes, and value-proxy design can create harms.

## 8.6 Assignment requirements

| ID | Requirement | Project response |
| --- | --- | --- |
| REQ-001 | Use assigned dataset and frame aligned real-world problem. | Employee resignation risk and constrained retention-prioritization simulation. |
| REQ-002 | Describe source, features, target, size, quality, suitability. | Sections 8.10-8.14 and generated EDA. |
| REQ-003 | Clean/preprocess and analyse with visualizations. | Shared audit plus model-specific fold-local adapters. |
| REQ-004 | Select and justify at least two suitable algorithms. | Six required advanced models, one per member: FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, TabNet. |
| REQ-005 | Implement in Python with tuning and pipelines. | Bounded searches, adapters, saved configuration, tests, and artefacts. |
| REQ-006 | Use metrics, k-fold CV, compare, justify best performer. | Five outer folds, PR-AUC primary, calibration/uncertainty/complexity comparison. |
| REQ-007 | Discuss ethics and bias mitigation. | Section 8.24 plus subgroup analysis. |
| REQ-008 | Prepare professional report, presentation, and AI-use declaration. | Sections 8.28-8.35. |

## 8.7 Assessment and acceptance criteria

| Assessment area | Evidence required |
| --- | --- |
| Problem framing and dataset understanding | Accurate source/target/feature/quality explanation and limitations |
| Preprocessing and EDA | Reproducible audit, leakage-safe transformations, meaningful visuals and insights |
| Model design and implementation | Six justified headline models; bounded tuning; model-specific adapters; shared folds; reproducibility |
| Evaluation and comparison | Five-fold CV, relevant metrics, OOF coverage, calibration, uncertainty, runtime, justified final selection |
| Report and teamwork | Professional evidence-backed report, ethics, AI transparency, reflections, references, clear individual model ownership |

Acceptance is based on correctness and critical evaluation, not on obtaining a high score. A verified no-signal outcome is acceptable.

## 8.8 Scope boundaries

### In scope

- binary classification of `Resigned`;
- data audit, EDA, and leakage controls;
- FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, TabNet;
- Dummy references and the required PCA Logistic benchmark; broader standalone Logistic use remains optional;
- model-specific preprocessing and bounded tuning;
- five-fold development CV and one-time formal holdout;
- calibration, threshold analysis, uncertainty, error analysis;
- top-20% prioritization with 10%/30% sensitivity;
- proxy sensitivity, subgroup analysis, ethics, reproducible artefacts;
- report, presentation, viva, and AI-use records.

### Out of scope unless officially required and logged

- deployment, web/mobile interface, dashboard, API, database, cloud system;
- causal estimation of why employees resign or whether interventions work;
- actual monetary budget optimization;
- verified employee valuation or replacement-cost estimation;
- model stacking, voting, additional foundation models, or unbounded model shopping;
- real individual employment decisions or automated action.

## 8.9 Correct terminology and prohibited claims

Use: **resignation-risk estimate**, **retention-value proxy**, **limited-capacity simulation**, **association**, **model behaviour**, **academic decision-support experiment**, **subgroup audit**.

Do not use without evidence: **employee worth**, **guaranteed resignation**, **cause of resignation**, **employees HR should save**, **money protected**, **optimal budget**, **fair model**, **production-ready**, **real employee data**, **full marks guaranteed**.

Complexity must be described accurately: FT-Transformer is a deep tabular model, but its use does not prove superiority. EBM is inherently interpretable but still non-causal. CatBoost explanations are model-behaviour explanations, not causal explanations.

## 8.10 Source material and provenance

| Source ID | Source | Authority and use | Verified notes |
| --- | --- | --- | --- |
| SRC-001 | `Group Assignment Specification.pdf` and Section 9.2 extract | Highest authority for assignment requirements and marks. | Checksum recorded; Section 9 contains a normalized project-relevant extract, not a complete transcript. |
| SRC-002 | `Proposal Presentation.pdf` and Section 9.3 extract | Highest authority for proposal content and duration. | Checksum recorded; Section 9 excludes exact administrative timing. |
| SRC-003 | `archive.zip` | Uploaded assigned dataset package. | Contains one CSV; archive checksum recorded. |
| SRC-004 | `Extended_Employee_Performance_and_Productivity_Data.csv` | Authoritative project artefact for all data facts. | 100,000 x 20; mechanically inspected; checksum recorded. |
| SRC-005 | [Kaggle dataset page](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data) | Third-party listing for source link and licence metadata. | Listing reports CC0: Public Domain. It is not evidence of original organizational provenance. |
| SRC-006 to SRC-008; SRC-010 to SRC-011 | Prior dated conversation decisions | Historical authority for earlier approved changes. | Superseded items remain historical; newer logged decisions take precedence. |
| SRC-012 | Conversation decision on 2026-08-06 | Historical authority for the superseded three-model / three-pair registry. | CatBoost, EBM, FT-Transformer required at that time; Dummy mandatory; Logistic optional. |
| SRC-013 | Conversation decision on 2026-08-07 | Authority for the active six-model / individual-ownership registry. | M1 FT-Transformer; M2 CatBoost; M3 XGBoost; M4 LightGBM; M5 EBM; M6 TabNet. |
| SRC-014 | Lecturer/course PCA requirement on 2026-08-19 | Higher-authority methodology requirement for dimensionality reduction. | Requires PCA variance-retention/loadings analysis and the PCA Logistic Regression demonstration benchmark. |
| SRC-015 | User-approved strict nested evaluation + verified live-repository audit on 2026-09-11 | Authority for the active final headline-model evaluation protocol and current repository-state synchronization. | Requires outer-validation independence from tuning/selection and treats current pre-harmonization model runs as development evidence rather than automatically final unbiased estimates. |
| SRC-016 | User-approved single-file Markdown context policy on 2026-09-11 | Authority for the active persistent-handoff format. | One canonical Markdown context file only; prior mirrored-context workflow retired. |

### Provenance position

**DECIDED** - The project will say that the dataset is the exact assigned Kaggle CSV, while original organization, sampling process, collection procedure, and generation method are not documented in the supplied evidence. Deterministic and highly regular structures are consistent with algorithmic construction, so the data will be treated as likely synthetic or constructed for limitation analysis without presenting that inference as confirmed fact.

## 8.11 Input and dataset profile

| Property | Verified value |
| --- | --- |
| Archive | `archive.zip` |
| CSV | `Extended_Employee_Performance_and_Productivity_Data.csv` |
| CSV SHA-256 | `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb` |
| Rows | 100,000 |
| Columns | 20 |
| Storage format | CSV, ASCII text |
| Missing values | 0 total |
| Exact duplicate rows | 0 |
| Duplicate `Employee_ID` values | 0 |
| Target | `Resigned` Boolean |
| Positive class | `True`: 10,010 records (10.01%) |
| Negative class | `False`: 89,990 records (89.99%) |
| Licence | CC0: Public Domain according to Kaggle listing |
| Salary unit | Unverified; must not be assumed |
| Original provenance | Undocumented in supplied evidence |

| Column | Semantic/storage type | Observed values | Role | Approved treatment | Important limitation |
| --- | --- | --- | --- | --- | --- |
| Employee_ID | Integer | 1 to 100,000; unique | Identifier | Exclude from model; preserve for row/split traceability | Index-like leakage/noise if modelled |
| Department | Categorical | 9 categories | Predictor | Model-specific fold-local categorical handling | May proxy organizational structure; all job-title combinations occur |
| Gender | Categorical | Female, Male, Other | Sensitive predictor/audit attribute | Use in full model; model-specific fold-local handling; exclude in fairness sensitivity models | Potential direct discrimination; subgroup sizes must be reported |
| Age | Integer | 22 to 60 | Sensitive predictor/audit attribute | Retain in full model; scale only where the model adapter requires it; exclude in second fairness sensitivity model | Potential age discrimination; no causal interpretation |
| Job_Title | Categorical | 7 categories | Predictor | Model-specific fold-local categorical handling | Strong deterministic relationship with salary |
| Hire_Date | Date-time text | 2014-09-07 to 2024-09-03; identical time component | Audit field | Parse for audit; exclude raw value from primary model | Near-duplicate of tenure; generation artefact |
| Years_At_Company | Integer | 0 to 10 | Predictor | Retain as tenure representation | Near-redundant with Hire_Date; implied hire age can be implausible |
| Education_Level | Categorical | High School, Bachelor, Master, PhD | Predictor | Model-specific fold-local categorical handling | Do not assume equal spacing without a separate sensitivity test |
| Performance_Score | Integer/ordinal | 1 to 5 | Predictor | Retain numeric or ordinal; document representation | Salary is deterministically based partly on this field |
| Monthly_Salary | Numeric | 3,850 to 9,000; 28 distinct values | Predictor | Retain in primary model; unit unverified; optional redundancy sensitivity | Completely determined by Job_Title and Performance_Score |
| Work_Hours_Per_Week | Integer | 30 to 60 | Predictor | Retain; fold-local scaling only for adapters that require it | Bounded regular distribution; no automatic outlier removal |
| Projects_Handled | Integer | 0 to 49 | Predictor | Retain; fold-local scaling only for adapters that require it | Count semantics/time window undocumented |
| Overtime_Hours | Integer | 0 to 29 | Predictor | Retain; fold-local scaling only for adapters that require it | Time window and units undocumented |
| Sick_Days | Integer | 0 to 14 | Predictor | Retain; fold-local scaling only for adapters that require it | Time window undocumented; potentially sensitive workplace attribute |
| Remote_Work_Frequency | Integer percentage | 0, 25, 50, 75, 100 | Predictor | Retain numeric; optional categorical sensitivity | Meaning appears percentage but observation window undocumented |
| Team_Size | Integer | 1 to 19 | Predictor | Retain; fold-local scaling only for adapters that require it | Organizational context undocumented |
| Training_Hours | Integer | 0 to 99 | Predictor | Retain; fold-local scaling only for adapters that require it | Time window undocumented |
| Promotions | Integer count | 0 to 2 | Predictor | Retain; document numeric/ordinal representation; scale only where required | Time window undocumented |
| Employee_Satisfaction_Score | Numeric | 1.00 to 5.00 | Predictor | Retain; fold-local scaling only for adapters that require it | Measurement instrument and timing undocumented |
| Resigned | Boolean | False or True | Target | Map True=1 and False=0 | Label timing and definition beyond Boolean status undocumented |

## 8.11.1 Exact technical data contract

This contract is intended for direct code generation. It reflects the mechanically verified assigned CSV and must be re-asserted in `data_validation.py` before modelling.

- Filename: `Extended_Employee_Performance_and_Productivity_Data.csv`
- SHA-256: `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb`
- Encoding: ASCII text; valid UTF-8 decoding
- Delimiter: comma
- Header row: present
- Expected row count: 100,000
- Expected column count: 20
- Nullability: no null values are expected in any raw column
- Boolean target mapping: `False -> 0`; `True -> 1`
- Unknown categorical values: fail the validation report for audit visibility. Handling is model-specific and fold-local: XGBoost/the Logistic benchmark use unknown-safe one-hot encoding; CatBoost uses native strings; LightGBM uses training-fold category mappings; EBM uses nominal types; FT-Transformer and TabNet use explicit unknown integer codes.

Exact ordered header:

```text
Employee_ID,Department,Gender,Age,Job_Title,Hire_Date,Years_At_Company,Education_Level,Performance_Score,Monthly_Salary,Work_Hours_Per_Week,Projects_Handled,Overtime_Hours,Sick_Days,Remote_Work_Frequency,Team_Size,Training_Hours,Promotions,Employee_Satisfaction_Score,Resigned
```

Expected raw pandas-compatible types after loading and explicit conversion:

| Column group | Expected type | Validation rule |
| --- | --- | --- |
| `Employee_ID`, `Age`, `Years_At_Company`, `Performance_Score`, `Monthly_Salary`, `Work_Hours_Per_Week`, `Projects_Handled`, `Overtime_Hours`, `Sick_Days`, `Remote_Work_Frequency`, `Team_Size`, `Training_Hours`, `Promotions` | Integer | Reject non-integral or missing values. |
| `Employee_Satisfaction_Score` | Float | Reject missing/non-numeric values; verify observed 1-5 range. |
| `Department`, `Gender`, `Job_Title`, `Education_Level` | String/category | Verify expected category register; report unexpected values. |
| `Hire_Date` | String parsed to datetime | Parse all values; raw field remains excluded from the primary feature matrix. |
| `Resigned` | Boolean | Permit only `False` and `True`; map `True` to the positive class. |

Environment contract: formal target remains Python 3.11.x (tested development environment is Python 3.14.0 on Windows x64). The repository must maintain a tested dependency specification (`requirements.txt`, `environment.yml`, or equivalent) and record the environment used for formal execution.

## 8.12 Categories, ranges, and verified distributions

### 8.12.1 Categorical counts

#### `Department`

| Value | Count |
| --- | --- |
| Customer Support | 11116 |
| Engineering | 10956 |
| Finance | 11200 |
| HR | 10960 |
| IT | 11131 |
| Legal | 11118 |
| Marketing | 11216 |
| Operations | 11181 |
| Sales | 11122 |

#### `Gender`

| Value | Count |
| --- | --- |
| Female | 48001 |
| Male | 48031 |
| Other | 3968 |

#### `Job_Title`

| Value | Count |
| --- | --- |
| Analyst | 14261 |
| Consultant | 14210 |
| Developer | 14195 |
| Engineer | 14217 |
| Manager | 14325 |
| Specialist | 14507 |
| Technician | 14285 |

#### `Education_Level`

| Value | Count |
| --- | --- |
| Bachelor | 50041 |
| High School | 30004 |
| Master | 14904 |
| PhD | 5051 |

#### `Remote_Work_Frequency`

| Value | Count |
| --- | --- |
| 0 | 19837 |
| 25 | 20235 |
| 50 | 19706 |
| 75 | 20173 |
| 100 | 20049 |

#### `Performance_Score`

| Value | Count |
| --- | --- |
| 1 | 20120 |
| 2 | 20013 |
| 3 | 19999 |
| 4 | 19940 |
| 5 | 19928 |

#### `Promotions`

| Value | Count |
| --- | --- |
| 0 | 33296 |
| 1 | 33436 |
| 2 | 33268 |

### 8.12.2 Numeric ranges

| Column | Minimum | Maximum | Mean | Median | Std. deviation |
| --- | --- | --- | --- | --- | --- |
| Age | 22.00 | 60.00 | 41.029 | 41.000 | 11.244 |
| Years_At_Company | 0.00 | 10.00 | 4.476 | 4.000 | 2.869 |
| Performance_Score | 1.00 | 5.00 | 2.995 | 3.000 | 1.415 |
| Monthly_Salary | 3850.00 | 9000.00 | 6403.211 | 6500.000 | 1372.509 |
| Work_Hours_Per_Week | 30.00 | 60.00 | 44.957 | 45.000 | 8.942 |
| Projects_Handled | 0.00 | 49.00 | 24.431 | 24.000 | 14.470 |
| Overtime_Hours | 0.00 | 29.00 | 14.515 | 15.000 | 8.664 |
| Sick_Days | 0.00 | 14.00 | 7.009 | 7.000 | 4.332 |
| Remote_Work_Frequency | 0.00 | 100.00 | 50.090 | 50.000 | 35.351 |
| Team_Size | 1.00 | 19.00 | 10.014 | 10.000 | 5.495 |
| Training_Hours | 0.00 | 99.00 | 49.506 | 49.000 | 28.890 |
| Promotions | 0.00 | 2.00 | 1.000 | 1.000 | 0.816 |
| Employee_Satisfaction_Score | 1.00 | 5.00 | 2.999 | 3.000 | 1.151 |

## 8.13 Data-quality, provenance, and signal audit

### 8.13.1 Structural quality

- **VERIFIED** - No missing values.
- **VERIFIED** - No exact duplicate rows.
- **VERIFIED** - `Employee_ID` is unique.
- **VERIFIED** - All dates parse successfully.
- **VERIFIED** - All 63 combinations of 9 departments and 7 job titles occur.

### 8.13.2 Construction-like artefacts

- **VERIFIED** - Every `Hire_Date` has the identical time component `08:03:05.556036`.
- **VERIFIED** - `Hire_Date` and `Years_At_Company` are almost perfectly redundant; using the dataset maximum date as an anchor, floored tenure matches 99,020 of 100,000 rows.
- **VERIFIED** - `Monthly_Salary` is completely determined by `Job_Title` and `Performance_Score`: all 35 combinations map to one salary value.
- **VERIFIED** - Only 28 distinct salary values occur.
- **VERIFIED** - `Age - Years_At_Company` implies a starting age below 18 in 3,784 records and below 16 in 1,503 records.
- **INFERENCE** - These regularities are consistent with an algorithmically generated or constructed dataset. The source does not authoritatively confirm generation.

### 8.13.3 Class imbalance

- Positive records: 10,010 (10.01%).
- Negative records: 89,990 (89.99%).
- A classifier predicting only the negative class obtains 89.99% accuracy.
- Therefore, raw accuracy is secondary and must be interpreted against this baseline.

### 8.13.4 Audit-only target-signal diagnostic and exposure register

The following values were produced during the context audit solely to determine whether the target framing was defensible. They are not formal project results and must be reproduced independently through the approved repository.

| Audit model | ROC-AUC | Average Precision | Accuracy | Balanced accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Class-weighted Logistic Regression | 0.4906 | 0.0976 | 0.5021 | 0.4923 | 0.0973 | 0.4800 | 0.1618 |
| Balanced Random Forest | 0.4971 | 0.1005 | 0.8999 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Reference | 0.5000 | 0.1001 | 0.8999 | 0.5000 | Not applicable | 0.0000 | 0.0000 |

**Interpretation:** The audit evidence is consistent with little usable target signal. The formal project must test this conclusion using the approved development cross-validation, bounded tuning, and one-time formal-holdout corroboration. It must not repeatedly tune until a favourable score appears.

| Exposure field | Recorded value |
| --- | --- |
| Purpose | Preliminary project-framing diagnostic only |
| Dataset checksum | SRC-004 checksum recorded |
| Exact split method | Not retained in the prior audit record |
| Random seed | Not retained |
| Evaluation row IDs or indices | Not retained |
| Exact preprocessing and code commit | Not retained at repository-grade reproducibility |
| Holdout-data exposure | Some holdout evaluation occurred; exact overlap with the future formal holdout cannot be proven or excluded |
| Formal implication | A later 20% holdout may be locked against all future tuning, but it must not be called guaranteed pristine from all historical audit exposure |
| Formal evaluation policy | Five-fold stratified CV on the development partition is the primary comparison evidence; the locked 20% formal holdout is one-time corroboration with this limitation disclosed |

**DECIDED:** Preserve the diagnostic for audit transparency; do not cite it as a final finding; do not hide or delete the exposure limitation.

## 8.14 Assumptions and constraints

| ID | Statement | Status | Handling |
| --- | --- | --- | --- |
| A-001 | The uploaded CSV is the assigned artefact. | VERIFIED | Protect by checksum and filename. |
| A-002 | `Resigned=True` is the positive event of interest. | DECIDED | Use consistently in all metrics and plots. |
| A-003 | Salary units are unknown. | VERIFIED unresolved semantics | Report values without currency. |
| A-004 | Original collection and generation process are unknown. | VERIFIED limitation | Use cautious provenance wording. |
| A-005 | Rows are treated as independent records for the assignment. | DECIDED pragmatic assumption | State that organization/time clustering cannot be tested. |
| A-006 | The current dataset is static. | SUPPORTED by listing | Record checksum; re-audit if changed. |
| A-007 | Computing resources are suitable for conventional tabular models. | DECIDED | Use bounded searches. |
| A-008 | HR capacity is represented as an equal-cost selected fraction, not monetary amounts. | DECIDED modelling assumption | Primary 20%; sensitivity 10% and 30%. |
| A-009 | The retention-value proxy is an illustrative ordinal prioritization aid, not true employee value. | DECIDED | Publish formula and sensitivity results. |
| A-010 | The primary proxy uses equal weights because no verified business weights exist. | DECIDED | Do not tune weights on the formal holdout. |
| A-011 | No intervention-effect data exist. | VERIFIED limitation | Do not claim selected support prevents resignation. |
| A-012 | Probability estimates require calibration assessment before use in the combined priority score. | DECIDED | Report Brier/reliability and freeze calibration choice. |
| A-013 | The six headline models require model-specific representations rather than one shared transformed matrix. | DECIDED | Same rows/folds; separate fold-local adapters and tests. |
| A-014 | CatBoost, XGBoost, LightGBM, InterpretML, PyTorch/RTDL, and pytorch-tabnet APIs must be validated in the project environment before use. | VERIFIED implementation constraint | Maintain a tested dependency specification in the repository and run smoke tests. |
| C-001 | Final report maximum is 15 pages. | VERIFIED | Prioritize high-mark sections and concise visuals. |
| C-002 | At least two algorithms and k-fold CV are required. | VERIFIED | Use six headline models and five-fold CV; retain mandatory Dummy baselines and the required PCA Logistic benchmark; broader standalone Logistic Regression remains optional. |
| C-004 | No deployment is required. | VERIFIED absence plus DECIDED scope | Do not prioritize application development. |

## 8.15 Deliberately unresolved placeholders

| Placeholder ID | Required answer | How to obtain it | Blocks current work? |
| --- | --- | --- | --- |
| GROUP-NUMBER | Official group number. | Group or LMS supplies it. | No |
| LAB-BATCH-OR-SESSION | Lab batch/session identifier. | Group timetable or module information supplies it. | No |
| TEAM-MEMBER-NAMES | Names and student IDs mapped to Member 1-6 roles. | Group supplies identities and confirms ownership. | No |
| MODULE-COORDINATOR-OR-LECTURER | Name/title used on formal documents. | Official module information. | No |

## 8.16 Final methodology and technical decisions

| Decision | Active value |
| --- | --- |
| Target | `Resigned`, positive class `True` |
| Split | Stratified 80% development / 20% formal holdout, seed 42 |
| Outer CV | Five saved stratified outer folds on development data |
| Final headline-model evaluation | **Strict nested outer-CV / inner-selection**: for every outer fold, all hyperparameter selection, class/loss-weight selection, preprocessing decisions, and early stopping use only that outer fold's training partition; the outer-validation fold is prediction-only after fold-specific selection |
| Primary metric | Average Precision / PR-AUC |
| Required headline models | FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, TabNet |
| Individual ownership | M1 FT; M2 CatBoost; M3 XGBoost; M4 LightGBM; M5 EBM; M6 TabNet |
| Mandatory baselines | Dummy prior and most-frequent |
| Logistic Regression | Required specifically for the PCA demonstration benchmark; broader standalone use remains optional |
| Early stopping | Training-only inner validation inside each outer training fold where required |
| Class imbalance | Model-appropriate class/loss weighting considered inside each outer-training selection process; no pre-split resampling |
| Calibration | OOF Brier/reliability assessment; development-only calibration if justified |
| Thresholds | Report reference 0.50 plus a secondary F1-maximizing threshold derived from development OOF evidence only |
| Fairness age bands | `22-29`, `30-39`, `40-49`, `50-60` |
| Primary capacity | Top 20%; fixed 10% and 30% sensitivity |
| Priority score | Frozen calibrated resignation probability × frozen retention-value proxy |
| Holdout | Used once only after model, preprocessing, calibration, proxy, capacity, strategy, and threshold freeze; historical preliminary-audit exposure caveat remains mandatory |
| Negative result | Valid and required if models remain near baseline |

No headline model is preselected as the winner. Complexity, popularity, or deep-learning status is not evidence of superior performance.

Current development results produced before full strict-nested-protocol harmonization remain valuable development evidence, but they must not automatically be labeled the final unbiased headline-model estimates.

## 8.17 End-to-end workflow and experimental protocol

1. Verify exact CSV filename, SHA-256, shape, schema, and target orientation.
2. Preserve raw data unchanged.
3. Reproduce and peer-review the data audit.
4. Use `Employee_ID` only as a deterministic row identifier; never as a predictor.
5. Use the one saved stratified 80/20 development/formal-holdout split.
6. Use the five saved stratified outer folds within development data.
7. Freeze the feature contract and model-specific adapter contracts.
8. Perform development-only EDA without holdout-guided design.
9. Execute mandatory Dummy baselines and the required PCA Logistic benchmark.
10. Before final headline-model comparison, finalize bounded model-family search spaces and remove unresolved configuration TODOs.
11. For **each model and each outer fold**:
    1. designate the outer-validation fold as prediction-only;
    2. use only the corresponding outer-training partition for preprocessing fitting, candidate search, class/loss-weight selection, and any feature/representation choices;
    3. perform inner cross-validation or a training-only inner validation split for tuning and early stopping;
    4. select the fold-specific configuration using inner evidence only;
    5. refit as required on the allowed outer-training data using only development-derived controls;
    6. generate predictions for the untouched outer-validation rows exactly once;
    7. save row IDs, configuration, preprocessing metadata, runtime, seed/device, selected epoch/tree count, failures, and probabilities.
12. Concatenate the five prediction-only outer-validation outputs into complete OOF predictions.
13. Verify complete OOF coverage, no duplicate/missing rows, identical row order/fold membership, and no formal-holdout IDs.
14. Calculate prediction metrics, fold uncertainty, calibration diagnostics, runtime, reproducibility, and valid explanation evidence.
15. Compare all six headline models using the strict nested OOF evidence.
16. Freeze the selected model family, preprocessing contract, hyperparameter-selection procedure, probability source/calibration plan, thresholds, proxy design, capacity scenarios, and ranking strategies.
17. Refit the frozen selected pipeline on the full development partition using only development-derived controls.
18. Access the formal holdout once for corroborative prediction and prioritization evaluation.
19. Perform final error, interpretation, subgroup, capacity, and proxy-sensitivity analysis without changing the frozen pipeline in response to holdout results.
20. Save artefacts, update logs, write the report, and rehearse the viva.

### Experimental fairness rule

All six members receive the same development rows, outer folds, target definition, primary metric, and evaluation contract. Search budgets may differ modestly by model family but must be bounded and fixed before each final nested evaluation. No member may use an outer-validation fold to select hyperparameters, class weights, early-stopping epochs, or preprocessing decisions. A member may not expand tuning merely because their model is underperforming.

## 8.18 Architecture, repository, data handling, and leakage controls

### 8.18.1 Component responsibilities

The following table defines the intended architectural responsibility of each component. It does **not** imply that every listed module is currently implemented; live implementation status is recorded in Sections 5.3 and 8.32.

| Component / Module | Responsibility | Required output |
| --- | --- | --- |
| `src.data.validation` | checksum/schema/types/ranges/target checks | validation report |
| `src.data.splits` | save/reload 80/20 split and five outer folds | row-level split/fold registry |
| `src.data.feature_contract` | predictors/exclusions/types/sensitive fields | versioned feature contract |
| `src.models.base` | shared adapter interface (`ModelAdapter`) and row-parity assertions | adapter metadata |
| `src.models.runner` | cross-validation and out-of-fold execution harness | OOF predictions and fold metrics |
| `src.models.ft_transformer` | M1 FT architecture/tensor interface/training metadata | model, checkpoint, OOF probabilities |
| `src.models.catboost_model` | M2 native-categorical CatBoost | model, parameters, OOF probabilities |
| `src.models.xgboost_model` | M3 encoded-matrix XGBoost | model, encoder metadata, OOF probabilities |
| `src.models.lightgbm_model` | M4 category-aware LightGBM | model, category metadata, OOF probabilities |
| `src.models.ebm_model` | M5 explicit-type EBM and explanations | model, terms, OOF probabilities |
| `src.models.tabnet_model` | M6 TabNet preprocessing/training/history | model, metadata, OOF probabilities |
| `src.evaluation.metrics` | binary classification and ranking metrics calculation | summary and fold metrics |
| `src.evaluation.oof_validation` | validation and integrity checks on OOF predictions | validation assertions |
| `src.evaluation.calibration` | reliability/Brier/development-only calibration | calibration curves and Brier scores |
| `src.evaluation.fairness` | subgroup prediction/selection audit | subgroup metrics tables |
| `src.evaluation.plots` | visualization utilities for metrics, curves, and distributions | evaluation figures |
| `src.prioritization.ranking` | simulated capacity prioritization strategies and lift analysis | prioritization tables |
| `src.prioritization.value_proxy` | retention-value proxy calculation and sensitivity proxies | proxy score columns |

### 8.18.2 Model-adapter contract

| Model | Input representation | Categorical handling | Numerical handling | Required exported metadata |
| --- | --- | --- | --- | --- |
| Dummy | target statistics/constant strategy | none | none | strategy, prevalence |
| Logistic Regression benchmark | sklearn matrix in Pipeline | fold-fitted one-hot | fold-fitted scaling | feature names, encoder/scaler, coefficients |
| FT-Transformer | separate continuous/categorical tensors | fold-fitted integer vocabularies + embeddings | fold-fitted imputation/standardization | vocabularies, cardinalities, scaler, architecture, epoch, seed/device |
| CatBoost | pandas DataFrame or `Pool` | named native categorical columns | raw approved numeric values unless justified | categorical names/indices, parameters, tree count |
| XGBoost | fold-local encoded matrix | fold-fitted one-hot with unknown handling | numeric matrix; scaling not required for trees | encoder state, feature names, booster parameters |
| LightGBM | pandas/category-aware DataFrame or documented codes | training-fold category mapping | raw approved numeric values | category mapping, feature names, booster parameters |
| EBM | DataFrame/compatible array | explicit nominal feature types | explicit continuous types | feature names/types, bins, terms, interactions |
| TabNet | numeric matrix + categorical metadata | fold-fitted integer codes with `cat_idxs`/`cat_dims` | fold-fitted imputation/standardization | vocabularies, cat metadata, scaler, architecture, epoch/history, seed/device |

All adapters must output probabilities indexed by the same saved row IDs.

### 8.18.3 Leakage and integrity controls

- No preprocessing object is fitted on the full dataset before outer CV.
- No categorical vocabulary, encoder, scaler, binning rule, or category map may inspect outer validation or formal holdout.
- No early-stopping metric may use outer validation or formal holdout.
- No proxy normalizer may use the formal holdout.
- No threshold, calibration method, capacity, or strategy is chosen from formal-holdout performance.
- `Employee_ID` is for tracking only; raw `Hire_Date` is excluded from primary modelling.
- Holdout rows remain inaccessible to member model notebooks until the freeze gate.
- Every OOF file must contain each development row exactly once per model.
- Any skipped fold, NaN probability, duplicate/missing row, or row mismatch invalidates comparison until corrected.

## 8.19 Preprocessing and EDA plan

### 8.19.1 Shared data preparation

- verify checksum, columns/types, and Boolean target mapping;
- preserve raw rows unless an exclusion is justified and logged;
- use the same predictor policy across headline models apart from representation requirements;
- keep `Gender` and `Age` for the primary predictive experiment but perform sensitive-feature ablations;
- exclude `Employee_ID` and raw `Hire_Date`; preserve `Years_At_Company`;
- fit every transformation only within applicable training data.

### 8.19.2 Model-specific preparation

**FT-Transformer (M1)** — fold-local standardized continuous tensors; fold-local categorical vocabularies with missing/unknown codes; training-only inner validation; PyTorch checkpoints; no one-hot categoricals.

**CatBoost (M2)** — categorical strings/column names passed natively; no forced one-hot/scaling; class-weight options tested inside development CV; no holdout `eval_set`.

**XGBoost (M3)** — fold-local `OneHotEncoder(handle_unknown="ignore")` for categorical variables; numerical features passed without tree-unnecessary scaling; inner validation only if early stopping is used.

**LightGBM (M4)** — fold-local category mapping/pandas categorical handling; preserve feature names; no tree-unnecessary standardization; inner validation only if early stopping is used.

**EBM (M5)** — preserve feature names; explicit nominal/continuous feature types; bounded interactions; report final term list.

**TabNet (M6)** — fold-local categorical integer codes and `cat_idxs`/`cat_dims`; fold-local continuous scaling; training-only inner validation; save training history/model reload artefacts.

**Logistic Regression benchmark** — fold-local one-hot + scaling inside a Pipeline. The PCA comparison is required; broader standalone Logistic Regression work is optional and must not delay headline-model completion.

### 8.19.3 EDA questions and figures

The development-only EDA must answer, at minimum:

- What is the `Resigned=True` prevalence, and why does the approximately 10% positive rate make raw accuracy insufficient?
- What are the distributions and ranges of the approved numerical predictors?
- What are the category counts for the approved categorical predictors?
- How do target rates vary across predictors and groups, with explicit association-not-causation wording?
- Which variables show deterministic or near-deterministic redundancy, especially `Monthly_Salary` versus `Job_Title`/`Performance_Score` and `Hire_Date` versus `Years_At_Company`?
- Are there construction-like or provenance limitations that affect interpretation?
- How are `Gender` and `Age` represented before the later subgroup audit?
- What are the distributions of the retention-value-proxy components?
- Which figures/tables are saved, reproducible, and generated from development data only?

EDA must not use formal-holdout outcomes to design features, select models, tune thresholds, choose proxy definitions, or alter the pre-registered capacity scenarios.

### 8.19.4 PCA Dimensionality Reduction Protocol

To fulfill the IT2011 course requirement for unsupervised dimensionality reduction (SRC-014), the project includes a dedicated PCA investigation on numerical employee predictors:

1. **Target Features:** All 13 approved numerical predictors: `Age`, `Years_At_Company`, `Performance_Score`, `Monthly_Salary`, `Work_Hours_Per_Week`, `Projects_Handled`, `Overtime_Hours`, `Sick_Days`, `Remote_Work_Frequency`, `Team_Size`, `Training_Hours`, `Promotions`, `Employee_Satisfaction_Score`. Note: These include both continuous variables and discrete count/ordinal numeric variables.
2. **Exclusions:** `Employee_ID` (identifier), raw `Hire_Date` (unparsed string), and `Resigned` (target).
3. **Standardization:** Input features are centered and scaled using `StandardScaler` fitted strictly on development-training data (`split_registry.csv` development partition).
4. **Unsupervised Construction:** PCA is strictly unsupervised; components are constructed to maximize captured variance across numerical predictors without knowledge of `Resigned`.
5. **Verified Saved PCA Artifacts (`results/pca/` — GENERATED; FINAL NOTEBOOK VERIFICATION PENDING):**
   - `explained_variance.csv` — component-level explained-variance and cumulative-variance evidence.
   - `pca_loadings.csv` — component loadings for the approved numerical predictors.
   - `pca_summary.csv` — retained-component thresholds and PCA summary evidence.
   - `logistic_regression_pca_comparison.csv` — StandardScaler -> LogisticRegression versus StandardScaler -> PCA(>=95% variance) -> LogisticRegression benchmark results.
   - Scree/projection figures are methodologically appropriate outputs, but this canonical record does not claim specific figure files exist unless they are independently verified in the live repository.
6. **Frozen Component Selection Rule:** For predictive modeling in the PCA benchmark comparison, the number of components is frozen to the **minimum number of principal components needed to explain at least 95% ($\ge 0.95$) of variance in the applicable outer-training fold**. PC1 and PC2 are used strictly for 2D visualization and interpretation, not automatically as the predictive component set.
7. **Headline Model Protection:**
   - **CatBoost (M2):** Primary pipeline uses native categorical features without PCA.
   - **XGBoost (M3):** Primary pipeline uses fold-local one-hot encoding without PCA.
   - **LightGBM (M4):** Primary pipeline uses category-aware mappings without PCA.
   - **EBM (M5):** Primary pipeline preserves original feature names and univariate/interaction shapes for inherent interpretability without PCA.
   - **FT-Transformer (M1):** Primary pipeline preserves numerical and categorical feature tokens without PCA.
   - **TabNet (M6):** Primary pipeline preserves continuous scaling and categorical embeddings without PCA.
8. **Required PCA Demonstration Benchmark via Logistic Regression:**
   - Logistic Regression is **not** a seventh headline model; the six headline models remain unchanged.
   - Logistic Regression is **required specifically in a narrow capacity as the PCA demonstration benchmark** comparing:
     - Pipeline A (Standard): `StandardScaler` -> `LogisticRegression`
     - Pipeline B (PCA-Augmented): `StandardScaler` -> `PCA(n_components_95pct)` -> `LogisticRegression`
     evaluated under the identical 5-fold development CV registry.
   - Any broader standalone Logistic Regression experimentation beyond this PCA demonstration remains optional.
9. **Leakage Controls:** Scaler and PCA parameters are fitted strictly on outer-training folds; the formal 20% holdout is never accessed for component selection, variance tuning, or model evaluation. Descriptive development-only PCA may use the full development partition only if clearly labeled descriptive, and must not leak into CV evaluation.


Minimum EDA evidence remains: target prevalence/imbalance; numerical/categorical distributions; target rates with cautious interpretation; feature redundancy/correlation; construction-like patterns; proxy-component distributions; sensitive-group representation; and clear association-versus-causation wording.

## 8.20 Model and algorithm selection

| Model | Status / owner | Why included | Main limitations |
| --- | --- | --- | --- |
| Dummy `prior` | Mandatory shared baseline | prevalence probability and PR-AUC reference | not learned |
| Dummy `most_frequent` | Mandatory shared baseline | exposes misleading majority-class accuracy | no ranking |
| Logistic Regression | Required PCA demonstration benchmark; otherwise optional shared benchmark | transparent conventional reference | basic; may miss nonlinearities |
| FT-Transformer | Required — M1 | distinct Transformer-based deep tabular model | compute, seed/device variability, explanation burden |
| CatBoost | Required — M2 | strong mixed-tabular boosting with native categoricals | post-hoc explanations, calibration/tuning risk |
| XGBoost | Required — M3 | mature high-performance gradient boosting | preprocessing/tuning burden; overlaps with boosting family |
| LightGBM | Required — M4 | efficient leaf-wise boosting and fast large-table training | leaf-wise overfitting risk; categorical/version handling must be controlled |
| EBM | Required — M5 | nonlinear but inherently interpretable additive model | interaction growth can reduce clarity |
| TabNet | Required — M6 | attentive neural architecture designed for tabular data | compute/seed sensitivity; attention masks are not causal explanations |
| Random Forest, Decision Tree, RBF-SVM | Superseded | historical options only | not active |

### Model comparison research questions

1. Which of the six models has the strongest and most stable development PR-AUC?
2. Are differences practically meaningful relative to fold variability?
3. Are probabilities calibrated enough for prioritization?
4. Do FT-Transformer or TabNet justify their extra training complexity?
5. How do CatBoost, XGBoost, and LightGBM differ despite belonging to the boosting family?
6. Does EBM retain useful performance while providing clearer inherent explanations?
7. Does the selected model improve top-k prioritization over repeated random selection?

## 8.21 Tuning and configuration plan

### 8.21.1 Global configuration

- five saved stratified outer folds; Average Precision primary scoring; base seed 42;
- strict nested selection for final headline evidence: all tuning/weighting/early-stopping choices use only the current outer-training partition;
- search spaces must be bounded and finalized before the corresponding final nested evaluation; current TODO-marked configs are preliminary and must not be called frozen;
- prefer explicit bounded candidate lists or a documented bounded randomized protocol; no unbounded model shopping;
- log configuration, inner-selection evidence, fold, runtime, device/seed, failures, and output paths;
- no member may change the shared split/folds or use outer-validation performance to alter that fold's configuration.

### 8.21.2 Logistic Regression benchmark search

The PCA demonstration benchmark is required. Any broader hyperparameter search is optional, capped at eight compatible penalty/`C`/class-weight configurations, and must not delay the six required models.

### 8.21.3 FT-Transformer search — M1

Maximum twelve staged configurations. Candidate blocks 2/3; block dimension 64/128/192; heads 4/8 with divisibility; attention/FFN dropout 0.1/0.2; AdamW learning rate 1e-4/3e-4/1e-3; weight decay 1e-6/1e-5/1e-4; batch 256/512; positive-class weight none or training-fold ratio. Maximum 100 epochs; patience 12 on inner-validation PR-AUC. Save architecture, vocabularies, scaler, curves, selected epoch, seed/device, and reload test.

### 8.21.4 CatBoost search — M2

Maximum twelve configurations over iterations 300/600/1000; depth 4/6/8; learning rate 0.03/0.05/0.1; L2 3/5/10; random strength 0/1/2; class weights none/balanced. Use `Logloss`, seed 42, and only training-inner validation for early stopping if enabled.

### 8.21.5 XGBoost search — M3

Maximum twelve configurations. Primary candidates: `n_estimators` 300/600/1000; `max_depth` 3/5/7; learning rate 0.03/0.05/0.1; `min_child_weight` 1/5/10; subsample 0.8/1.0; `colsample_bytree` 0.8/1.0; `reg_lambda` 1/5/10; `scale_pos_weight` 1 or training-fold negative/positive ratio. Use `binary:logistic`, `tree_method="hist"`, seed 42. Early stopping, if used, uses training-inner validation only.

### 8.21.6 LightGBM search — M4

Maximum twelve configurations. Primary candidates: `n_estimators` 300/600/1000; `num_leaves` 15/31/63; `max_depth` -1/6/10; learning rate 0.03/0.05/0.1; `min_child_samples` 20/50/100; subsample 0.8/1.0 (with fixed `subsample_freq: 1` so bagging takes effect when subsample < 1.0); `colsample_bytree` 0.8/1.0; `reg_lambda` 0/1/5; class weights none/balanced. Use binary objective and seed 42. Early-stopping callbacks may only see training-inner validation.

### 8.21.7 EBM search — M5

Maximum ten configurations over learning rate 0.005/0.015/0.03; max leaves 2/3; minimum samples per leaf 2/4/10; interactions 0/5/10; bounded rounds/patience. Prefer fewer interactions under practical equivalence; internal validation stays inside outer training.

### 8.21.8 TabNet search — M6

Maximum ten staged configurations. Constructor candidate parameters: `n_d`/`n_a` 8/16/32; `n_steps` 3/5/7; `gamma` 1.2/1.5; `lambda_sparse` 1e-5/1e-4/1e-3; `optimizer_params=dict(lr=...)` 0.01/0.02; `cat_idxs`/`cat_dims` from fold training. Fit-time candidate parameters: `batch_size` 1024/2048; `virtual_batch_size` 128/256 with divisibility; `weights` 0 or 1 (automatic inverse-frequency balancing); `max_epochs` 150; `patience` 15. Early stopping on training-inner validation MUST use Average Precision / PR-AUC via a custom `pytorch_tabnet.metrics.Metric` class (e.g. `TabNetAveragePrecision` wrapping `sklearn.metrics.average_precision_score`), not default ROC-AUC. Save training history, preprocessing/category metadata, selected epoch, seed/device, and reload artefact.

### 8.21.9 Threshold, calibration, proxy, and capacity selection

Reference threshold 0.5 plus a secondary F1-maximizing threshold derived only from pooled development OOF predictions. Assess Brier/reliability for all six models; calibration choice remains development-only. The retention-value proxy, 20% primary capacity, 10%/30% sensitivities, and ranking strategies remain frozen before formal holdout.

## 8.22 Evaluation and comparison framework

### 8.22.1 Prediction, calibration, complexity, and transparency metrics

| Category | Evidence |
| --- | --- |
| Primary ranking | Average Precision / PR-AUC |
| Secondary discrimination | ROC-AUC |
| Threshold performance | precision, recall, F1, balanced accuracy, confusion matrix; accuracy with warning |
| Stability | five fold values, mean, standard deviation, failures |
| Calibration | Brier score, reliability curve, optional calibration slope/intercept |
| Runtime/resources | fit/predict time, device, tree/round/epoch count; deep-model parameter count when feasible |
| Reproducibility | seed, environment, reload test, deterministic limitations |
| Interpretation | family-valid EBM/tree/deep explanation evidence |
| Complexity | preprocessing burden, training code, artefact count, compute and maintenance |

### 8.22.2 Prioritization metrics

Precision@capacity, Recall@capacity, Lift@capacity, captured-positive count, value-weighted recall@capacity, selected proxy-value share, and repeated-random uncertainty.

### 8.22.3 Model and strategy selection rule

1. Confirm complete OOF coverage for all six required models.
2. Rank by mean development PR-AUC and inspect fold variation/practical difference.
3. Reject leakage, unstable failures, row mismatches, or unreproducible artefacts.
4. Compare calibration because probabilities feed the priority score.
5. Compare runtime, complexity, valid interpretation, and subgroup behaviour.
6. Under practical equivalence, prefer stronger reproducibility and clearer valid interpretation over unnecessary complexity.
7. Freeze the winner before formal holdout.
8. Do not call a model operationally useful unless predefined usable-signal and top-k lift rules are met.

### 8.22.4 Uncertainty and robustness

Report all fold values/mean/SD; bootstrap formal-holdout metrics when defined; run additional-seed sensitivity for selected deep configurations when feasible; report class-weight, sensitive-feature, capacity, and proxy sensitivity; state when uncertainty prevents a confident winner.

## 8.23 Explainability and interpretability plan

### EBM — M5
Global feature-effect functions, retained interactions, local additive contributions, shape/sparse-region inspection; associations only, not causes.

### CatBoost — M2
Native importance and/or SHAP when reproducible and understood; global summary plus selected local cases; all explanations labelled post-hoc model behaviour.

### XGBoost — M3
Gain/cover/split importance and/or SHAP when validated; preserve encoded-to-original feature mapping; do not infer causality.

### LightGBM — M4
Split/gain importance and/or SHAP when validated; preserve category/feature mapping; explain leaf-wise model behaviour without causal claims.

### FT-Transformer — M1
Permutation importance, grouped ablation/masking, or validated gradient attribution. Raw attention weights must not be treated as causal or automatically as feature importance.

### TabNet — M6
Feature importance and attentive masks may be reported as model-behaviour diagnostics only after implementation validation. They are not causal explanations. Complement with permutation/ablation where feasible.

### Logistic Regression benchmark
Report encoded coefficients with reference-level cautions; associative, not causal.

### Cross-model explanation comparison
Compare broad agreement/disagreement and stability across folds. Do not force family-specific explanation measures onto one numeric scale.

## 8.24 Ethics, fairness, privacy, security, and risk analysis

### 8.24.1 Intended and prohibited use

Intended: academic model comparison and simulated prioritization.

Prohibited: real hiring, firing, promotion, surveillance, disciplinary action, or automated retention decisions.

### 8.24.2 Ethical risks

- false positives may stigmatize employees;
- false negatives may miss employees needing support;
- value proxies may encode salary hierarchy or tenure bias;
- sensitive attributes and correlated proxies may produce disparities;
- explanations may be misread as causal;
- generated scores may create false precision;
- model complexity may obstruct accountability;
- dataset provenance and realism are uncertain.

### 8.24.3 Fairness and subgroup analysis

Report by gender and age band, subject to minimum denominators:

- sample size and prevalence;
- precision, recall, false-positive rate;
- predicted-positive and selection rates;
- representation ratio in top-k;
- value-weighted selection share.

Use the canonical age bands `22-29`, `30-39`, `40-49`, and `50-60`. Any notebook scaffold using different age cohorts must be updated before execution.

Run full, no-Gender, and when feasible no-Gender/no-Age sensitivity models. Separate disparities introduced by risk probability from those introduced by the value proxy.

### 8.24.4 Security and privacy controls

- keep raw data in controlled project storage;
- do not publish credentials or private member information;
- avoid row-level outputs in public reports/slides;
- use row identifiers only for reproducibility;
- scrub document metadata before formal release when required;
- do not infer or publish identities.

## 8.25 Quality assurance and testing plan

| Test ID | Test |
| --- | --- |
| T-001 | CSV checksum, shape, schema, target mapping |
| T-002 | Raw data unchanged |
| T-003 | Split/outer-fold reproducibility and no overlap |
| T-004 | Every transformation fitted only on allowed training rows |
| T-005 | Identical OOF row coverage across FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, TabNet |
| T-006 | Dummy baselines present |
| T-007 | CatBoost receives named native categorical values, not a forced one-hot matrix |
| T-008 | XGBoost encoder fitted on outer training only; unknown categories work |
| T-009 | LightGBM category mapping fitted on outer training only; unknown/missing handling works |
| T-010 | EBM feature names/types and interactions valid |
| T-011 | FT categorical vocabularies fitted on outer training only; tensor dtypes/shapes/cardinalities valid |
| T-012 | TabNet categorical vocabularies/cat metadata and numerical scaling fitted on outer training only |
| T-013 | Inner validation contains only outer-training rows |
| T-014 | Search counts do not exceed frozen caps |
| T-015 | No holdout access before freeze gate |
| T-016 | Probability range, NaN, and row-index checks |
| T-017 | Calibration uses development-only data |
| T-018 | Proxy normalizers fitted development-only |
| T-019 | Capacity selections contain exactly expected k records |
| T-020 | Random baseline reproducible and repeated |
| T-021 | FT-Transformer reload reproduces predictions |
| T-022 | CatBoost reload reproduces predictions |
| T-023 | XGBoost reload plus encoder reproduces predictions |
| T-024 | LightGBM reload plus category metadata reproduces predictions |
| T-025 | EBM reload reproduces predictions |
| T-026 | TabNet reload plus preprocessing/category metadata reproduces predictions |
| T-027 | Explanation artefacts reproducible and correctly labelled |
| T-028 | Subgroup denominators and unstable-rate rules enforced |
| T-029 | Cross-document model/version/team consistency |
| T-030 | Fresh-kernel notebook execution and unit tests pass |

### Review gates

1. Data gate.
2. Split/fold gate.
3. Adapter smoke-test gate.
4. Search-space freeze gate.
5. OOF completeness gate.
6. Model/calibration freeze gate.
7. Formal-holdout access gate.
8. Report/presentation evidence gate.
9. Final cross-document and viva-readiness gate.

## 8.26 Deliverable specifications

| Deliverable | Required content |
| --- | --- |
| Repository | source, notebooks, configs, tests, reproducible paths and instructions |
| Data audit/EDA | saved tables, figures, insights, limitations |
| Split registry | development/holdout and outer-fold membership |
| Prediction artefacts | OOF and formal-holdout probabilities with row IDs |
| FT-Transformer artefacts | state_dict, architecture, vocabularies, scaler/imputer, epoch, curves, seed/device, reload evidence |
| CatBoost artefacts | native model, metadata, parameters, explanations, reload evidence |
| XGBoost artefacts | model/booster, fold-local encoder metadata, parameters, explanations, reload evidence |
| LightGBM artefacts | model/booster, category metadata, parameters, explanations, reload evidence |
| EBM artefacts | model, feature types, terms/interactions, explanations, reload evidence |
| TabNet artefacts | saved model, preprocessing/category metadata, architecture/training history, seed/device, reload evidence |
| Evaluation | common metrics, calibration, uncertainty, runtime, errors |
| Prioritization | strategy/capacity/proxy comparisons |
| Ethics/fairness | subgroup tables, risks, mitigations, limitations |
| Report | professional, evidence-backed, within formal page limit |
| Presentation | required proposal/final content as applicable |
| Viva package | methodology, results, individual ownership, question bank |
| AI-use declaration | actual tools, tasks, extent, verification, ownership |

## 8.27 Demonstration or prototype specification

A GUI is not required. A reproducible notebook or command-line demonstration should:

1. load/verify the data contract;
2. load a frozen model and preprocessing artefact;
3. generate probability predictions for a controlled dataset partition;
4. show model-appropriate explanation evidence;
5. apply the frozen proxy/capacity strategy;
6. output aggregate metrics and selected-row identifiers for academic inspection;
7. display limitations and prohibited-use warning.

Do not expose real-looking employee names or imply operational deployment.

## 8.28 Final report plan

| Section | Required evidence |
| --- | --- |
| 1. Introduction and Problem Statement | two-stage problem, scope, stakeholders, research questions |
| 2. Dataset Description | source, schema, target, size, quality, limitations |
| 3. Preprocessing and EDA | shared audit, model-specific adapters, visuals, insights |
| 4. Model Design and Implementation | Dummy; required PCA Logistic benchmark (broader standalone Logistic optional); FT-Transformer; CatBoost; XGBoost; LightGBM; EBM; TabNet; bounded tuning; fold-local early stopping |
| 5. Evaluation and Comparison | five-fold OOF, PR-AUC, calibration, stability, runtime, interpretation, formal holdout |
| 6. Ethical Considerations and Bias Mitigation | harms, subgroup analysis, proxy risks, prohibited use |
| 7. AI Tool Usage Declaration | actual tools/tasks/extent/verification/ownership |
| 8. Reflections and Lessons Learned | complexity versus evidence, teamwork, limitations, negative-result possibility |
| 9. References | formal sources, dataset, official model documentation, research paper |

### Report writing rules

- Do not spend excessive pages explaining generic algorithms.
- Show why each model is suitable for this dataset and what representation it uses.
- Emphasize the 30-mark evaluation/comparison evidence.
- Report fold variation and no-signal possibility.
- Do not describe Logistic Regression as a main advanced model.
- Do not claim FT-Transformer or TabNet should win because they are deep learning.
- Do not assume CatBoost, XGBoost, and LightGBM are redundant simply because all are gradient-boosting models; compare their actual evidence.
- Keep all claims traceable to saved artefacts.

## 8.29 Proposal presentation plan

### 8.29.1 Working slide structure

| Slide | Content | Lead |
| --- | --- | --- |
| 1 | Title, group, module, two-stage objective | Member 1 |
| 2 | Dataset source, size, target, feature groups | Member 2 |
| 3 | Data quality, imbalance, construction/provenance limitations | Member 2 |
| 4 | Real-world framing and retention-capacity simulation | Member 3 |
| 5 | Six-model portfolio and one-model-per-member ownership | Member 3 |
| 6 | Model-specific preprocessing across boosting, EBM, and deep models | Member 4 |
| 7 | Shared 80/20 split, five-fold CV, leakage controls | Member 4 |
| 8 | Tuning, PR-AUC, calibration, and comparison | Member 5 |
| 9 | Prioritization, ethics, subgroup checks, weak-signal contingency | Member 5 |
| 10 | AI-tool plan, teamwork, next steps | Member 6 |

### 8.29.2 Slide content rules

- State that the assignment requires at least two algorithms; the group will train six headline models, one per member.
- Present Dummy as a baseline and Logistic Regression as required for the PCA demonstration benchmark, with any broader standalone use clearly labelled optional.
- Explain that the six headline models span boosted-tree, interpretable additive, and deep-tabular approaches.
- Do not imply completed model results at proposal stage.
- Do not guarantee full marks or model superiority.
- Explain that all models use identical saved rows/folds with model-specific fold-local preprocessing.
- Make performance, calibration, stability, interpretability, and computational complexity trade-offs visible.

## 8.30 Team responsibility map

### 8.30.1 Individual model ownership

| Member | Required headline model | Primary responsibility | Closely related shared responsibility |
| --- | --- | --- | --- |
| Member 1 | FT-Transformer | preprocessing adapter, architecture, PyTorch training/tuning, checkpoints, OOF probabilities | deep-model seed/device/reload documentation |
| Member 2 | CatBoost | native-categorical training/tuning, OOF probabilities, model explanation artefacts | data validation, split/feature-contract support |
| Member 3 | XGBoost | fold-local encoder, training/tuning, OOF probabilities, explanation artefacts | EDA and visualization support |
| Member 4 | LightGBM | category-aware training/tuning, OOF probabilities, explanation artefacts | common evaluator/runtime/comparison integration |
| Member 5 | EBM | feature types, training/tuning, terms/interactions/local explanations | interpretability and subgroup-analysis support |
| Member 6 | TabNet | fold-local category/scaler adapter, neural training/tuning, history/reload, OOF probabilities | deep-model training-curve and artefact QA |

### 8.30.2 Ownership rule

Each member must personally be able to explain their model's input representation, core idea, important hyperparameters, class-imbalance handling, training process, OOF evaluation, calibration implications, limitations, and saved artefacts. Shared code is allowed, but individual ownership must remain visible in commits/notebooks/results/viva evidence.

### 8.30.3 Shared obligations

All members must understand the dataset, shared split/folds, leakage controls, PR-AUC, calibration, baselines, model comparison, prioritization, limitations, ethics, and actual results. Every member must peer-review at least one other member's model output or shared component. Report writing, presentation practice, prioritization integration, and final QA are shared.

## 8.31 Timeline and milestones

Use milestone dependencies rather than exact administrative dates.

| Milestone | Completion evidence | Status |
| --- | --- | --- |
| M1 Six-model decision synchronized | canonical context and repository model registry | COMPLETED |
| M2 Audit/EDA complete | executed `01_data_audit.ipynb`, `03_eda.ipynb`, saved audit/EDA artifacts | COMPLETED |
| M3 Shared split/folds + feature contract complete | `split_registry.csv`, five 16,000-row folds, contracts/tests | COMPLETED |
| M4 Shared Dummy baselines evaluated | OOF predictions and baseline summary metrics (`04_baselines.ipynb`) | COMPLETED |
| M4a PCA dimensionality-reduction analysis | four canonical CSV artifacts in `results/pca/`; final notebook verification cell still pending | RESULTS GENERATED — FINAL VERIFICATION PENDING |
| M5 Model adapters implemented | adapters present for FT, CatBoost, XGBoost, LightGBM, EBM, TabNet plus Logistic benchmark | COMPLETED |
| M6 Search spaces harmonized for strict nested evaluation | bounded configs exactly aligned with methodology and adapters | ACTIVE — PARTIAL; TODOs remain in five configs |
| M7 Six-model protocol-complete nested evaluation | complete unbiased nested OOF evidence for all six headline models | IN PROGRESS |
| M7a Current pre-harmonization development evidence | FT, XGBoost, TabNet local results; EBM upstream results | AVAILABLE — NOT FINAL NESTED COMPARISON EVIDENCE |
| M8 Model comparison & calibration | `11_model_comparison.ipynb`, `12_calibration.ipynb` | PENDING |
| M9 Fairness & prioritization | `13_fairness_analysis.ipynb`, `14_retention_prioritization.ipynb` | PENDING |
| M10 Formal holdout evaluated once | post-freeze holdout corroboration | NOT PERMITTED BEFORE FREEZE |
| M11 Report/presentation/viva ready | final report <= 15 pages, presentation, QA/viva evidence | PENDING |

## 8.32 Current project status

| Item | Status |
| --- | --- |
| Dataset and target | VERIFIED |
| Two-stage scope | DECIDED |
| Six-model headline registry | DECIDED |
| One model per member ownership | DECIDED |
| Dummy baselines | COMPLETED (`04_baselines.ipynb`) |
| Logistic / PCA benchmark | COMPLETED DEVELOPMENT BENCHMARK; PCA notebook final verification cell still pending |
| Audit notebook | COMPLETED (`01_data_audit.ipynb`) |
| Shared split/fold registry | COMPLETED / VERIFIED (`data/processed/split_registry.csv`) |
| Development-only EDA | COMPLETED (`03_eda.ipynb` with canonical outputs) |
| PCA dimensionality-reduction analysis | RESULTS GENERATED — FINAL VERIFICATION PENDING (`23/24` code cells executed; four CSV artifacts saved) |
| Automated unit test suite | 186 COLLECTED / 186 PASSED (100% passing) |
| Model adapters | IMPLEMENTED for all six headline models plus Logistic benchmark |
| Model candidate configurations | MIXED: TabNet and CatBoost candidate protocols complete; FT/XGBoost/LightGBM/EBM configs require methodology alignment |
| FT-Transformer | Current 5-fold development results saved; final nested-protocol headline reevaluation required |
| CatBoost | COMPLETED FORMAL DEVELOPMENT EVALUATION; strict nested 5-fold CV (pooled AP 0.099350; chance baseline) |
| XGBoost | Current fixed-configuration 5-fold development results saved; final nested tuning/weighting alignment pending |
| LightGBM | Adapter/config/scaffold present; formal nested evaluation pending |
| EBM | Adapter/config present; executed development results available upstream on `origin/main`; local integration/verification pending; final nested-protocol status still to be verified |
| TabNet | T01-T08 development evaluation complete; T01 selected; final strict nested selected-model evaluation required |
| Model comparison | SCAFFOLD — NOT EXECUTED (`11_model_comparison.ipynb`) |
| Calibration | SCAFFOLD — NOT EXECUTED (`12_calibration.ipynb`) |
| Fairness | SCAFFOLD — NOT EXECUTED (`13_fairness_analysis.ipynb`); age-band scaffold must be aligned to canonical bands before execution |
| Retention prioritization | SCAFFOLD — NOT EXECUTED (`14_retention_prioritization.ipynb`) |
| Formal holdout access | LOCKED (`15_final_holdout_evaluation.ipynb`); not permitted until complete development freeze |
| Final report/presentation artefacts | PENDING |

## 8.33 Results record and remaining templates

### 8.33.1 Experiment register

| Experiment ID | Member | Model | Fold/seed | Configuration | Device | Runtime | Output path | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TO BE GENERATED | | | | | | | | |

### 8.33.2 Cross-validation comparison

| Model | Owner | Development AP evidence | AP SD | Evidence/status | ROC-AUC evidence | Brier | Threshold/F1 note | Runtime | Failures |
| --- | --- | --- | ---: | --- | --- | ---: | --- | ---: | --- |
| Dummy prior | Shared | 0.100085 pooled AP reference | | saved development baseline | 0.499917 pooled | | n/a | | 0 |
| Dummy most frequent | Shared | 0.100100 pooled AP reference | | saved development baseline | 0.500000 pooled | | n/a | | 0 |
| Logistic (no PCA) | Shared | 0.100270 pooled AP | | saved development benchmark | 0.499881 pooled | | required PCA comparison reference | | 0 |
| Logistic + PCA (>=95% variance) | Shared | 0.099721 pooled AP | | saved development PCA benchmark | 0.499724 pooled | | required PCA demonstration benchmark | | 0 |
| FT-Transformer | M1 | 0.099696 current CV mean; 0.099979 pooled AP | 0.002237 | current pre-v4.5 development run | 0.497570 CV mean; 0.497254 pooled | 0.247413 pooled | current run only; final nested reevaluation required | | 0 reported |
| CatBoost | M2 | 0.100956 CV mean; 0.099350 pooled AP | 0.003008 | completed formal strict-nested development run | 0.501267 CV mean; 0.500193 pooled | 0.211764 pooled | 0.098728 precision, 0.253871 recall, 0.142168 F1 @ 0.50 | ~368s (~6.1 min) | 0 reported |
| XGBoost | M3 | 0.103245 current CV mean; 0.102628 pooled AP | 0.003592 | current fixed-configuration development run | 0.506130 pooled | | 0.50 currently reported; final nested protocol pending | | 0 reported |
| LightGBM | M4 | TO BE GENERATED | | | | | | | |
| EBM | M5 | UPSTREAM RESULT EXISTS — LOCAL VERIFICATION REQUIRED | | | | | | | |
| TabNet | M6 | 0.101706 selected-candidate pooled AP (T01) | | current candidate-selection evidence; final nested selected-model estimate pending | 0.504556 pooled | | current candidate evidence | | 0 reported |

### 8.33.3 Complexity and evidence comparison

| Model | Owner | Representation | Interpretability | Training complexity | Artefact complexity | Reproducibility | Practical conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FT-Transformer | M1 | | | | | | |
| CatBoost | M2 | | | | | | |
| XGBoost | M3 | | | | | | |
| LightGBM | M4 | | | | | | |
| EBM | M5 | | | | | | |
| TabNet | M6 | | | | | | |

### 8.33.4 Formal-holdout prediction evaluation

| Frozen model | PR-AUC | ROC-AUC | Brier | Precision | Recall | F1 | 95% interval | Caveat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TO BE GENERATED | | | | | | | | Historical audit-exposure caveat |

### 8.33.5 Prioritization strategy comparison at 20%

| Strategy | Selected n | Precision@20 | Recall@20 | Lift@20 | Positive count | Value-weighted recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated random | | | | | | |
| Risk only | | | | | | |
| Proxy value only | | | | | | |
| Risk × proxy value | | | | | | |

### 8.33.6 Capacity and proxy sensitivity

| Model/probability source | Proxy | Capacity | Strategy | Precision | Recall | Lift | Value-weighted recall |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| | | | | | | | |

### 8.33.7 Confusion matrix

| Threshold rule | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| 0.5 | | | | |
| OOF F1 threshold | | | | |

### 8.33.8 Fairness and subgroup template

| Grouping | Group | n | Prevalence | Precision | Recall | FPR | Selection rate | Representation ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gender | | | | | | | | |
| Age band | | | | | | | | |

### 8.33.9 Explainability and error analysis template

| Model | Owner | Method | Global evidence | Local/error cases | Stability | Valid interpretation | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FT-Transformer | M1 | | | | | | |
| CatBoost | M2 | | | | | | |
| XGBoost | M3 | | | | | | |
| LightGBM | M4 | | | | | | |
| EBM | M5 | | | | | | |
| TabNet | M6 | | | | | | |

### 8.33.10 Final conclusion template

- Best development model:
- Practical-equivalence assessment:
- Formal-holdout corroboration:
- Calibration conclusion:
- Did either deep model justify added complexity?:
- Boosting-family comparison conclusion:
- EBM interpretability/performance conclusion:
- Prioritization lift over random:
- Proxy/capacity sensitivity:
- Subgroup findings:
- Operational-use conclusion:
- Additional data needed:

## 8.34 AI-assistant rules for code and writing

### Code rules

- use exact repository paths and current files;
- validate checksum/schema before data work;
- preserve row IDs and saved folds;
- write tests for leakage and OOF coverage;
- keep adapters separate;
- use training-only inner validation for iterative models;
- record seed/device and package environment;
- save configurations and artefacts;
- fail loudly on row mismatch, unknown schema, NaN predictions, or incomplete folds;
- never access holdout before freeze;
- never silently replace FT-Transformer with another model.

### Writing rules

- distinguish VERIFIED, DECIDED, RECOMMENDED, LIVE RECORD, TO BE GENERATED, and SUPERSEDED;
- do not invent scores or completed work;
- do not call model explanations causal;
- do not call the proxy employee worth;
- do not guarantee full marks;
- describe Logistic Regression as required for the PCA demonstration benchmark and otherwise optional;
- describe FT-Transformer as a distinct deep model whose value must be demonstrated;
- keep claims tied to artefacts and formal requirements.

## 8.35 AI-tool plan and actual-use live record

### 8.35.1 Preliminary AI-tool plan

| Tool category | Planned support | Required verification |
| --- | --- | --- |
| General assistants | framing, explanations, report editing, viva questions | compare with formal sources and project decisions |
| Coding assistants | boilerplate, debugging, tests, refactoring | execute, test, peer-review, inspect leakage |
| Research/search tools | official model/package documentation | use primary sources and record access |
| Document tools | canonical Markdown maintenance plus report/presentation generation | structural, content, and visual QA appropriate to each artefact |

### 8.35.2 Actual AI-tool usage - LIVE RECORD

| Record ID | Date | Tool/version | Task supported | Extent | Verification performed | Human ownership/status |
| --- | --- | --- | --- | --- | --- | --- |
| AI-001 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Interpreted assignment/proposal requirements and planned presentation/report structure. | Moderate | Checked against formal PDFs. | User reviewed; group responsible. |
| AI-002 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Audited archive/CSV and identified schema, checksums, class balance, deterministic relationships, and provenance limitations. | Moderate | Code-based inspection and source review. | User approved recommendations. |
| AI-003 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Recommended target, methodology, metrics, ethics, team roles, repository, and no-signal contingency. | Moderate | Tied to rubric and audit evidence. | User approved. |
| AI-004 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Generated canonical Markdown and the prior project-context package under the historical workflow, now retired by SRC-016/Q-052. | Moderate | YAML/content checks and document rendering. | Group owns maintenance. |
| AI-005 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Expanded context to v3 with source extracts, data contract, holdout caveat, evaluation/fairness rules, and synchronization protocol. | Extensive | Validated YAML/headings/checksums and rendered the prior document package. | User requested update. |
| AI-006 | 2026-07-29 | ChatGPT - GPT-5.6 Thinking | Updated title and exact assessed-build definition in v3.1. | Moderate | Content checks and document rendering. | User approved. |
| AI-007 | 2026-07-28 to 2026-08-01 | ChatGPT - GPT-5.6 Thinking | Guided Jupyter audit notebook, result exports, chart creation, GitHub setup, and collaboration workflow. | Extensive | User executed code and supplied screenshots/results. | User performed steps; group retains responsibility. |
| AI-008 | 2026-08-01 | ChatGPT - GPT-5.6 Thinking | Formalized resignation-risk plus constrained retention prioritization; defined capacity, proxies, strategies, metrics, safeguards, and v4.0 context. | Extensive | Rechecked assignment text, preserved dataset evidence, validated Markdown/YAML and rendered the prior document package. | User approved scope; group owns implementation. |
| AI-009 | 2026-08-05 | ChatGPT - GPT-5.6 Thinking | Reviewed model options, researched CatBoost/InterpretML/scikit-learn behaviour, replaced Random Forest/Decision Tree with CatBoost/EBM, and regenerated v4.1 context. | Extensive | Preserved formal requirements and dataset facts; validated model references; rendered and inspected the prior document package. | User approved; group owns execution. |
| AI-010 | 2026-08-06 | ChatGPT - GPT-5.6 Thinking | Recorded canonical repository, removed nonessential handoff gaps, aligned requirement IDs, corrected generated-document tables, and updated v4.1.1 under the historical context workflow now retired by SRC-016/Q-052. | Extensive | Validated YAML, consistency searches, generated-document output, structural and visual QA. | User approved; group owns repository maintenance. |
| AI-011 | 2026-08-06 | ChatGPT - GPT-5.6 Thinking | Created v4.2.0 three-model / three-pair methodology (CatBoost, EBM, FT-Transformer). | Extensive | YAML/content/render QA. | Superseded by v4.3.0 for active model registry and ownership. |
| AI-012 | 2026-08-07 | ChatGPT - GPT-5.6 Sol | Updated the handoff to six individually owned headline models; added XGBoost, LightGBM, and TabNet preprocessing/search/artefact rules; retained shared split/fold/holdout/evaluation safeguards; regenerated the v4.3.0 context package under the historical workflow now retired by SRC-016/Q-052. | Extensive | Reviewed supplied v4.2.0 files, checked current official XGBoost/LightGBM/TabNet APIs, validated machine-readable YAML, stale-reference searches, and generated-document rendering. | User directed the six-model update; group owns implementation, testing, and academic use. |
| AI-013 | 2026-08-19 | Antigravity AI Assistant | Conducted deep codebase/environment verification audit; corrected FT-Transformer (`ffn_d_hidden_multiplier`, `residual_dropout`) and TabNet (`optimizer_params={'lr': ...}`, `__init__` vs `.fit()`) constructor signatures; added LightGBM `subsample_freq=1` requirement; synchronized live state with completed split registry, EDA, baselines, and 92 automated tests; updated repository structure to modular subpackage layout; bumped context to v4.3.1. | Extensive | Executed runtime API tests in Python, verified dataset properties, ran 92 pytest unit tests, inspected repository artifacts, and validated Markdown content consistency. | User directed deep audit; group owns implementation and academic integrity. |
| AI-014 | 2026-08-19 | Antigravity AI Assistant | Integrated IT2011 course-required PCA (Principal Component Analysis) dimensionality-reduction protocol; established variance retention, loadings inspection, scree/projection plots, and PCA-augmented benchmark modeling without disrupting primary headline model representations; updated notebook plan (`04a_pca_analysis.ipynb`) and results directory (`results/pca/`); bumped context to v4.4.0. | Extensive | Validated PCA methodology and leakage rules, verified non-disruptive notebook numbering, checked scikit-learn PCA pipeline compatibility, and updated machine-readable specification. | User directed course-requirement integration; group owns academic execution. |
| AI-015 | 2026-09-11 | ChatGPT - GPT-5.6 Sol + Antigravity repository audit | Re-audited the live 100,000-row repository, rejected a contaminated earlier audit, synchronized canonical live state, and formalized the user-approved strict nested outer-CV / inner-selection protocol for final headline-model evaluation. | Extensive | Repository identity gate, dataset/split SHA checks, notebook/config/source/result inspection, test-suite execution, and cross-check against corrected README/EVALUATION_GUIDE. | User explicitly approved the nested methodology; group owns implementation and academic use. |
| AI-016 | 2026-09-11 | ChatGPT - GPT-5.6 Sol | Performed a complete v4.5.0 consistency audit and produced v4.5.1 as a non-methodological correction release. | Extensive | Full-file scans for version/file-reference drift, Logistic/PCA rule consistency, holdout wording, PCA artifact claims, current model/result status, results-table completeness, YAML parseability, heading/table consistency, and final stale-reference checks. | User requested the full review-fix-review cycle; group owns final adoption and repository synchronization. |
| AI-017 | 2026-09-11 | ChatGPT - GPT-5.6 Sol | Performed the final v4.5.1-to-v4.5.2 full-document consistency pass and correction cycle. | Extensive | Rechecked active/historical version references, Logistic/PCA terminology, holdout wording, source-register coverage, stable-vs-live section mapping, feature-policy consistency, empty sections, Markdown structure, and embedded YAML parseability; repeated stale-pattern scans after fixes. | User requested a complete review-fix-review cycle; group owns final adoption and repository synchronization. |
| AI-018 | 2026-09-11 | ChatGPT - GPT-5.6 Sol | Upgraded v4.5.2 to v4.6.0 by retiring the former mirrored-context workflow and converting the project handoff to one canonical Markdown file only, then performed full-file consistency validation. | Extensive | Removed active secondary-format dependencies, updated source/decision/change records, rechecked version/path/source terminology, validated embedded YAML and Markdown structure, and ran repeated stale-reference scans. | User explicitly directed the single-file policy; group owns final repository adoption. |
| AI-019 | 2026-09-11 | ChatGPT - GPT-5.6 Sol | Re-audited v4.6.0 after the single-file policy change and produced v4.6.1 to remove the last legacy wording that could imply recreating a second persistent context file. | Extensive | Reparsed all YAML blocks, checked Markdown fences/tables/headings, scanned active and historical version references, verified source IDs, and searched for retired multi-file instructions and version drift. | User requested a second independent verification; group owns final repository adoption. |

Future records must include actual tool/version if known, precise tasks, extent, verification, retained evidence, and human ownership.

## 8.36 Decision and change log

| Decision ID | Date | Status | Topic | Active decision | Authority | Reconsideration |
| --- | --- | --- | --- | --- | --- | --- |
| Q-001 to Q-030 | 2026-07-28 to 2026-08-01 | DECIDED / historical | Initial methodology and two-stage scope | Retained except where explicitly superseded | Formal sources and user decisions | Higher-authority or logged change |
| Q-032 | 2026-08-05 | SUPERSEDED | Two-advanced-model registry | CatBoost and EBM only | Earlier user decision | Superseded by Q-039 |
| Q-033 | 2026-08-05 | DECIDED | Earlier core models | Random Forest and Decision Tree removed | User | Explicit reinstatement before holdout |
| Q-034 | 2026-08-05 | SUPERSEDED IN PART | Model-specific preprocessing | CatBoost/EBM rules retained; expanded for FT | User/method requirements | Q-040 |
| Q-035 | 2026-08-05 | SUPERSEDED IN PART | Performance-transparency rule | Expanded to performance-calibration-stability-interpretability-complexity across the then-active model set | User | Q-041 |
| Q-036 to Q-038 | 2026-08-05 to 2026-08-06 | SUPERSEDED IN PART | Earlier multi-format handoff, environment, repository | The earlier multi-format handoff is superseded by Q-052; environment and repository decisions remain retained | User | Q-052 for context-file policy; otherwise logged change |
| Q-039 | 2026-08-06 | SUPERSEDED | Required advanced model registry | CatBoost, EBM, FT-Transformer; Dummy mandatory; Logistic optional | Earlier user decision | Superseded by Q-045 |
| Q-040 | 2026-08-06 | DECIDED | FT-Transformer protocol | RTDL/PyTorch, fold-local standardization/vocabularies, embeddings, training-only inner validation, bounded search, checkpoint/reload | User approval plus official implementation guidance | Verified incompatibility plus approved redesign |
| Q-041 | 2026-08-06 | SUPERSEDED IN PART | Model selection | Common OOF/calibration/stability/runtime/reproducibility/interpretation rule retained and expanded to six models | Methodological review | Q-048 |
| Q-042 | 2026-08-06 | SUPERSEDED | Team division | One advanced model per two-member pair | Earlier user decision | Superseded by Q-046 |
| Q-043 | 2026-08-06 | SUPERSEDED IN PART | Optional/fallback models | Logistic optional and RBF-SVM superseded retained; XGBoost contingency status superseded | Earlier user decision | Q-045 |
| Q-044 | 2026-08-06 | DECIDED | Formal-source handoff content | Keep project-relevant requirement extracts; exclude nonessential exact administrative timing/submission logistics from active handoff | Prior user cleanup instruction | Need for exact official wording |
| Q-045 | 2026-08-07 | DECIDED / SUPERSEDED IN PART BY Q-050 | Six-model headline registry | FT-Transformer, CatBoost, XGBoost, LightGBM, EBM, TabNet; Dummy mandatory; Logistic was optional at that time, later required for the PCA demonstration by Q-050 | Explicit user decision | Higher-authority requirement or logged pre-holdout change |
| Q-046 | 2026-08-07 | DECIDED | Individual model ownership | M1 FT; M2 CatBoost; M3 XGBoost; M4 LightGBM; M5 EBM; M6 TabNet | Explicit user decision / accepted mapping in update request | Group-approved revision before headline training |
| Q-047 | 2026-08-07 | DECIDED | Six-model preprocessing contract | Same rows/folds; model-specific fold-local adapters; no shared full-data transformed matrix | Methodological requirement | Logged pre-holdout redesign |
| Q-048 | 2026-08-07 | DECIDED | Six-model selection rule | Compare all six on common OOF PR-AUC plus calibration, stability, runtime, reproducibility, valid interpretation, practical equivalence | Methodological review | Logged pre-holdout change |
| Q-049 | 2026-08-19 | DECIDED | Parameter signature and live-state synchronization | Correct FT-Transformer, TabNet, and LightGBM search specifications to match installed library APIs; update live state to reflect completed split registry, EDA, baselines, and 92 passing unit tests; update repository layout to modular subpackage tree; bump to v4.3.1 | Technical verification and runtime inspection | Higher-authority or logged change |
| Q-050 | 2026-08-19 | DECIDED | PCA Dimensionality-Reduction Integration | Add course-required PCA dimensionality reduction on standardized numerical features (`04a_pca_analysis.ipynb`, `results/pca/`); include PCA-augmented Logistic Regression benchmark; preserve non-PCA model-specific fold-local representations for all six headline models; bump to v4.4.0 | Module lecturer / course requirement (SRC-014) | Higher-authority or logged change |
| Q-051 | 2026-09-11 | DECIDED | Strict nested headline-model evaluation | For every outer fold, all hyperparameter selection, class/loss-weight selection, preprocessing decisions, and early stopping use only that outer fold's training partition; the outer-validation fold is prediction-only after fold-specific selection. Existing current development results remain evidence but are not automatically final unbiased nested estimates. | Explicit user approval after live-repository audit | Higher-authority requirement or logged pre-holdout methodology change |
| Q-052 | 2026-09-11 | DECIDED | Single-file canonical project context | Maintain one persistent canonical project-context file in Markdown only. The earlier mirrored-format handoff workflow is retired; future context updates occur only in the canonical Markdown file. | Explicit user project-workflow decision (SRC-016) | Higher-authority requirement or explicit logged workflow change |

### Change log

| Version | Date | Change | Rationale | Status |
| --- | --- | --- | --- | --- |
| 2.0.0 | 2026-07-28 | Created AI-first context. | Complete handoff. | SUPERSEDED |
| 3.0.0 | 2026-07-28 | Added data contract, requirements, evaluation, fairness, synchronization. | Execution-safe handoff. | SUPERSEDED |
| 3.1.0 | 2026-07-29 | Adopted assignment-aligned title/build. | Clarify assessed workflow. | SUPERSEDED |
| 4.0.0 | 2026-08-01 | Added constrained retention prioritization. | Connect prediction to limited resources. | SUPERSEDED |
| 4.1.0 | 2026-08-05 | Adopted CatBoost/EBM and model-specific adapters. | More distinctive/interpretable comparison. | SUPERSEDED |
| 4.1.1 | 2026-08-06 | Added repository and handoff cleanup. | Clean synchronized package. | SUPERSEDED |
| 4.2.0 | 2026-08-06 | Locked CatBoost, EBM, and FT-Transformer with three two-member pairs. | Prior active methodology. | SUPERSEDED |
| 4.3.0 | 2026-08-07 | Expanded to six required headline models with one model per member; activated XGBoost, LightGBM, and TabNet; added six-model preprocessing/search/evaluation/team/repository rules while retaining shared split/CV/holdout safeguards. | Satisfy the one-model-per-member requirement without sacrificing fair comparison or leakage control. | SUPERSEDED |
| 4.3.1 | 2026-08-19 | Corrected FT-Transformer constructor kwargs (`ffn_d_hidden_multiplier`, `residual_dropout`) and TabNet parameter routing (`optimizer_params={'lr': ...}`, `__init__` vs `.fit()`); added LightGBM `subsample_freq=1` requirement; synchronized live state with completed `split_registry.csv`, `03_eda.ipynb` (20 outputs), `04_baselines.ipynb`, and 92 automated tests; updated repository structure to modular subpackage tree (`src/data/`, `src/models/`, `src/evaluation/`, `src/prioritization/`). | Ensure runtime code generation accuracy and synchronize persistent context with actual Stage 4 repository progress. | SUPERSEDED |
| 4.4.0 | 2026-08-19 | Integrated IT2011 course-required PCA (Principal Component Analysis) dimensionality-reduction protocol on numerical features; added dedicated analysis workflow (`04a_pca_analysis.ipynb`), artifact directory (`results/pca/`), component loadings inspection, scree/projection plots, and a PCA-augmented Logistic Regression benchmark comparison; preserved non-PCA representations for all six headline models. | Fulfill formal course dimensionality-reduction requirement while preserving model-specific preprocessing and experimental validity. | SUPERSEDED |
| 4.5.0 | 2026-09-11 | Synchronized live repository state; corrected stale PCA/test/model/notebook/config status; formalized strict nested outer-CV / inner-selection for final headline-model evaluation; preserved existing FT/XGBoost/TabNet/upstream EBM results as development evidence rather than automatically final unbiased estimates; aligned fairness age-band and holdout wording. | Prevent model-selection optimism, eliminate stale documentation, and keep the canonical source aligned with verified repository evidence. | SUPERSEDED |
| 4.5.1 | 2026-09-11 | Corrected canonical-status/version drift and prior mirrored-format filename references, Logistic/PCA benchmark wording, malformed holdout language, PCA artifact naming, EBM result-status wording, results-table completeness, and reference numbering; preserved the v4.5.0 nested methodology unchanged. | Produce a consistent patch release without changing approved methodology or verified project state. | SUPERSEDED |
| 4.5.2 | 2026-09-11 | Final consistency patch: corrected residual active-version drift, completed Logistic/PCA wording synchronization, clarified single-file canonical-context handling, corrected stable/live section mapping, removed an XGBoost scaling inconsistency, completed the EDA subsection, expanded active source references, clarified architecture-versus-implementation status, and improved the current-results table semantics. | Eliminate remaining cross-section ambiguity without changing the approved v4.5.0 strict nested methodology or verified repository state. | SUPERSEDED |
| 4.6.0 | 2026-09-11 | Retired the former mirrored-context workflow and established one canonical Markdown context file only; removed all active secondary-format dependencies, updated handoff/update rules, added SRC-016 and Q-052, and revalidated the complete context without changing the approved strict nested ML methodology. | Simplify the persistent handoff, eliminate synchronization risk, and match the user-approved single-file project-context policy. | SUPERSEDED |
| 4.6.1 | 2026-09-11 | Final single-file-policy consistency patch: removed the last historical actionable wording that instructed maintenance of multiple context files, clarified legacy AI-use records as retired workflow history, tightened current maintenance wording, and reran full structural/YAML/stale-reference validation. | Prevent future assistants from reconstructing the retired secondary persistent context while preserving historical audit meaning and leaving the approved ML methodology unchanged. | SUPERSEDED |
| 4.6.2 | 2026-09-12 | Canonical context patch: synchronized completed formal CatBoost strict-nested 5-fold development evaluation (pooled AP 0.099350; chance baseline); synchronized automated test suite to 186/186 passing across 14 modules; purged completed PCA test action items from active queues; preserved approved strict nested ML methodology, candidate schemas, and baseline rules unchanged. | Synchronize canonical project context with verified Step 2C repository evidence, maintain zero stale active items, and uphold Section 7.1 semantic-versioning maintenance rules without altering methodology or assignment scope. | ACTIVE |

## 8.37 Remaining identity and ownership inputs

| ID | Input | Handling | Blocks technical work? |
| --- | --- | --- | --- |
| EXT-001 | Official group number and lab batch/session | Add to formal artefacts when known | No |
| EXT-002 | Six names/student IDs mapped to Members 1-6 | Replace numbered roles after group confirmation | No |
| EXT-003 | Lecturer/module-coordinator name | Use official module information | No |
| EXT-004 | Confirmed CPU/GPU execution environment | Record after smoke test; do not assume GPU | No, but affects FT-Transformer and TabNet runtime |

## 8.38 References and source-file register

### Formal and project sources

1. **SRC-001:** Faculty of Computing. *IT2011 - Artificial Intelligence and Machine Learning: Group Assignment Specification*. Year 2 Semester 1, 2026. Local file and SHA-256 recorded in Section 1.
2. **SRC-002:** Faculty of Computing. *IT2011 - Artificial Intelligence and Machine Learning: Proposal Presentation*. Year 2 Semester 1, 2026. Local file and SHA-256 recorded in Section 1.
3. **SRC-003:** Assigned archive, `archive.zip`; checksum recorded.
4. **SRC-004:** `Extended_Employee_Performance_and_Productivity_Data.csv`; checksum recorded.
5. **SRC-005:** Kaggle, *Employee Performance and Productivity Data*: `https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data`.
6. **SRC-006 to SRC-008; SRC-010 to SRC-011:** Prior explicit user decisions recorded in the machine-readable source register.
7. **SRC-012:** User decision, 2026-08-06: prior CatBoost/EBM/FT-Transformer three-model / three-pair registry; retained as superseded history.
8. **SRC-013:** User decision, 2026-08-07: six required headline models, one per member — M1 FT-Transformer, M2 CatBoost, M3 XGBoost, M4 LightGBM, M5 EBM, M6 TabNet; Dummy mandatory; Logistic was optional at that time. The later PCA requirement in SRC-014 makes the PCA Logistic benchmark required while broader standalone Logistic use remains optional.
9. **SRC-014:** Lecturer/course requirement, 2026-08-19: demonstrate PCA on the approved numerical predictors, including variance retention, loadings interpretation, and a PCA-augmented benchmark comparison.
10. **SRC-015:** User-approved methodology decision and verified live-repository audit, 2026-09-11: use strict nested outer-CV / inner-selection for final headline-model evaluation and treat current pre-harmonization model results as development evidence rather than automatically final unbiased estimates.
11. **SRC-016:** User project-workflow decision, 2026-09-11: maintain one persistent canonical project-context file in Markdown only and retire the prior mirrored-context workflow.

### Official software and research sources supporting the active methodology

12. CatBoost. *CatBoostClassifier Python reference*. `https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier`.
13. CatBoost. *Categorical features*. `https://catboost.ai/docs/en/features/categorical-features`.
14. CatBoost. *Training parameters and parameter tuning*. `https://catboost.ai/docs/en/references/training-parameters/` and `https://catboost.ai/docs/en/concepts/parameter-tuning`.
15. InterpretML. *Explainable Boosting Machine*. `https://interpret.ml/docs/ebm.html`.
16. InterpretML. *ExplainableBoostingClassifier API*. `https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html`.
17. InterpretML. *EBM hyperparameters*. `https://interpret.ml/docs/hyperparameters.html`.
18. Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021. `https://arxiv.org/abs/2106.11959`.
19. Yandex Research. *rtdl-revisiting-models*, official recommended FT-Transformer package implementation. `https://github.com/yandex-research/rtdl-revisiting-models`.
20. scikit-learn. *Probability calibration*. `https://scikit-learn.org/stable/modules/calibration.html`.
21. PyTorch. Official documentation for tensors, reproducibility, optimizers, loss functions, data loading, checkpointing, and device management: `https://pytorch.org/docs/stable/`.
22. XGBoost. *XGBClassifier Python API*. `https://xgboost.readthedocs.io/en/stable/python/python_api.html`.
23. LightGBM. *LGBMClassifier Python API*. `https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html`.
24. DreamQuark. *pytorch-tabnet documentation / TabNetClassifier*. `https://dreamquark-ai.github.io/tabnet/`.

### Reference maintenance rules

- Original formal files remain authoritative.
- Use official documentation and primary research for technical claims.
- Do not put exact package versions in this context; record them in the tested environment artefact.
- Update access dates/environment evidence when formal execution occurs.
- Do not cite AI-generated prose as evidence for model behaviour or results.

# 9. Embedded normalized requirement extracts

## 9.1 Authority and exclusion policy

These extracts preserve the project-relevant requirements from the supplied formal PDFs. They are normalized summaries, not complete verbatim transcripts. The original PDFs remain authoritative.

Deliberately excluded from this active AI handoff:

- exact administrative deadlines and tentative dates;
- submission logistics that do not affect technical design;
- unverified lecturer feedback or proposal-approval claims;
- citation-style or institutional AI-policy details not required for the active methodology.

Use the original PDF when exact wording, page appearance, or a newer revision matters.

## 9.2 SRC-001 - Group Assignment Specification: project-relevant extract

- Module: IT2011 - Artificial Intelligence and Machine Learning.
- Group size: six members.
- Objective: design and implement an end-to-end AI/ML solution for a real-world problem using the assigned dataset and critically evaluate effectiveness and ethical implications.
- Learning outcomes: apply AI/ML techniques and critically evaluate algorithm effectiveness/impact.
- Step 1, 10 marks: understand assigned dataset, frame aligned real-world problem, justify features/target/size/quality, and briefly state planned AI-tool use.
- Step 2, 25 marks: clean/preprocess, perform EDA with visualizations, and provide insights/patterns.
- Step 3, 25 marks: select and justify at least two suitable algorithms, implement in Python, include hyperparameter tuning and pipeline steps.
- Step 4, 30 marks: use relevant metrics, perform k-fold cross-validation, compare models, and justify the best performer.
- Step 5, 10 marks: professional report with introduction/problem, dataset, preprocessing/EDA, model design/implementation, evaluation/comparison, ethics/bias mitigation, AI-tool declaration, reflections, and references.
- AI-tool declaration: record tool/version if known, supported tasks, extent, verification/testing, and group ownership.
- Teamwork, visuals, and clarity are evaluated.

## 9.3 SRC-002 - Proposal Presentation: project-relevant extract

- Use the assigned open-source dataset; dataset selection is not required.
- Include source and direct link/citation.
- Review dataset structure, features, target, size, and format.
- Define a real-world problem aligned with the dataset.
- Justify dataset suitability and discuss quality/limitations.
- Briefly state planned generative-AI tools and intended tasks.
- Format: slide-based PowerPoint.
- Duration: 15 minutes per group.
- Required topics: assigned dataset/source, problem domain, dataset justification, planned AI-tool use.

# 10. Reusable AI update templates

## 10.1 Live-state update template

```yaml
status: planned | in_progress | blocked | completed | superseded
last_updated: YYYY-MM-DD
updated_by: person_or_tool
owner: member_or_role_if_known
evidence:
  - file, commit, result ID, official notice, or source ID
summary: concise current fact
next_action: one concrete next action
```

## 10.2 Experiment record template

```yaml
experiment_id: EXP-XXX
date: YYYY-MM-DD
dataset_sha256: 2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb
split_id: SPLIT-XXX
code_commit: git_commit_hash
environment_file: requirements.txt_or_environment.yml
model: exact_pipeline_name
search_space: recorded_configuration
cv_results_file: path
threshold_rule: reference_0.5_and_oof_F1_threshold
formal_holdout_used: false
results: TO BE GENERATED
verified_by: member_names_or_roles
```

## 10.3 AI-use record template

```yaml
record_id: AI-XXX
date: YYYY-MM-DD
tool_and_version: exact_if_known
tasks_supported:
  - precise task
extent: minimal | moderate | extensive
verification:
  - test, source comparison, manual review, or reproduction method
retained_evidence:
  - prompt/output/file/commit reference
human_ownership: responsible member or group
```

## 10.4 Decision-change template

```yaml
decision_id: Q-XXX
date: YYYY-MM-DD
status: DECIDED | SUPERSEDED
topic: concise topic
value: current decision
reason: evidence and rationale
authority: lecturer | formal source | group | user approval
approved_by: person_or authority
affected_items:
  - code
  - data
  - results
  - report
  - slides
  - timeline
reconsider_if: precise trigger
```

