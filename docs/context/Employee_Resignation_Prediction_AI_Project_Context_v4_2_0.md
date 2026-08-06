---
title: "Design and Evaluation of Machine Learning Models for Employee Resignation Risk and Resource-Constrained Retention Prioritization"
subtitle: "Canonical project specification, live state, and AI handoff memory"
author: "IT2011 Project Group"
date: "2026-08-06"
version: "4.2.0"
---

# AI entry point - read this first

This package is intentionally duplicated in two synchronized formats for AI compatibility.

```yaml
ai_handoff_entry:
  canonical_file: Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.md
  companion_file: Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.docx
  canonical_rule: Read Markdown first. Use Word only as a synchronized fallback or navigation-friendly rendering.
  duplicate_rule: Repeated content across the two files is one source, not independent evidence.
  conflict_rule: If substantive content differs, follow Markdown, report the mismatch, and regenerate Word.
  repository_url: https://github.com/Sankalpa-KMCP/Employee_Resignation_Prediction
  active_target: Resigned
  positive_class: true
  active_model_registry:
    mandatory_shared_baselines:
      - DummyClassifier(strategy="prior")
      - DummyClassifier(strategy="most_frequent")
    optional_lightweight_benchmark:
      - Logistic Regression
    required_advanced_models:
      - CatBoost Classifier
      - Explainable Boosting Machine (ExplainableBoostingClassifier)
      - FT-Transformer (rtdl_revisiting_models.FTTransformer with PyTorch)
    pair_ownership:
      - Pair 1 / Members 1-2: CatBoost plus data audit, preprocessing, split registry, and feature contract
      - Pair 2 / Members 3-4: EBM plus EDA, interpretation, and subgroup-analysis support
      - Pair 3 / Members 5-6: FT-Transformer plus shared evaluation, calibration, prioritization, and deep-model artefacts
    superseded_core_models:
      - Random Forest Classifier
      - Decision Tree Classifier
      - RBF-SVM
    contingency_only:
      - XGBoost, only after a logged pre-holdout substitution decision if FT-Transformer cannot be made reproducible
  primary_prediction_metric: Average Precision / PR-AUC
  primary_capacity: 0.20
  formal_holdout_rule: Never use the formal holdout for preprocessing design, tuning, early stopping, calibration choice, threshold choice, proxy choice, capacity choice, strategy choice, or model selection.
  execution_context:
    conceptual_questions: These two context files are sufficient.
    data_dependent_work: Also require the exact assigned CSV and checksum verification.
    implementation_work: Use the canonical GitHub repository and inspect the relevant current files before making code-dependent claims.
    generated_outputs: Experiment outputs are repository artefacts created during implementation; they are not missing context inputs and must never be invented.
  current_methodology_change: The user-approved v4.2.0 decision locks CatBoost, EBM, and FT-Transformer as the three required advanced models, one per two-member pair. Dummy is mandatory; Logistic Regression is optional.
```

**Fast operational summary:** Build and compare CatBoost, EBM, and FT-Transformer under the same locked development/holdout rows and five outer stratified folds. Use model-specific, fold-fitted preprocessing. Retain two DummyClassifier strategies as mandatory no-skill references; run Logistic Regression only as an optional lightweight benchmark. Select by development evidence, not model reputation. Assess calibration before using probabilities in the retention-prioritization score. Do not claim real employee worth, causal resignation drivers, intervention success, a real monetary budget, operational readiness, legal fairness, or suitability for actual employment decisions.

# Document map

- **AI entry point** - canonical-file rule, active model registry, pair ownership, and artefact requirements.
- **0. AI execution contract** - authority, mandatory behaviour, startup check, and anti-hallucination rules.
- **1. Machine-readable project memory** - structured project, dataset, methodology, live state, requirements, and update rules.
- **2. AI quick reference** - one-minute summary, terminology, and prohibited assumptions.
- **3. Current operating state** - milestone, completed work, blockers, risks, and next actions.
- **4-7. AI operating guidance** - task routing, file manifest, dependencies, synchronization, and update protocol.
- **8. Full canonical project specification** - requirements, dataset evidence, methodology, models, evaluation, ethics, deliverables, team plan, results templates, and logs.
- **9. Embedded normalized requirement extracts** - project-relevant requirements from the supplied formal PDFs, excluding nonessential administrative timing and submission logistics.
- **10. Reusable AI update templates** - live-state, experiment, AI-use, and decision-change records.

The Word file also provides a navigation pane based on these headings.

# 0. AI execution contract

## 0.1 Document purpose

This file is the persistent AI-first project memory for **Design and Evaluation of Machine Learning Models for Employee Resignation Risk and Resource-Constrained Retention Prioritization**. It is intended to let a new AI assistant, team member, reviewer, or future project session understand the project without relying on conversational memory.

**VERIFIED** - The project is an IT2011 group assignment for six members. It requires an end-to-end AI/ML solution, at least two suitable algorithms, Python implementation, hyperparameter tuning, pipelines, k-fold cross-validation, model comparison, ethical evaluation, a 15-minute proposal presentation, and a final report of no more than 15 pages.

**DECIDED** - The project retains `Resigned` as the binary target and now has two linked analytical stages:

1. estimate whether the available employee-related variables contain usable out-of-sample signal for `Resigned=True`; and
2. simulate how limited HR retention capacity could be prioritized when only a fixed share of records can be reviewed or supported.

**DECIDED** - The primary capacity scenario is the top 20% of records. The project compares random selection, risk-only ranking, value-proxy-only ranking, and a combined risk-plus-retention-value-proxy ranking. Capacity sensitivity at 10% and 30% is reported when feasible.

**DECIDED** - The retention-value measure is explicitly a **proxy**, not actual employee worth. The primary proxy is an equal-weight average of development-fitted min-max normalized `Performance_Score`, `Monthly_Salary`, and `Years_At_Company`. A no-salary proxy and a normalized multiplicative proxy are sensitivity analyses because salary is deterministically related to job title and performance in this dataset.

**DECIDED** - The three required advanced models are CatBoost, EBM, and FT-Transformer, assigned one per two-member pair. DummyClassifier is mandatory as a shared no-skill reference; Logistic Regression is optional and not a headline model.

**DECIDED** - The assessed build remains an academic, reproducible Python study. It does not estimate whether a real retention intervention would succeed, does not perform genuine monetary budget optimization, and must not be used for real employment actions. Academic record-level ranking is permitted only inside the controlled simulation on the assigned dataset.

## 0.2 Canonical-source rule

1. This Markdown file is the canonical source of truth for project context.
2. The Word file named `Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.docx` is a synchronized companion generated from this content.
3. If Markdown and Word differ substantively, use the Markdown version and regenerate the Word companion.
4. Code, notebooks, results, reports, and slides must not silently override a **DECIDED** item in this file. A change must be logged first.
5. The assigned CSV is identified by filename and SHA-256. Never substitute a similarly named dataset without an explicit, logged decision and checksum update.

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
- use only development-training data and training-fold/out-of-fold outputs for preprocessing choices, tuning, feature selection, and threshold selection;
- avoid claiming causality, operational readiness, legal compliance, or real-world employee provenance;
- state when a requested conclusion is unsupported by the data;
- update the live state, result record, AI-use record, and decision log after material work;
- report missing companion files, checksum mismatches, schema changes, or requirement conflicts before proceeding.

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
- [ ] The active model registry is mandatory Dummy baselines, optional Logistic Regression, and required CatBoost, EBM, and FT-Transformer.
- [ ] Random Forest, Decision Tree, and RBF-SVM appear only as historical references unless a new logged decision reinstates them.
- [ ] CatBoost, EBM, and FT-Transformer use separate appropriate feature representations but identical saved rows/folds; optional Logistic Regression uses its own Pipeline.
- [ ] The primary capacity is 20%; 10% and 30% are sensitivity scenarios, not separate tuned targets.
- [ ] The retention-value measure is called a proxy and is not described as actual employee worth, replacement cost, or financial contribution.
- [ ] Proxy normalization parameters are fitted on development-training data only and then applied to validation/formal-holdout records.
- [ ] The formal 20% holdout is unavailable to tuning, calibration choice, threshold selection, proxy-definition selection, capacity selection, or strategy selection after repository initialization.
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
- CatBoost, EBM, and FT-Transformer are the approved required advanced models. DummyClassifier is mandatory and Logistic Regression is optional. Do not add or substitute Random Forest, Decision Tree, RBF-SVM, XGBoost, LightGBM, TabNet, TabPFN, stacking, additional targets, deployment components, causal intervention models, or monetary optimization engines without a new logged pre-holdout decision.
- A target pivot after viewing formal holdout results is prohibited. A justified pivot before formal holdout evaluation requires a logged group or lecturer decision.
- Any change to the 20% primary capacity, value-proxy definition, ranking strategies, or evaluation metrics must be chosen from development evidence only and logged before formal-holdout access.
- If the Word companion is missing, outdated, or inconsistent, continue using this Markdown file and report that regeneration is required.

## 0.10 Mandatory AI startup acknowledgement

Before performing project-dependent work, an AI must provide a compact **Context acknowledgement** containing:

1. project objective and the two-stage build;
2. current phase and immediate milestone;
3. target column and positive class;
4. active model registry, pair ownership, model-specific representations, and primary prediction metric;
5. primary prioritization capacity, strategies, proxy definition, and ranking metrics;
6. formal holdout restriction and the historical preliminary-audit exposure caveat;
7. principal provenance or data-quality limitation;
8. prohibited real-world operational use and absence of intervention-effect evidence;
9. unresolved placeholders relevant to the requested task;
10. files or evidence currently available; and
11. the next action it will perform.

This acknowledgement is a parsing check, not a request for the user to repeat decisions. Do not ask the user to reconfirm a **DECIDED** item unless higher-authority evidence conflicts with it.

## 0.11 Stable context, live state, and duplicate-input handling

- Sections 0, 2, 4-9 are stable project context unless changed through the decision protocol.
- Section 1 `live_state`, Section 3, and the dated status/result/AI-use/change records are live information.
- A newer dated **LIVE RECORD** overrides an older status statement, but it cannot silently override a formal requirement or **DECIDED** item.
- Every live update must include status, `last_updated`, updater/evidence, and owner when known.
- When both the Markdown and Word files are supplied, read the Markdown as canonical and use Word as a synchronized fallback. Duplicate text is not independent evidence.
- The two files are sufficient for conceptual project understanding. Data-dependent execution still requires the exact CSV; implementation-dependent work must inspect the canonical repository and the specific files relevant to the task.

# 1. Machine-readable project memory

```yaml
document_control:
  context_schema_version: '4.2'
  project_context_version: 4.2.0
  synchronization_id: ERPC-20260806-V4.2.0
  generated_at: 2026-08-06 Asia/Colombo
  canonical_file: Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.md
  word_companion_file: Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.docx
  substantive_content_rule: Both files contain the same project information. Markdown is canonical; Word is a synchronized
    rendering/fallback.
  duplicate_input_rule: When both files are supplied, read Markdown first, treat repeated text as one source, and
    do not reconcile formatting-only differences.
  stable_context_sections:
  - '0'
  - '2'
  - '4'
  - '5'
  - '6'
  - '7'
  - '8'
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
  - 9 embedded formal source transcripts
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
    - shared split/fold registry and model-specific preprocessing adapters
    - DummyClassifier non-learning baselines and an optional Logistic Regression reference benchmark
    - CatBoost Classifier advanced predictive pipeline with native categorical handling
    - Explainable Boosting Machine advanced inherently interpretable pipeline with explicit feature types
    - FT-Transformer advanced deep tabular pipeline using fold-local numerical normalization, categorical integer
      encoding, embeddings, and PyTorch training controls
    - hyperparameter tuning within five-fold stratified development cross-validation
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
  current_phase: implementation planning and repository synchronization after the user-approved three-model, three-pair
    decision
  current_milestone: commit synchronized v4.2.0 context, update project artefacts to CatBoost/EBM/FT-Transformer,
    and implement the shared split, model adapters, and pair ownership protocol
  last_updated: '2026-08-06'
  updated_by: ChatGPT following the user-approved CatBoost, EBM, and FT-Transformer model decision
  evidence:
  - SRC-001
  - SRC-002
  - SRC-004
  - SRC-006
  - SRC-007
  - SRC-008
  - SRC-010
  - SRC-011
  - SRC-012
  completed:
  - formal requirement extraction
  - dataset audit and data contract
  - initial methodology and scope decisions
  - canonical GitHub repository URL recorded
  - 01_data_audit notebook created with reusable CSV outputs and target-distribution figure
  - project scope expanded to include resource-constrained retention-prioritization simulation
  - CatBoost, Explainable Boosting Machine, and FT-Transformer selected as the three required advanced models
  - DummyClassifier retained as the mandatory no-skill reference; Logistic Regression retained only as an optional
    lightweight benchmark
  - three two-member model pairs and cross-review responsibilities approved
  - context v4.2.0 generated with synchronized three-model implementation, tuning, evaluation, artefact, report,
    and viva rules
  active_next_actions:
  - replace repository context files with synchronized v4.2.0 Markdown and Word copies
  - update proposal slides, scripts, report plan, README, notebooks, and configuration files to the CatBoost/EBM/FT-Transformer
    registry
  - 'implement the shared split registry and four adapters: Dummy/optional Logistic, CatBoost, EBM, and FT-Transformer'
  - complete and peer-review the formal data audit and EDA notebooks
  - smoke-test CatBoost, InterpretML, PyTorch, and rtdl_revisiting_models in the chosen environment before formal
    experiments
  blockers:
  - team names and final ownership mapping are not supplied in the context
  - real intervention costs, outcomes, replacement costs, and business value are unavailable; monetary optimization
    claims remain blocked
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
  authority: explicit user decision to use the resignation-prediction title and regenerate the two-file AI handoff
- id: SRC-008
  name: User-approved retention-prioritization scope expansion
  date: '2026-08-01'
  authority: explicit user decision to extend the project while retaining Resigned as the target
- id: SRC-010
  name: User-approved advanced-model and AI-handoff update
  date: '2026-08-05'
  authority: explicit user decision
  statement: Use Dummy and Logistic Regression as references, CatBoost and EBM as required advanced models, and
    update both synchronized AI context files.
  status: SUPERSEDED BY SRC-012
- id: SRC-011
  name: User-supplied canonical repository and v4.1.1 cleanup instruction
  date: '2026-08-06'
  authority: explicit user decision
  statement: Record the GitHub repository, remove administrative and generated-output items from unresolved handoff
    requirements, correct synchronization defects, and update both files.
- id: SRC-012
  name: User-approved three-model and three-pair methodology update
  date: '2026-08-06'
  authority: explicit user decision
  statement: Use CatBoost, Explainable Boosting Machine, and FT-Transformer as the three required advanced models,
    one per two-member pair; retain DummyClassifier as the shared mandatory baseline and Logistic Regression only
    as an optional lightweight benchmark; update both synchronized AI context files.
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
    model_policy: categorical predictor; optional one-hot for Logistic Regression; native categorical for CatBoost;
      nominal feature type for EBM; fold-fitted integer vocabulary with missing/unknown codes and embeddings for
      FT-Transformer
  - name: Gender
    raw_dtype: string/category
    nullable: false
    role: sensitive predictor and audit attribute
    model_policy: sensitive categorical predictor and audit attribute; optional one-hot for Logistic Regression;
      native categorical for CatBoost; nominal feature type for EBM; fold-fitted integer vocabulary with missing/unknown
      codes and embeddings for FT-Transformer; exclude in no-Gender sensitivity model
  - name: Age
    raw_dtype: integer
    nullable: false
    role: sensitive predictor and audit attribute
    model_policy: numeric sensitive predictor and audit attribute; optional fold-fitted standardization for Logistic
      Regression; native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer;
      exclude in no-Gender/no-Age sensitivity model
  - name: Job_Title
    raw_dtype: string/category
    nullable: false
    role: predictor
    model_policy: categorical predictor; optional one-hot for Logistic Regression; native categorical for CatBoost;
      nominal feature type for EBM; fold-fitted integer vocabulary with missing/unknown codes and embeddings for
      FT-Transformer
  - name: Hire_Date
    raw_dtype: date-time text
    nullable: false
    role: audit field
    model_policy: parse for audit; exclude raw value from every primary model
  - name: Years_At_Company
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Education_Level
    raw_dtype: string/category
    nullable: false
    role: predictor
    model_policy: categorical predictor; optional one-hot for Logistic Regression; native categorical for CatBoost;
      nominal feature type for EBM; fold-fitted integer vocabulary with missing/unknown codes and embeddings for
      FT-Transformer
  - name: Performance_Score
    raw_dtype: integer/ordinal
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Monthly_Salary
    raw_dtype: integer/numeric
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Work_Hours_Per_Week
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Projects_Handled
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Overtime_Hours
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Sick_Days
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Remote_Work_Frequency
    raw_dtype: integer percentage
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Team_Size
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Training_Hours
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Promotions
    raw_dtype: integer count
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
  - name: Employee_Satisfaction_Score
    raw_dtype: float/numeric
    nullable: false
    role: predictor
    model_policy: retain as numeric/ordinal as documented; optional fold-fitted standardization for Logistic Regression;
      native numeric for CatBoost and EBM; fold-fitted imputation and standardization for FT-Transformer
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
  prohibited_claim: Do not call the formal holdout completely untouched from all prior exploratory modelling.
methodology:
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
    scope: development training partition
    role: primary model comparison and tuning evidence
    inner_validation_for_iterative_models: Within each outer development-CV training fold, create a training-only
      stratified validation subset for early stopping and epoch selection. Never use the outer validation fold or
      formal holdout to control training.
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
    optional_lightweight_benchmark:
      model: Logistic Regression
      role: transparent conventional reference only; not a required advanced model and not owned as a full two-member
        model workstream
      execution_rule: Run only if time permits after the shared evaluator is ready; do not allow it to delay the
        three required advanced models.
    required_advanced_models:
    - model: CatBoost Classifier
      role: advanced gradient-boosted predictive model with native categorical handling
    - model: Explainable Boosting Machine
      implementation: interpret.glassbox.ExplainableBoostingClassifier
      role: advanced inherently interpretable nonlinear model
    - model: FT-Transformer
      implementation: rtdl_revisiting_models.FTTransformer with PyTorch
      role: advanced attention-based deep tabular model using numerical and categorical feature tokens
    superseded_core_models:
    - Random Forest Classifier
    - Decision Tree Classifier
    - RBF-SVM
    contingency_only_model:
      model: XGBoost
      status: not active
      trigger: Only if FT-Transformer cannot be made reproducible in the tested environment after a documented smoke-test
        and bounded debugging attempt.
      decision_rule: Requires an explicit logged pre-holdout group/user decision before substitution.
    addition_rule: No additional model family, stacking, voting, or selected-model ensemble without a logged pre-holdout
      decision.
  model_specific_preprocessing:
    shared:
    - exclude Employee_ID and raw Hire_Date from primary modelling
    - preserve identical row order, target mapping, development/holdout membership, and outer fold IDs
    - fit every imputer, encoder, scaler, binning rule, category vocabulary, and proxy normalizer only on the applicable
      training data
    - apply no target-derived preprocessing outside training folds
    optional_logistic_regression:
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
    status: optional benchmark only
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
  model_selection_rule: Rank the three required advanced models by mean development-CV Average Precision using identical
    outer folds. Inspect fold stability, calibration, runtime, training failures, and explanation quality. Use a
    practical-equivalence rule rather than choosing a numerically higher score automatically. When performance is
    practically equivalent, prefer the model with stronger reproducibility, lower operational complexity, and clearer
    valid interpretation. Freeze the complete model, preprocessing, calibration source, and threshold before formal
    holdout use.
  threshold_policy:
    reference_threshold: 0.5
    secondary_threshold: threshold maximizing F1 on pooled out-of-fold development predictions
    restriction: never select threshold from formal holdout
    reporting: report both thresholds and full precision/recall trade-off
  uncertainty:
    cv: mean, standard deviation, and fold values
    formal_holdout: 95% percentile bootstrap interval with 2000 stratified resamples and seed 42 when metric is
      defined
  calibration_rule: Generate out-of-fold probabilities for CatBoost, EBM, and FT-Transformer; include optional Logistic
    Regression only if executed. Compare Brier score and reliability curves. If development evidence shows material
    miscalibration, compare development-only sigmoid or isotonic calibration where sample support is adequate. Because
    class weighting can alter probability calibration, assess weighted and unweighted probability behaviour explicitly.
    Freeze the probability source before formal holdout access.
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
      optimizer: AdamW using model.make_parameter_groups() or the package default optimizer
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
  packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - catboost
  - interpret
  - torch
  - rtdl_revisiting_models
  - jupyter
  - joblib
  - pytest
  - git
  dependency_specification: Maintain and smoke-test a compatible requirements.txt, environment.yml, or equivalent
    repository specification. Do not hard-code package versions in this context file; record exact tested versions
    in the execution environment artefact.
  compatibility_rule: Record the tested environment identifier with formal execution and verify imports, CPU/GPU
    device selection, fitting, probability output, explanation, checkpoint reload, and serialization.
  serialization_rule: 'Save family-appropriate artefacts: CatBoost native model plus metadata; verified joblib/pickle-compatible
    EBM artefact; FT-Transformer PyTorch state_dict plus architecture configuration, feature vocabularies, scaler
    state, selected epoch, and device-independent reload test; optional Logistic Regression sklearn pipeline when
    executed.'
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
- name: Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.md
  format: Markdown
  status: canonical project context
- name: Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.docx
  format: DOCX
  status: synchronized companion with same substantive content
fixed_decisions:
- Primary target remains Resigned with True as the positive class.
- 'The project has two linked stages: resignation-risk modelling and constrained retention-prioritization simulation.'
- The three required advanced models are CatBoost Classifier, Explainable Boosting Machine, and FT-Transformer.
- 'The six members work as three two-member model pairs: Pair 1 CatBoost, Pair 2 EBM, and Pair 3 FT-Transformer,
  each with additional rubric ownership and cross-review obligations.'
- DummyClassifier prior and most_frequent strategies are mandatory shared no-skill references.
- Logistic Regression is optional as a lightweight conventional benchmark; it is not a required advanced model and
  must not displace work on the three selected models.
- Random Forest, Decision Tree, and RBF-SVM are superseded as core models; XGBoost is contingency-only and requires
  a logged pre-holdout substitution decision.
- Each model uses its own appropriate preprocessing representation while sharing the same source rows, target mapping,
  split, outer folds, evaluation metrics, and holdout restrictions.
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
- Calibration must be evaluated because probabilities are used in the priority score.
- A valid negative result is acceptable and must be reported honestly.
- ChatGPT usage is disclosed and maintained as a live record.
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
- Do not add XGBoost, LightGBM, TabNet, TabPFN, stacking, or another model family without a logged pre-holdout decision.
- Do not assume FT-Transformer is the winner because it is deep learning or uses attention.
- Do not interpret raw attention weights as causal explanations or validated feature importance.
startup_acknowledgement:
  required_before_project_dependent_work: true
  fields:
  - project objective and two-stage scope
  - current phase
  - target and positive class
  - active model registry, pair roles, model-specific representations, and primary prediction metric
  - three model-pair ownership map
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

**Required advanced models:**

- CatBoost Classifier - performance-oriented gradient boosting with native categorical handling;
- Explainable Boosting Machine (`ExplainableBoostingClassifier`) - nonlinear, inherently interpretable generalized additive boosting;
- FT-Transformer (`rtdl_revisiting_models.FTTransformer`) - attention-based deep learning for mixed tabular inputs.

**References:** `DummyClassifier(strategy="prior")` and `DummyClassifier(strategy="most_frequent")` are mandatory no-skill baselines. Logistic Regression is optional as a lightweight conventional benchmark; it is not one of the three headline models and must not delay them.

**Pair ownership:** Members 1-2 own CatBoost plus shared data/split foundations. Members 3-4 own EBM plus EDA and interpretation. Members 5-6 own FT-Transformer plus shared evaluation, calibration, prioritization, and deep-model artefacts. All pairs use the same saved rows and outer folds and must cross-review another pair.

**Why this set:** The comparison spans three materially different model families: gradient-boosted trees, interpretable additive boosting, and attention-based deep tabular learning. The evaluator-facing question is not only which model scores highest, but what performance, calibration, stability, interpretability, and computational trade-offs each family creates.

**Fixed evaluation:** Stratified 80/20 development/formal-holdout split with seed 42; five-fold stratified CV on the development partition; PR-AUC/Average Precision as the primary prediction metric; calibration assessment for every advanced model; one-time formal-holdout corroboration after the complete design is frozen.

**Model-specific preprocessing:** CatBoost receives named categorical columns directly. EBM receives preserved names with explicit nominal/continuous types. FT-Transformer receives fold-fitted standardized continuous tensors and fold-fitted categorical integer codes with missing/unknown tokens and cardinalities. No model reuses another model's transformed matrix.

**Prioritization:** Primary capacity is 20%; 10% and 30% are fixed sensitivities. Compare repeated random selection, risk-only, value-proxy-only, risk-times-value-proxy, and threshold-reference strategies. The retention-value score is a transparent proxy, not actual employee worth.

**Critical limitation:** Preliminary audit-only evidence suggests the target may contain little predictive signal. Complex models cannot create absent information. A rigorous no-signal result remains valid and must prevent operational claims.

**Current phase:** Synchronize v4.2.0, update repository artefacts to the three-model registry, implement the locked split and model adapters, then complete development-only experiments. Canonical repository: `https://github.com/Sankalpa-KMCP/Employee_Resignation_Prediction`.

## 2.2 Critical terminology

| Term | Project-specific meaning |
| --- | --- |
| Positive class | `Resigned=True`. |
| Required advanced model | One of CatBoost, EBM, or FT-Transformer. Each is a headline comparative model and pair-owned workstream. |
| Shared baseline | Mandatory DummyClassifier reference evaluated under the same row/fold protocol but not presented as an advanced solution. |
| Optional benchmark | Logistic Regression, used only if time permits as a transparent conventional reference. |
| Outer fold | One of five StratifiedKFold development-CV folds used for unbiased model comparison. |
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

**LIVE RECORD - 2026-08-06:** CatBoost, EBM, and FT-Transformer are the three required advanced models, one per two-member pair. DummyClassifier baselines are mandatory; Logistic Regression is optional. The current milestone is to commit the synchronized v4.2.0 handoff, update repository artefacts, and implement the shared split plus model-specific adapters.

Canonical repository: `https://github.com/Sankalpa-KMCP/Employee_Resignation_Prediction`.

## 3.2 Immediate objectives

1. Replace repository context files with the synchronized v4.2.0 Markdown and Word copies.
2. Update README, slides, notebooks, report plan, model registry, and search-space configuration.
3. Implement one saved stratified development/holdout split and five saved outer folds.
4. Implement CatBoost, EBM, FT-Transformer, Dummy, and optional Logistic adapters without sharing incompatible transformed matrices.
5. Smoke-test imports, fitting, probability output, checkpoint reload, and CPU/GPU device selection.
6. Complete formal EDA and peer review.
7. Run development-only model comparison, calibration, prioritization, explanation, subgroup, and robustness analyses.

## 3.3 Completed work

- **VERIFIED** - Formal assignment requirements, dataset identity, schema, target distribution, and audit limitations are recorded.
- **COMPLETED** - Canonical GitHub repository URL is recorded.
- **IN PROGRESS / FUNCTIONAL** - `01_data_audit.ipynb` and reusable audit outputs exist; fresh-kernel and peer-review checks remain required.
- **DECIDED 2026-08-01** - Project includes constrained retention-prioritization simulation after resignation-risk modelling.
- **SUPERSEDED 2026-08-05** - Earlier CatBoost-plus-EBM-only core registry.
- **DECIDED 2026-08-06** - CatBoost, EBM, and FT-Transformer are required advanced models; Dummy is mandatory; Logistic Regression is optional; work is divided into three two-member pairs.
- **COMPLETED 2026-08-06** - Context v4.2.0 synchronizes the three-model methodology, pair ownership, preprocessing, tuning, evaluation, artefacts, report, and viva rules.

## 3.4 Work that may proceed now

- commit v4.2.0 and update active-model references;
- finish audit and EDA notebooks;
- build split/fold controls and model adapters;
- implement bounded development-CV searches and training-only early stopping;
- implement calibration, interpretability, ranking, fairness, and sensitivity modules;
- draft evidence-backed report/presentation content;
- prepare pair-specific and whole-project viva explanations.

## 3.5 Blocked or conditional work

- genuine monetary budget optimization is blocked because intervention costs, effects, replacement costs, and verified business-value data are absent;
- causal claims that a retention action prevents resignation are blocked;
- production use, employment decisions, legal-compliance claims, and fairness certification are outside scope;
- FT-Transformer GPU use is conditional on the tested environment; CPU execution remains a valid fallback;
- XGBoost substitution is blocked unless a logged pre-holdout decision formally replaces FT-Transformer;
- final numerical findings remain conditional on executing the approved workflow and must not be invented.

## 3.6 Work not to prioritize now

- dashboard, GUI, website, API, database, cloud deployment, or mobile application;
- model stacking, voting, foundation models, or extra model families;
- unrestricted hyperparameter search;
- synthetic provenance narratives or assumed salary currency;
- automated record-level employment actions.

## 3.7 Current risks

| ID | Risk | Likelihood | Impact | Active mitigation |
| --- | --- | --- | --- | --- |
| R-001 | Target may contain little usable signal. | High | High | Dummy baselines, PR-AUC, uncertainty, no-signal rule. |
| R-002 | Historical preliminary audit exposed aggregate holdout-related diagnostics. | Certain | Medium | Freeze protocol; preserve exposure caveat; use holdout only as corroboration. |
| R-003 | Salary determinism may dominate proxy or model behaviour. | High | High | No-salary sensitivity and explicit redundancy analysis. |
| R-004 | Deep-model complexity may not produce a meaningful gain. | Medium | Medium | Fair folds, bounded search, runtime reporting, practical-equivalence rule. |
| R-005 | Model-specific adapters may create row/fold parity errors. | Medium | High | Shared registry, row-ID assertions, adapter tests, prediction-coverage tests. |
| R-006 | FT-Transformer category vocabularies or early stopping may leak validation information. | Medium | High | Fit vocabularies/scalers on outer training only; create inner validation only inside training. |
| R-007 | PyTorch results may vary by device/seed. | Medium | Medium | Seed controls, deterministic settings where feasible, device metadata, additional-seed sensitivity for selected configuration. |
| R-008 | Probabilities may be poorly calibrated, especially under class weighting. | Medium | High | OOF reliability curves, Brier score, development-only calibration comparison. |
| R-009 | Pair work may drift into inconsistent pipelines. | Medium | High | Shared contracts, weekly integration, cross-pair review, single evaluator. |
| R-010 | AI-generated code or explanations may be accepted without ownership. | Medium | High | Tests, source verification, manual review, AI-use records, viva preparation. |

## 3.8 Next concrete actions

1. Commit v4.2.0 context files.
2. Update `model_registry.yaml`, `search_spaces.yaml`, README, and notebooks.
3. Finalize the data contract, saved split, outer folds, and row-parity tests.
4. Implement and smoke-test CatBoost, EBM, and FT-Transformer minimal pipelines.
5. Freeze the bounded search spaces before formal model comparison.
6. Generate OOF probabilities and common comparison tables.
7. Add calibration and prioritization only after OOF coverage tests pass.

# 4. Task router for AI assistants

| Future request | Must read | Must verify | May generate | Must not invent |
| --- | --- | --- | --- | --- |
| Understand the project | Sections 0-3 and 8.1-8.10 | Version, two-stage scope, three-model registry, pair ownership, status | Concise status-aware summary | New objectives, dates, or results |
| Inspect data | Sections 5, 8.10-8.15, 8.18-8.19 | CSV checksum, shape, schema | Audit code, tables, EDA plan | Units, provenance, missingness, semantics |
| Generate preprocessing code | Sections 8.17-8.19 | Feature policy, fold boundaries, model adapter | Leakage-safe transformations and tests | Full-data fitting or one shared model matrix |
| Generate CatBoost code | Sections 8.18-8.22 | Native categorical contract, folds, bounded search | CatBoost pipeline, tuning, OOF predictions | One-hot input or holdout early stopping |
| Generate EBM code | Sections 8.18-8.23 | Feature names/types, interaction limits | EBM pipeline and explanations | Reusing CatBoost/FT encodings or causal claims |
| Generate FT-Transformer code | Sections 8.18-8.23 | Continuous/categorical tensors, vocabularies, inner validation, device | PyTorch training loop, checkpointing, OOF predictions | Holdout early stopping, untracked category codes, attention-as-causality |
| Generate baseline code | Sections 8.20-8.22 | Shared rows/folds and baseline role | Dummy baselines; optional Logistic benchmark | Presenting a baseline as a required advanced model |
| Generate prioritization code | Sections 8.16-8.24 | Frozen capacity, proxy, calibration, strategies, ranking metrics | Random and top-k comparisons | Actual worth, intervention benefit, optimal-budget claims |
| Tune models | Section 8.21 | Search caps, seed, scoring, early-stopping boundaries | Bounded search code and result schema | Formal-holdout tuning or unlimited search |
| Evaluate results | Sections 8.22-8.25 and 8.33 | OOF coverage, saved predictions, identical rows/folds | Prediction, calibration, ranking, uncertainty, error, fairness tables | A winner from unstable differences |
| Write report/presentation | Sections 8.6-8.9 and relevant evidence sections | Formal structure and actual generated results | Evidence-backed prose and figures | Fabricated findings or omitted limitations |
| Prepare viva | Sections 8.2, 8.13, 8.17-8.25, 8.30, 8.33 | Approved decisions and actual outputs | Pair and whole-project questions/answers | Results not generated |
| Debug code | Sections 5, 8.17-8.18, 8.25 and repository | Environment, paths, split, failing tests | Minimal fix with regression test | Method changes that conceal failures |
| Update status | Sections 3, 7, 8.31-8.36 | Evidence | Status, risk, AI-use, decision records | Unverified completion |
| Recommend scope change | Sections 0.9, 6, 8.8, 8.36 | Trigger, cost, authority | Decision proposal | Automatic unlogged implementation |
| Handle missing file | Sections 5 and 7 | Manifest, checksum, exact filename | Recovery steps | Substituting a different file |

# 5. Companion-file manifest and handoff package

## 5.1 Required files for project-dependent work

| ID | Canonical filename or path | Required for | Integrity rule | Status |
| --- | --- | --- | --- | --- |
| CTX-MD | `Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.md` | All project-dependent work | Canonical; version/date must match. | VERIFIED generated |
| CTX-DOCX | `Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.docx` | Synchronized fallback/navigation | Regenerate from CTX-MD after substantive changes. | VERIFIED generated |
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
├── requirements.txt
├── environment-lock.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── Extended_Employee_Performance_and_Productivity_Data.csv
│   └── processed/
├── configs/
│   ├── split_config.yaml
│   ├── model_registry.yaml
│   ├── search_spaces.yaml
│   ├── training_config.yaml
│   └── prioritization_config.yaml
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_prioritization.ipynb
├── src/
│   ├── data_validation.py
│   ├── split_registry.py
│   ├── feature_contract.py
│   ├── model_adapters.py
│   ├── baselines.py
│   ├── catboost_model.py
│   ├── ebm_model.py
│   ├── ft_transformer_model.py
│   ├── torch_training.py
│   ├── calibrate.py
│   ├── explain.py
│   ├── prioritize.py
│   ├── evaluate.py
│   └── fairness.py
├── tests/
│   ├── test_data_contract.py
│   ├── test_split_registry.py
│   ├── test_model_adapters.py
│   ├── test_ft_transformer_vocab.py
│   ├── test_oof_coverage.py
│   ├── test_calibration.py
│   ├── test_prioritization.py
│   └── test_serialization.py
├── models/
│   ├── catboost/
│   ├── ebm/
│   ├── ft_transformer/
│   └── optional_logistic_regression/
├── results/
│   ├── predictions/
│   ├── metrics/
│   ├── calibration/
│   ├── training_curves/
│   ├── explanations/
│   └── prioritization/
├── reports/
├── presentations/
└── docs/
    └── context/
        ├── Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.md
        └── Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.docx
```

## 5.4 Missing or mismatched file protocol

1. Stop data-dependent work if the exact CSV is absent or its checksum differs.
2. Never substitute another copy automatically.
3. Compare filename, archive contents, shape, schema, and checksum.
4. Record a new source ID and decision if the assigned artefact officially changes.
5. Continue from Markdown if DOCX is missing and regenerate it.
6. Treat the package as incomplete if Markdown is missing.
7. Treat an untested FT-Transformer package/API as an environment issue, not permission to silently change models.

## 5.5 Synchronization rule

- Markdown is authoritative.
- Make material changes in Markdown first.
- Regenerate and visually inspect Word after every material change.
- Both files must show the same version, model registry, pair ownership, state, placeholders, and references.

## 5.6 Handoff package levels

**Conceptual handoff:** These two synchronized files are sufficient for ordinary project understanding and planning.

**Data-execution handoff:** Add the exact raw CSV or original archive and verify its checksum.

**Implementation handoff:** Inspect the canonical GitHub repository, exact tested environment, and generated artefacts required by the task.

**Formal-source verification:** Use the original PDFs when exact wording, layout, or a newer revision matters. Section 9 contains normalized project-relevant extracts, not complete verbatim transcripts.

# 6. Dependency map

## 6.1 Technical and procedural dependencies

```text
Formal requirements
    -> approved two-stage framing
        -> target and feature policy
            -> exact dataset and checksum
                -> saved development/holdout split
                    -> saved five outer folds
                        -> model-specific fold-local adapters
                            -> Dummy references and bounded CatBoost/EBM/FT-Transformer training
                                -> OOF probabilities and calibration assessment
                                    -> model/preprocessing/calibration freeze
                                        -> proxy normalization and prioritization strategy freeze
                                            -> one-time formal-holdout corroboration
                                                -> error, explanation, subgroup, and robustness analysis
                                                    -> report, presentation, and viva

FT-Transformer
    -> tested PyTorch + rtdl_revisiting_models environment
        -> device selection and reproducible seeds
            -> fold-local categorical vocabularies and continuous scaler
                -> training-only inner validation
                    -> checkpoint and reload tests

Usable resignation-risk signal
    -> meaningful risk-based prioritization
        -> comparison with repeated random, value-only, and risk-only strategies

Actual intervention cost/effect evidence
    -> genuine budget optimization
        -> NOT AVAILABLE; blocked and outside scope

Actual AI assistance
    -> live usage records
        -> final AI Tool Usage Declaration
```

## 6.2 Blocking table

| Dependency | Blocks | Current status | Continue? |
| --- | --- | --- | --- |
| Exact assigned CSV/checksum | Data-dependent work | VERIFIED available | Yes |
| Approved target/framing | Methodology | DECIDED | Yes |
| Split/fold protocol | Formal modelling | DECIDED; implementation pending | Yes, implement first |
| Tested CatBoost/InterpretML/PyTorch/RTDL environment | Model execution | TO BE GENERATED | Yes, smoke-test before experiments |
| Canonical repository | Collaboration | URL recorded | Yes |
| Named team map | Personal accountability | PLACEHOLDER | Yes, numbered pair roles active |
| Real intervention/business-value evidence | Genuine optimization/causal claims | NOT AVAILABLE | No; retain simulation-only framing |

# 7. Update protocol and handoff rules

## 7.1 Required update sequence

For every material change:

1. Increment semantic version: patch for corrections; minor for methodology/schema/team-workflow changes; major for target/dataset/scope changes.
2. Update date, synchronization ID, filenames, and document-control table.
3. Update machine-readable memory.
4. Update Section 3 live state.
5. Update model registry, pair map, decision records, risks, and placeholders.
6. Update methodology, result templates, references, and AI-use records.
7. Add a change-log entry with evidence and approver.
8. Regenerate Word from Markdown.
9. Render and inspect every DOCX page.
10. Run content-consistency searches and compare version/model references across both files.

## 7.2 Result update rule

A result becomes **VERIFIED** only when its data checksum, split/fold ID, code commit, environment, model configuration, seed/device, and output artefact are identifiable. Unsaved notebook cells, slides, conversations, and preliminary audit diagnostics are insufficient.

## 7.3 Decision update rule

Record decision ID, date, previous value, new value, rationale, authority, affected sections, approver, and reconsideration condition. Never silently substitute a model or expose the formal holdout.

## 7.4 Reusable handoff instruction

> Read `Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.md` first and treat it as canonical. The DOCX is a synchronized fallback. The required advanced models are CatBoost, EBM, and FT-Transformer, one per two-member pair. DummyClassifier baselines are mandatory; Logistic Regression is optional. Use identical saved rows and outer folds but model-specific fold-local representations. Fit FT-Transformer category vocabularies/scalers and any inner early-stopping validation only inside each outer training fold. Verify the exact CSV checksum before data work and inspect the current GitHub files before code-dependent claims. Never invent results, never tune or early-stop on the formal holdout, and update live state, AI-use, decisions, version, and Word after material changes.

# 8. Full canonical project specification

## 8.1 Document control

| Field | Value | Status |
| --- | --- | --- |
| Context version | 4.2.0 | ACTIVE |
| Date | 2026-08-06 | VERIFIED |
| Canonical file | `Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.md` | DECIDED |
| Word companion | `Employee_Resignation_Prediction_AI_Project_Context_v4_2_0.docx` | DECIDED |
| Canonical repository | `https://github.com/Sankalpa-KMCP/Employee_Resignation_Prediction` | DECIDED |
| Current phase | Three-model implementation planning and repository synchronization | LIVE RECORD |
| Required advanced models | CatBoost, EBM, FT-Transformer | DECIDED |
| Mandatory baselines | DummyClassifier `prior` and `most_frequent` | DECIDED |
| Optional benchmark | Logistic Regression | DECIDED |
| Pair ownership | Pair 1 CatBoost; Pair 2 EBM; Pair 3 FT-Transformer | DECIDED |
| Superseded core models | Random Forest, Decision Tree, RBF-SVM | SUPERSEDED |
| Primary metric | Average Precision / PR-AUC | DECIDED |
| Primary capacity | Top 20% | DECIDED |

## 8.2 Executive project summary

The group is designing, implementing, and critically evaluating a two-stage AI/ML study using the assigned employee dataset.

1. **Prediction stage:** determine whether available employee-related features provide reproducible out-of-sample signal for `Resigned=True`.
2. **Prioritization stage:** use frozen, calibrated probability outputs in a controlled top-k simulation to compare how limited HR review/retention capacity might be allocated.

The three required advanced models are:

1. **CatBoost Classifier** - gradient-boosted trees and native categorical handling;
2. **Explainable Boosting Machine** - nonlinear additive boosting with inherent global/local interpretability;
3. **FT-Transformer** - attention-based deep tabular learning with continuous and categorical feature tokens.

DummyClassifier provides mandatory no-skill references. Logistic Regression is optional as a conventional transparent benchmark and is not a headline advanced model.

The central evaluation is multidimensional: predictive ranking, threshold performance, calibration, stability, runtime, reproducibility, interpretation quality, subgroup behaviour, and prioritization lift. No model is assumed to win because of reputation or complexity.

All record-level prioritization is an academic simulation. The project does not establish causality, intervention effectiveness, actual employee value, financial optimality, legal fairness, or production readiness.

### 8.2.1 Exact assessed build

The assessed build is an end-to-end, reproducible Python workflow containing:

1. immutable raw-data identity and checksum checks;
2. data audit, cleaning decisions, and EDA;
3. saved 80/20 stratified development/formal-holdout split;
4. saved five outer development-CV folds;
5. model-specific leakage-safe adapters;
6. mandatory DummyClassifier references and optional Logistic benchmark;
7. bounded CatBoost, EBM, and FT-Transformer tuning;
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
- implement three advanced and structurally different supervised classifiers;
- use identical saved rows/folds and model-appropriate preprocessing;
- tune within bounded development-only protocols;
- compare prediction, calibration, stability, interpretability, and computational cost;
- simulate top-k prioritization using a clearly labelled retention-value proxy;
- analyse subgroup behaviour and ethical risks;
- produce an honest positive or negative conclusion supported by saved evidence.

### 8.4.2 Project questions

- PQ-01: Does any model consistently exceed the Dummy prior baseline in development PR-AUC?
- PQ-02: How do CatBoost, EBM, and FT-Transformer compare under identical outer folds?
- PQ-03: Does FT-Transformer's added training complexity create a stable gain over CatBoost or EBM?
- PQ-04: Does CatBoost's performance justify its lower inherent transparency than EBM?
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
| REQ-004 | Select and justify at least two suitable algorithms. | Three required advanced models: CatBoost, EBM, FT-Transformer. |
| REQ-005 | Implement in Python with tuning and pipelines. | Bounded searches, adapters, saved configuration, tests, and artefacts. |
| REQ-006 | Use metrics, k-fold CV, compare, justify best performer. | Five outer folds, PR-AUC primary, calibration/uncertainty/complexity comparison. |
| REQ-007 | Discuss ethics and bias mitigation. | Section 8.24 plus subgroup analysis. |
| REQ-008 | Prepare professional report, presentation, and AI-use declaration. | Sections 8.28-8.35. |

## 8.7 Assessment and acceptance criteria

| Assessment area | Evidence required |
| --- | --- |
| Problem framing and dataset understanding | Accurate source/target/feature/quality explanation and limitations |
| Preprocessing and EDA | Reproducible audit, leakage-safe transformations, meaningful visuals and insights |
| Model design and implementation | Justified CatBoost, EBM, FT-Transformer; bounded tuning; pipelines/adapters; reproducibility |
| Evaluation and comparison | Five-fold CV, relevant metrics, OOF coverage, calibration, uncertainty, runtime, justified final selection |
| Report and teamwork | Professional evidence-backed report, ethics, AI transparency, reflections, references, clear pair and individual ownership |

Acceptance is based on correctness and critical evaluation, not on obtaining a high score. A verified no-signal outcome is acceptable.

## 8.8 Scope boundaries

### In scope

- binary classification of `Resigned`;
- data audit, EDA, and leakage controls;
- CatBoost, EBM, FT-Transformer;
- Dummy references and optional Logistic benchmark;
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
| SRC-006 to SRC-011 | Prior dated conversation decisions | Historical authority for earlier approved changes. | Superseded items remain historical; newer logged decisions take precedence. |
| SRC-012 | Conversation decision on 2026-08-06 | Authority for the active three-model and three-pair registry. | CatBoost, EBM, FT-Transformer required; Dummy mandatory; Logistic optional. |

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
| Department | Categorical | 9 categories | Predictor | One-hot encode | May proxy organizational structure; all job-title combinations occur |
| Gender | Categorical | Female, Male, Other | Sensitive predictor/audit attribute | Use in full model; exclude in fairness sensitivity model; one-hot encode | Potential direct discrimination; subgroup sizes must be reported |
| Age | Integer | 22 to 60 | Sensitive predictor/audit attribute | Retain in full model; exclude in second fairness sensitivity model; scale for LR | Potential age discrimination; no causal interpretation |
| Job_Title | Categorical | 7 categories | Predictor | One-hot encode | Strong deterministic relationship with salary |
| Hire_Date | Date-time text | 2014-09-07 to 2024-09-03; identical time component | Audit field | Parse for audit; exclude raw value from primary model | Near-duplicate of tenure; generation artefact |
| Years_At_Company | Integer | 0 to 10 | Predictor | Retain as tenure representation | Near-redundant with Hire_Date; implied hire age can be implausible |
| Education_Level | Categorical | High School, Bachelor, Master, PhD | Predictor | One-hot encode | Do not assume equal spacing without a separate sensitivity test |
| Performance_Score | Integer/ordinal | 1 to 5 | Predictor | Retain numeric or ordinal; document representation | Salary is deterministically based partly on this field |
| Monthly_Salary | Numeric | 3,850 to 9,000; 28 distinct values | Predictor | Retain in primary model; unit unverified; optional redundancy sensitivity | Completely determined by Job_Title and Performance_Score |
| Work_Hours_Per_Week | Integer | 30 to 60 | Predictor | Retain; scale for LR | Bounded regular distribution; no automatic outlier removal |
| Projects_Handled | Integer | 0 to 49 | Predictor | Retain; scale for LR | Count semantics/time window undocumented |
| Overtime_Hours | Integer | 0 to 29 | Predictor | Retain; scale for LR | Time window and units undocumented |
| Sick_Days | Integer | 0 to 14 | Predictor | Retain; scale for LR | Time window undocumented; potentially sensitive workplace attribute |
| Remote_Work_Frequency | Integer percentage | 0, 25, 50, 75, 100 | Predictor | Retain numeric; optional categorical sensitivity | Meaning appears percentage but observation window undocumented |
| Team_Size | Integer | 1 to 19 | Predictor | Retain; scale for LR | Organizational context undocumented |
| Training_Hours | Integer | 0 to 99 | Predictor | Retain; scale for LR | Time window undocumented |
| Promotions | Integer count | 0 to 2 | Predictor | Retain; scale for LR or categorical sensitivity | Time window undocumented |
| Employee_Satisfaction_Score | Numeric | 1.00 to 5.00 | Predictor | Retain; scale for LR | Measurement instrument and timing undocumented |
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
- Unknown categorical values: fail the validation report for audit visibility; modelling pipelines use `OneHotEncoder(handle_unknown="ignore")` so later inference fails safely rather than structurally

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

Environment contract: target Python 3.11.x. The repository must maintain a tested dependency specification (`requirements.txt`, `environment.yml`, or equivalent) and record the environment used for formal execution.

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
| A-013 | CatBoost, EBM, and FT-Transformer require model-specific representations rather than one shared transformed matrix. | DECIDED | Same rows/folds; separate adapters and tests. |
| A-014 | CatBoost, InterpretML, PyTorch, and rtdl_revisiting_models APIs must be validated in the project environment before use. | VERIFIED implementation constraint | Maintain a tested dependency specification in the repository and run smoke tests. |
| C-001 | Final report maximum is 15 pages. | VERIFIED | Prioritize high-mark sections and concise visuals. |
| C-002 | At least two algorithms and k-fold CV are required. | VERIFIED | Use CatBoost, EBM, FT-Transformer, and five-fold CV; retain Dummy baselines and optional Logistic Regression. |
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
| CV | Five saved stratified outer folds on development data |
| Primary metric | Average Precision / PR-AUC |
| Required advanced models | CatBoost, EBM, FT-Transformer |
| Mandatory baselines | Dummy prior and most-frequent |
| Optional benchmark | Logistic Regression |
| Pair ownership | Pair 1 CatBoost; Pair 2 EBM; Pair 3 FT-Transformer |
| Early stopping | Training-only inner validation inside each outer training fold |
| Class imbalance | Model-appropriate weights/loss weighting plus threshold analysis; no pre-split resampling |
| Calibration | OOF Brier/reliability assessment; development-only calibration if justified |
| Primary capacity | Top 20%; fixed 10% and 30% sensitivity |
| Priority score | Frozen calibrated resignation probability × frozen retention-value proxy |
| Holdout | Used once only after model, preprocessing, calibration, proxy, capacity, strategy, and threshold freeze |
| Negative result | Valid and required if models remain near baseline |

The three advanced models are equally legitimate candidates. The project must not select CatBoost, EBM, or FT-Transformer in advance.

## 8.17 End-to-end workflow and experimental protocol

1. Verify exact CSV filename, SHA-256, shape, schema, and target orientation.
2. Preserve raw data unchanged.
3. Reproduce and peer-review the data audit.
4. Create a persistent row identifier for experimental tracking without using `Employee_ID` as a predictor.
5. Create and save the stratified 80/20 development/formal-holdout split.
6. Create and save five stratified outer folds within development data.
7. Freeze the feature contract and model-specific adapter contracts.
8. Perform EDA using development data; use target-aware plots carefully and avoid holdout-guided design.
9. Implement mandatory Dummy baselines.
10. Implement minimal smoke-test versions of CatBoost, EBM, and FT-Transformer.
11. Freeze bounded search spaces.
12. For every model and outer fold:
    - fit all preprocessing on outer training rows only;
    - create training-only inner validation when early stopping is required;
    - train/tune without using outer validation or formal holdout for training control;
    - predict the outer validation rows exactly once;
    - save fold configuration, runtime, seed/device, failures, and probability outputs.
13. Verify complete OOF coverage and identical row order across candidates.
14. Calculate prediction metrics, fold uncertainty, calibration, and runtime.
15. Select a model/preprocessing/calibration configuration using the development-only rule.
16. Freeze thresholds, proxy design, capacity scenarios, and ranking strategies.
17. Refit the selected configuration on the full development partition using only development-derived training controls.
18. Access the formal holdout once for corroborative prediction and prioritization evaluation.
19. Perform error, interpretation, subgroup, capacity, and proxy sensitivity analysis.
20. Save artefacts, update logs, write the report, and rehearse the viva.

### Experimental fairness rule

Every headline model receives a reasonable but bounded tuning opportunity. Search budgets do not need identical configuration counts, but they must be declared before results and must not be expanded merely because one model is underperforming.

## 8.18 Architecture, repository, data handling, and leakage controls

### 8.18.1 Component responsibilities

| Component | Responsibility | Required output |
| --- | --- | --- |
| `data_validation.py` | Verify checksum, schema, types, ranges, target orientation | validation report |
| `split_registry.py` | Save/reload split and five outer folds | row-level split/fold registry |
| `feature_contract.py` | Declare predictors, exclusions, types, sensitive fields | versioned feature contract |
| `model_adapters.py` | Shared adapter interface and row-parity assertions | adapter metadata |
| `baselines.py` | Dummy and optional Logistic | OOF reference predictions |
| `catboost_model.py` | Native categorical CatBoost training | model, parameters, OOF probabilities |
| `ebm_model.py` | Explicit-type EBM training/explanations | model, terms, OOF probabilities |
| `ft_transformer_model.py` | Architecture/configuration and tensor interface | model definition and metadata |
| `torch_training.py` | FT training, early stopping, checkpoints, device/seed controls | curves, best epoch, checkpoint |
| `calibrate.py` | Reliability, Brier, optional development-only calibration | calibration artefacts |
| `evaluate.py` | Common prediction metrics, uncertainty, comparison | comparison tables |
| `explain.py` | Family-valid explanations | explanation artefacts |
| `prioritize.py` | Proxy, top-k strategies, ranking metrics | prioritization results |
| `fairness.py` | Subgroup prediction/selection audit | subgroup tables |

### 8.18.2 Model-adapter contract

| Model | Input representation | Categorical handling | Numerical handling | Required exported metadata |
| --- | --- | --- | --- | --- |
| Dummy | Target-only training statistics or constant strategy | none | none | strategy, prevalence |
| Optional Logistic | sklearn matrix in Pipeline | fold-fitted one-hot encoding with unknown handling | fold-fitted scaling | feature names, encoder/scaler state, coefficients |
| CatBoost | pandas DataFrame or `Pool` | named native categorical columns | raw approved numerical values unless justified | categorical names/indices, parameters, tree count |
| EBM | pandas DataFrame or compatible array | explicit nominal types | explicit continuous types; no unnecessary scaling | feature names/types, bins, terms, interactions |
| FT-Transformer | separate continuous and categorical tensors | fold-fitted integer vocabularies; missing/unknown codes; embeddings | fold-fitted imputation and standardization | vocabularies, cardinalities, scaler/imputer state, architecture, epoch, seed/device |

All adapters must produce predictions indexed by the same saved row IDs.

### 8.18.3 Leakage and integrity controls

- No preprocessing object is fitted on the full dataset before outer CV.
- No category vocabulary may inspect an outer validation fold or formal holdout.
- No early-stopping metric may use an outer validation fold or formal holdout.
- No proxy normalizer may use the formal holdout.
- No threshold, calibration method, capacity, or strategy is chosen from formal-holdout performance.
- `Employee_ID` is for tracking only; raw `Hire_Date` is excluded from primary modelling.
- Holdout rows remain inaccessible to model-pair notebooks until the freeze gate.
- Every OOF prediction file must contain each development row exactly once per model.
- Any failure, skipped fold, NaN probability, or row mismatch invalidates comparison until corrected.

## 8.19 Preprocessing and EDA plan

### 8.19.1 Shared data preparation

- verify column names/types and Boolean target mapping;
- identify missing, duplicate, invalid, impossible, and construction-like values;
- preserve raw rows unless an exclusion is explicitly justified and logged;
- use the same predictor policy for all main models, apart from representation requirements;
- keep `Gender` and `Age` available for the primary predictive experiment but perform sensitive-feature ablations;
- exclude `Employee_ID` and raw `Hire_Date`;
- preserve `Years_At_Company`;
- fit every transformation only within the applicable training data.

### 8.19.2 Model-specific preparation

**CatBoost**

- preserve categorical strings and column names;
- pass `cat_features` names/indices directly;
- do not one-hot encode;
- compare no class weights versus balanced weights within bounded search;
- do not use formal holdout as `eval_set`.

**EBM**

- preserve feature names and declare nominal/continuous types;
- inspect feature-type routing before formal CV;
- bound main effects/interactions;
- report the final term list and any interaction pruning.

**FT-Transformer**

- continuous predictors: training-fold imputation and standardization;
- categorical predictors: training-fold vocabulary, explicit missing/unknown indices, exported cardinalities;
- tensors: continuous float, categorical `int64`, binary target float for BCE-with-logits;
- create a stratified inner validation split from outer training data only;
- use mini-batches, checkpointing, early stopping, and device-independent reload;
- do not one-hot encode categorical features.

**Optional Logistic Regression**

- use one-hot encoding and standardization inside a Pipeline;
- execute only after required shared infrastructure is complete.

### 8.19.3 EDA questions and figures

Minimum EDA evidence:

- target prevalence and imbalance;
- numerical distributions and suspicious ranges;
- categorical frequencies;
- target rate by major categorical/ordinal groups with uncertainty-aware interpretation;
- feature redundancy/correlation, especially salary/job-title/performance construction;
- tenure, age, weekly-hours, overtime, satisfaction, and sick-day relationships;
- missingness/duplicates/outliers;
- proxy-component distributions and salary-removal sensitivity;
- sensitive-group representation;
- clear distinction between descriptive association and causation.

## 8.20 Model and algorithm selection

| Model | Status | Why included | Main limitations |
| --- | --- | --- | --- |
| Dummy `prior` | Mandatory baseline | Establishes prevalence probability and PR-AUC reference | Not a learned solution |
| Dummy `most_frequent` | Mandatory baseline | Exposes misleading majority-class accuracy | No probability ranking |
| Logistic Regression | Optional benchmark | Transparent conventional linear reference | Basic; may miss nonlinearities; not pair-owned headline model |
| CatBoost | Required advanced | Strong mixed-tabular model, native categoricals, nonlinear interactions | Post-hoc interpretation, tuning/runtime, calibration risk |
| EBM | Required advanced | Nonlinear additive boosting with inspectable feature effects/interactions | Interaction growth can reduce clarity; API/environment testing |
| FT-Transformer | Required advanced | Distinct deep-learning family; numerical/categorical feature tokens and self-attention | Greater compute, seed/device variability, less inherent explanation |
| XGBoost | Contingency only | Reliable fallback if FT implementation is irreparably blocked | Not active; overlaps with CatBoost; requires decision |
| Random Forest, Decision Tree, RBF-SVM | Superseded | Historical options only | Not active |

### Model comparison research questions

1. Which model has the strongest and most stable development PR-AUC?
2. Are differences practically meaningful relative to fold variability?
3. Are probabilities calibrated enough for prioritization?
4. Does FT-Transformer justify its extra training complexity?
5. Does CatBoost justify lower inherent transparency than EBM?
6. Does the selected model improve top-k prioritization over repeated random selection?

## 8.21 Tuning and configuration plan

### 8.21.1 Global configuration

- outer comparison: five saved stratified folds;
- primary scoring: Average Precision;
- fixed base seed: 42;
- search spaces frozen before results;
- maximum active required candidates: CatBoost, EBM, FT-Transformer;
- no formal-holdout tuning;
- log configuration, fold, runtime, device, seed, failures, and output paths;
- use bounded candidate lists/random search rather than exhaustive Cartesian grids.

### 8.21.2 Optional Logistic Regression search

Maximum eight configurations. Candidate dimensions:

- penalty `l1`/`l2` with compatible solver;
- `C` in a compact logarithmic range;
- `class_weight` none/balanced.

This benchmark must not delay required models.

### 8.21.3 CatBoost search

Maximum twelve configurations. Candidate dimensions:

- iterations: 300, 600, 1000;
- depth: 4, 6, 8;
- learning rate: 0.03, 0.05, 0.1;
- L2 regularization: 3, 5, 10;
- random strength: 0, 1, 2;
- class weighting: none or balanced.

Use `Logloss`, seed 42, no uncontrolled file writing. Use fixed iterations or training-only inner validation; never outer validation/formal holdout for early stopping.

### 8.21.4 EBM search

Maximum ten configurations. Candidate dimensions:

- learning rate: 0.005, 0.015, 0.03;
- max leaves: 2 or 3;
- minimum samples per leaf: 2, 4, 10;
- interactions: 0, 5, 10;
- bounded maximum rounds and early-stopping patience.

Prefer fewer interactions under practical equivalence. Keep internal validation inside outer training.

### 8.21.5 FT-Transformer search

Maximum twelve staged configurations, not a full grid.

**Core architecture candidates**

- blocks: 2 or 3;
- token/block dimension: 64, 128, or 192;
- attention heads: 4 or 8, with divisibility enforced;
- attention dropout: 0.1 or 0.2;
- feed-forward dropout: 0.1 or 0.2.

**Training candidates**

- AdamW learning rate: 1e-4, 3e-4, or 1e-3;
- weight decay: 1e-6, 1e-5, or 1e-4;
- batch size: 256 or 512;
- positive-class loss weight: none or outer-training negative/positive ratio;
- maximum 100 epochs;
- early stopping patience 12 on inner-validation PR-AUC.

Use `rtdl_revisiting_models.FTTransformer`; use `model.make_parameter_groups()` or package default optimizer. Save the best epoch, training curves, architecture, vocabularies, scaler state, seed, and device. Run an additional-seed sensitivity for the selected configuration when feasible.

### 8.21.6 Threshold and calibration selection

- reference threshold: 0.5;
- secondary threshold: F1-maximizing threshold from pooled OOF development probabilities;
- report precision-recall trade-off rather than only one threshold;
- compare Brier score and reliability curves;
- test sigmoid/isotonic calibration only from development evidence;
- freeze raw versus calibrated probability source before formal holdout.

### 8.21.7 Retention-value proxy configuration

Primary proxy:

`V_i = (normalized Performance_i + normalized Salary_i + normalized Tenure_i) / 3`

Each tilde denotes development-fitted min-max normalization. Sensitivities:

- no-salary proxy;
- epsilon-adjusted normalized multiplicative proxy.

No proxy is actual employee worth.

### 8.21.8 Capacity and strategy configuration

Capacities: 10%, 20% primary, 30%.

Strategies:

- repeated random selection at same capacity;
- top-k calibrated risk;
- top-k proxy value;
- top-k calibrated risk × proxy value;
- reference threshold with actual selected fraction reported.

## 8.22 Evaluation and comparison framework

### 8.22.1 Prediction, calibration, complexity, and transparency metrics

| Category | Evidence |
| --- | --- |
| Primary ranking | Average Precision / PR-AUC |
| Secondary discrimination | ROC-AUC |
| Threshold performance | precision, recall, F1, balanced accuracy, confusion matrix; accuracy with warning |
| Stability | fold values, mean, standard deviation, failures |
| Calibration | Brier score, reliability curve, calibration slope/intercept when implemented |
| Runtime/resources | fit time, prediction time, device, epoch/tree/round count, parameter count for FT when available |
| Reproducibility | seed, environment, reload test, deterministic limitations |
| Interpretation | EBM term completeness, CatBoost post-hoc explanation, FT permutation/ablation if valid |
| Complexity | preprocessing burden, training code, artefact count, compute and maintenance |

### 8.22.2 Prioritization metrics

- precision@capacity;
- recall@capacity;
- lift@capacity;
- captured-positive count;
- value-weighted recall@capacity;
- selected proxy-value share;
- repeated-random distribution and uncertainty.

### 8.22.3 Model and strategy selection rule

1. Confirm every candidate has complete OOF coverage.
2. Rank by mean development PR-AUC.
3. Inspect fold variation and practical difference.
4. Reject candidates with invalid leakage, unstable failures, or unreproducible artefacts.
5. Compare calibration because probability enters the priority score.
6. Compare runtime, complexity, interpretation, and subgroup behaviour.
7. Under practical equivalence, prefer stronger reproducibility and valid interpretability over unnecessary complexity.
8. Freeze the winner before formal holdout.
9. Do not call any model operationally useful unless it exceeds the predefined usable-signal and top-k lift rules.

### 8.22.4 Uncertainty and robustness

- report all five fold values, mean, and standard deviation;
- bootstrap formal-holdout metrics when defined;
- compare selected FT configuration across extra seeds when feasible;
- report class-weight sensitivity;
- report sensitive-feature ablations;
- report capacity and proxy sensitivity;
- explicitly state when uncertainty prevents a confident winner.

## 8.23 Explainability and interpretability plan

### Explainable Boosting Machine

- global feature-effect functions;
- retained pairwise interactions;
- local additive contribution breakdowns;
- inspection for implausible shapes and sparse-region instability;
- explicit statement that effects are model associations, not causes.

### CatBoost

- native feature importance and/or SHAP only when reproducible and understood;
- global summary and selected local cases;
- dependence plots only with sample-size/context warnings;
- label all explanations as post-hoc model-behaviour explanations.

### FT-Transformer

FT-Transformer is not inherently interpretable. Use only methods the group can validate and explain:

- permutation importance on untouched outer-validation/holdout predictions;
- grouped feature ablation or masking sensitivity;
- optional gradient-based attribution only after implementation validation;
- do not equate raw attention weights with feature importance or causality.

### Optional Logistic Regression

Report standardized/encoded coefficients with category/reference-level cautions. Coefficients are associative, not causal.

### Cross-model explanation comparison

Compare whether the models emphasize similar features, where they disagree, and whether explanation stability changes across folds. Do not force methods into a single numeric scale when their meanings differ.

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
| T-005 | Identical OOF row coverage across CatBoost, EBM, FT-Transformer |
| T-006 | Dummy baselines present |
| T-007 | CatBoost receives named native categorical values, not one-hot matrix |
| T-008 | EBM feature names/types and interactions valid |
| T-009 | FT categorical vocabularies fitted on outer training only; unknown/missing codes work |
| T-010 | FT tensor dtypes/shapes/cardinalities valid |
| T-011 | Inner validation contains only outer-training rows |
| T-012 | Search counts do not exceed frozen caps |
| T-013 | No holdout access before freeze gate |
| T-014 | Probability range, NaN, and row-index checks |
| T-015 | Calibration uses development-only data |
| T-016 | Proxy normalizers fitted development-only |
| T-017 | Capacity selections contain exactly expected k records |
| T-018 | Random baseline reproducible and repeated |
| T-019 | CatBoost native reload reproduces predictions |
| T-020 | EBM reload reproduces predictions |
| T-021 | FT state_dict plus preprocessing reload reproduces predictions on CPU |
| T-022 | Explanation artefacts reproducible and correctly labelled |
| T-023 | Subgroup denominators and unstable-rate rules enforced |
| T-024 | Cross-document model/version/team consistency |
| T-025 | Fresh-kernel notebook execution and unit tests pass |

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
| CatBoost artefacts | native model, metadata, parameters, explanations |
| EBM artefacts | model, feature types, terms/interactions, explanations |
| FT artefacts | state_dict, architecture, vocabularies, scaler/imputer, epoch, curves, seed/device |
| Evaluation | common metrics, calibration, uncertainty, runtime, errors |
| Prioritization | strategy/capacity/proxy comparisons |
| Ethics/fairness | subgroup tables, risks, mitigations, limitations |
| Report | professional, evidence-backed, within formal page limit |
| Presentation | required proposal/final content as applicable |
| Viva package | methodology, results, pair ownership, question bank |
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
| 4. Model Design and Implementation | Dummy; optional Logistic; CatBoost; EBM; FT; bounded tuning; fold-local early stopping |
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
- Do not claim FT-Transformer should win because it is deep learning.
- Keep all claims traceable to saved artefacts.

## 8.29 Proposal presentation plan

### 8.29.1 Working slide structure

| Slide | Content | Lead |
| --- | --- | --- |
| 1 | Title, group, module, two-stage objective | Member 1 |
| 2 | Dataset source, size, target, feature groups | Member 2 |
| 3 | Data quality, imbalance, construction/provenance limitations | Member 2 |
| 4 | Real-world framing and why prediction alone is insufficient | Member 3 |
| 5 | Top-20% capacity and retention-value proxy | Member 3 |
| 6 | Three model families and why they are diverse | Member 4 |
| 7 | CatBoost, EBM, FT-Transformer model-specific preprocessing | Member 5 |
| 8 | Planned CV, tuning, calibration, and metrics | Member 5 |
| 9 | Ethics, subgroup checks, weak-signal contingency | Member 6 |
| 10 | AI-tool plan, team division, next steps | Member 6 |

### 8.29.2 Slide content rules

- State that the assignment requires at least two algorithms; the group chose three advanced families.
- Describe Dummy as a baseline and Logistic Regression as optional.
- Explain one model per two-member pair.
- Do not imply completed model results at proposal stage.
- Do not guarantee full marks or model superiority.
- Explain that all models use identical rows/folds with different representations.
- Make the performance-versus-interpretability-versus-complexity comparison visible.

## 8.30 Team responsibility map

### 8.30.1 Pair ownership

| Pair | Members | Required model | Additional rubric ownership | Cross-review |
| --- | --- | --- | --- | --- |
| Pair 1 | Members 1-2 | CatBoost | data audit, cleaning decisions, split/fold registry, shared feature contract | Review Pair 3 FT preprocessing/row parity |
| Pair 2 | Members 3-4 | EBM | EDA, visualization, inherent interpretation, subgroup-analysis support | Review Pair 1 CatBoost explanation/configuration |
| Pair 3 | Members 5-6 | FT-Transformer | common evaluator, calibration, training artefacts, prioritization integration | Review Pair 2 EBM feature types/interactions |

### 8.30.2 Individual numbered roles

| Member | Primary role | Secondary/shared role |
| --- | --- | --- |
| Member 1 | CatBoost implementation/tuning | project integration, requirements, Git workflow |
| Member 2 | data audit, preprocessing, split/fold registry | CatBoost adapter tests and documentation |
| Member 3 | EBM implementation/tuning | EDA and feature-effect interpretation |
| Member 4 | EDA/visuals, EBM explanations | subgroup analysis and report figures |
| Member 5 | FT architecture/training/checkpoints | common OOF evaluator and runtime logging |
| Member 6 | FT preprocessing/validation | calibration, prioritization, ethics, AI-use consistency |

### 8.30.3 Shared obligations

All members must understand the dataset, split, leakage controls, PR-AUC, calibration, model roles, prioritization, limitations, ethics, and actual results. Every pair must peer-review code outside its own workstream. Report writing, presentation practice, and final QA are shared.

## 8.31 Timeline and milestones

Use milestone dependencies rather than exact administrative dates.

| Milestone | Completion evidence |
| --- | --- |
| M1 Context/model decision synchronized | v4.2.0 MD/DOCX and repository commit |
| M2 Audit/EDA complete | fresh-kernel outputs and peer review |
| M3 Split/adapters complete | saved registry and passing tests |
| M4 Three minimal models smoke-tested | fit/predict/reload for CatBoost, EBM, FT |
| M5 Search spaces frozen | configuration commit |
| M6 OOF comparison complete | complete predictions and metrics |
| M7 Model/calibration frozen | decision record |
| M8 Formal holdout evaluated once | access record and saved outputs |
| M9 Prioritization/fairness complete | strategy and subgroup tables |
| M10 Report/presentation/viva ready | final QA checklist |

## 8.32 Current project status

| Item | Status |
| --- | --- |
| Dataset and target | VERIFIED |
| Two-stage scope | DECIDED |
| CatBoost/EBM/FT registry | DECIDED |
| Three two-member pairs | DECIDED |
| Dummy baseline | DECIDED mandatory |
| Logistic benchmark | DECIDED optional |
| Audit notebook | IN PROGRESS / FUNCTIONAL |
| Split/fold registry | TO BE GENERATED |
| Model adapters | TO BE GENERATED |
| Formal model results | TO BE GENERATED |
| Formal holdout access | NOT YET PERMITTED |
| Prioritization results | TO BE GENERATED |
| Report/presentation final artefacts | TO BE GENERATED |

## 8.33 Planned results-record templates

### 8.33.1 Experiment register

| Experiment ID | Model | Fold/seed | Configuration | Device | Runtime | Output path | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TO BE GENERATED | | | | | | | |

### 8.33.2 Cross-validation comparison

| Model | Mean PR-AUC | SD | Fold values | ROC-AUC | Brier | F1 rule | Runtime | Failures |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Dummy prior | | | | | | | | |
| Optional Logistic | | | | | | | | |
| CatBoost | | | | | | | | |
| EBM | | | | | | | | |
| FT-Transformer | | | | | | | | |

### 8.33.3 Complexity and evidence comparison

| Model | Representation | Interpretability | Training complexity | Artefact complexity | Reproducibility | Practical conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| CatBoost | | | | | | |
| EBM | | | | | | |
| FT-Transformer | | | | | | |

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

| Model | Method | Global evidence | Local/error cases | Stability | Valid interpretation | Limitations |
| --- | --- | --- | --- | --- | --- | --- |
| CatBoost | | | | | | |
| EBM | | | | | | |
| FT-Transformer | | | | | | |

### 8.33.10 Final conclusion template

- Best development model:
- Practical-equivalence assessment:
- Formal-holdout corroboration:
- Calibration conclusion:
- FT complexity justified?:
- CatBoost transparency trade-off justified?:
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
- describe Logistic Regression as optional;
- describe FT-Transformer as a distinct deep model whose value must be demonstrated;
- keep claims tied to artefacts and formal requirements.

## 8.35 AI-tool plan and actual-use live record

### 8.35.1 Preliminary AI-tool plan

| Tool category | Planned support | Required verification |
| --- | --- | --- |
| General assistants | framing, explanations, report editing, viva questions | compare with formal sources and project decisions |
| Coding assistants | boilerplate, debugging, tests, refactoring | execute, test, peer-review, inspect leakage |
| Research/search tools | official model/package documentation | use primary sources and record access |
| Document tools | synchronized context/report/presentation generation | structural and visual QA |

### 8.35.2 Actual AI-tool usage - LIVE RECORD

| Record ID | Date | Tool/version | Task supported | Extent | Verification performed | Human ownership/status |
| --- | --- | --- | --- | --- | --- | --- |
| AI-001 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Interpreted assignment/proposal requirements and planned presentation/report structure. | Moderate | Checked against formal PDFs. | User reviewed; group responsible. |
| AI-002 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Audited archive/CSV and identified schema, checksums, class balance, deterministic relationships, and provenance limitations. | Moderate | Code-based inspection and source review. | User approved recommendations. |
| AI-003 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Recommended target, methodology, metrics, ethics, team roles, repository, and no-signal contingency. | Moderate | Tied to rubric and audit evidence. | User approved. |
| AI-004 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Generated canonical Markdown and synchronized Word context files. | Moderate | YAML/content checks and DOCX rendering. | Group owns maintenance. |
| AI-005 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Expanded context to v3 with source extracts, data contract, holdout caveat, evaluation/fairness rules, and synchronization protocol. | Extensive | Validated YAML/headings/checksums and rendered Word. | User requested update. |
| AI-006 | 2026-07-29 | ChatGPT - GPT-5.6 Thinking | Updated title and exact assessed-build definition in v3.1. | Moderate | Content checks and Word rendering. | User approved. |
| AI-007 | 2026-07-28 to 2026-08-01 | ChatGPT - GPT-5.6 Thinking | Guided Jupyter audit notebook, result exports, chart creation, GitHub setup, and collaboration workflow. | Extensive | User executed code and supplied screenshots/results. | User performed steps; group retains responsibility. |
| AI-008 | 2026-08-01 | ChatGPT - GPT-5.6 Thinking | Formalized resignation-risk plus constrained retention prioritization; defined capacity, proxies, strategies, metrics, safeguards, and v4.0 context. | Extensive | Rechecked assignment text, preserved dataset evidence, validated Markdown/YAML, rendered Word. | User approved scope; group owns implementation. |
| AI-009 | 2026-08-05 | ChatGPT - GPT-5.6 Thinking | Reviewed model options, researched CatBoost/InterpretML/scikit-learn behaviour, replaced Random Forest/Decision Tree with CatBoost/EBM, and regenerated v4.1 context. | Extensive | Preserved formal requirements and dataset facts; validated model references; rendered and inspected Word. | User approved; group owns execution. |
| AI-010 | 2026-08-06 | ChatGPT - GPT-5.6 Thinking | Recorded canonical repository, removed nonessential handoff gaps, aligned requirement IDs, corrected Word tables, and synchronized v4.1.1. | Extensive | Validated YAML, consistency searches, DOCX generation, structural and visual QA. | User approved; group owns repository maintenance. |
| AI-011 | 2026-08-06 | ChatGPT - GPT-5.6 Thinking | Reviewed v4.1.1, verified primary CatBoost/InterpretML/FT-Transformer documentation, changed the required registry to CatBoost/EBM/FT-Transformer, defined three pair workstreams, added FT preprocessing/tuning/evaluation/artefact rules, removed nonessential administrative timing from active requirement extracts, and regenerated synchronized v4.2.0 files. | Extensive | Parsed and validated YAML, searched superseded model/version statements, generated DOCX from canonical Markdown, rendered every page, visually inspected layouts, and compared extracted DOCX content against Markdown. | User approved the model decision; group owns implementation, testing, repository commit, and final academic use. |

Future records must include actual tool/version if known, precise tasks, extent, verification, retained evidence, and human ownership.

## 8.36 Decision and change log

| Decision ID | Date | Status | Topic | Active decision | Authority | Reconsideration |
| --- | --- | --- | --- | --- | --- | --- |
| Q-001 to Q-030 | 2026-07-28 to 2026-08-01 | DECIDED / historical | Initial methodology and two-stage scope | Retained except where explicitly superseded | Formal sources and user decisions | Higher-authority or logged change |
| Q-032 | 2026-08-05 | SUPERSEDED | Two-advanced-model registry | CatBoost and EBM only | Earlier user decision | Superseded by Q-039 |
| Q-033 | 2026-08-05 | DECIDED | Earlier core models | Random Forest and Decision Tree removed | User | Explicit reinstatement before holdout |
| Q-034 | 2026-08-05 | SUPERSEDED IN PART | Model-specific preprocessing | CatBoost/EBM rules retained; expanded for FT | User/method requirements | Q-040 |
| Q-035 | 2026-08-05 | SUPERSEDED IN PART | Performance-transparency rule | Expanded to performance-calibration-stability-interpretability-complexity across three models | User | Q-041 |
| Q-036 to Q-038 | 2026-08-05 to 2026-08-06 | DECIDED | Dual-file handoff, environment, repository | Retained | User | Logged change |
| Q-039 | 2026-08-06 | DECIDED | Required advanced model registry | CatBoost, EBM, and FT-Transformer; Dummy mandatory; Logistic optional | Explicit user decision | Higher-authority requirement or logged pre-holdout decision |
| Q-040 | 2026-08-06 | DECIDED | FT-Transformer protocol | RTDL/PyTorch, fold-local standardization/vocabularies, embeddings, training-only inner validation, bounded search, checkpoint/reload | User approval plus official implementation guidance | Verified incompatibility plus approved redesign |
| Q-041 | 2026-08-06 | DECIDED | Model selection | Compare three advanced models on common OOF evidence, calibration, stability, runtime, reproducibility, and valid interpretation; use practical equivalence | Methodological review | Logged pre-holdout change |
| Q-042 | 2026-08-06 | DECIDED | Team division | One advanced model per two-member pair with additional rubric ownership and cross-review | Explicit user decision | Group-approved role revision |
| Q-043 | 2026-08-06 | DECIDED | Optional/fallback models | Logistic optional; RBF-SVM superseded; XGBoost contingency-only with explicit substitution decision | Explicit user decision and feasibility review | Logged pre-holdout change |
| Q-044 | 2026-08-06 | DECIDED | Formal-source handoff content | Keep project-relevant requirement extracts; exclude nonessential exact administrative timing/submission logistics from active handoff | Prior user cleanup instruction | Need for exact official wording |

### Change log

| Version | Date | Change | Rationale | Status |
| --- | --- | --- | --- | --- |
| 2.0.0 | 2026-07-28 | Created AI-first context. | Complete handoff. | SUPERSEDED |
| 3.0.0 | 2026-07-28 | Added data contract, requirements, evaluation, fairness, synchronization. | Execution-safe handoff. | SUPERSEDED |
| 3.1.0 | 2026-07-29 | Adopted assignment-aligned title/build. | Clarify assessed workflow. | SUPERSEDED |
| 4.0.0 | 2026-08-01 | Added constrained retention prioritization. | Connect prediction to limited resources. | SUPERSEDED |
| 4.1.0 | 2026-08-05 | Adopted CatBoost/EBM and model-specific adapters. | More distinctive/interpretable comparison. | SUPERSEDED |
| 4.1.1 | 2026-08-06 | Added repository and handoff cleanup. | Clean synchronized package. | SUPERSEDED |
| 4.2.0 | 2026-08-06 | Locked CatBoost, EBM, and FT-Transformer as required advanced models; added pair ownership, FT adapter/training/search/calibration/artefact/testing rules, updated report/presentation/results plans, and normalized formal requirement extracts. | Implement the user-approved strongest three-family comparison while preserving leakage control, reproducibility, and AI handoff completeness. | ACTIVE |

## 8.37 Remaining identity and ownership inputs

| ID | Input | Handling | Blocks technical work? |
| --- | --- | --- | --- |
| EXT-001 | Official group number and lab batch/session | Add to formal artefacts when known | No |
| EXT-002 | Six names/student IDs mapped to Members 1-6 | Replace numbered roles after group confirmation | No |
| EXT-003 | Lecturer/module-coordinator name | Use official module information | No |
| EXT-004 | Confirmed CPU/GPU execution environment | Record after smoke test; do not assume GPU | No, but affects FT runtime |

## 8.38 References and source-file register

### Formal and project sources

1. **SRC-001:** Faculty of Computing. *IT2011 - Artificial Intelligence and Machine Learning: Group Assignment Specification*. Year 2 Semester 1, 2026. Local file and SHA-256 recorded in Section 1.
2. **SRC-002:** Faculty of Computing. *IT2011 - Artificial Intelligence and Machine Learning: Proposal Presentation*. Year 2 Semester 1, 2026. Local file and SHA-256 recorded in Section 1.
3. **SRC-003:** Assigned archive, `archive.zip`; checksum recorded.
4. **SRC-004:** `Extended_Employee_Performance_and_Productivity_Data.csv`; checksum recorded.
5. **SRC-005:** Kaggle, *Employee Performance and Productivity Data*: `https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data`.
6. **SRC-006 to SRC-011:** Prior explicit user decisions recorded in the machine-readable source register.
7. **SRC-012:** User decision, 2026-08-06: CatBoost, EBM, and FT-Transformer required; one per two-member pair; Dummy mandatory; Logistic optional; update synchronized handoff.

### Official software and research sources supporting the active methodology

8. CatBoost. *CatBoostClassifier Python reference*. `https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier`.
9. CatBoost. *Categorical features*. `https://catboost.ai/docs/en/features/categorical-features`.
10. CatBoost. *Training parameters and parameter tuning*. `https://catboost.ai/docs/en/references/training-parameters/` and `https://catboost.ai/docs/en/concepts/parameter-tuning`.
11. InterpretML. *Explainable Boosting Machine*. `https://interpret.ml/docs/ebm.html`.
12. InterpretML. *ExplainableBoostingClassifier API*. `https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html`.
13. InterpretML. *EBM hyperparameters*. `https://interpret.ml/docs/hyperparameters.html`.
14. Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021. `https://arxiv.org/abs/2106.11959`.
15. Yandex Research. *rtdl-revisiting-models*, official recommended FT-Transformer package implementation. `https://github.com/yandex-research/rtdl-revisiting-models`.
16. scikit-learn. *Probability calibration*. `https://scikit-learn.org/stable/modules/calibration.html`.
17. PyTorch. Official documentation for tensors, reproducibility, optimizers, loss functions, data loading, checkpointing, and device management: `https://pytorch.org/docs/stable/`.

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

