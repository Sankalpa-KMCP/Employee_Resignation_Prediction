---
title: "Design and Evaluation of Machine Learning Models for Employee Resignation Risk and Resource-Constrained Retention Prioritization"
subtitle: "Canonical project specification, live state, and AI handoff memory"
author: "IT2011 Project Group"
date: "2026-08-01"
version: "4.0.0"
---

# Document map

- **0. AI execution contract** - authority, mandatory behaviour, startup check, and anti-hallucination rules.
- **1. Machine-readable project memory** - structured project, dataset, methodology, live state, requirements, and update rules.
- **2. AI quick reference** - one-minute summary, terminology, and prohibited assumptions.
- **3. Current operating state** - milestone, completed work, blockers, risks, and next actions.
- **4-7. AI operating guidance** - task routing, file manifest, dependencies, synchronization, and update protocol.
- **8. Full canonical project specification** - requirements, dataset evidence, methodology, models, evaluation, ethics, deliverables, team plan, results templates, and logs.
- **9. Embedded official source transcripts** - complete normalized text of both supplied formal PDFs.
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

**DECIDED** - The assessed build remains an academic, reproducible Python study. It does not estimate whether a real retention intervention would succeed, does not perform genuine monetary budget optimization, and must not be used for real employment actions. Academic record-level ranking is permitted only inside the controlled simulation on the assigned dataset.

## 0.2 Canonical-source rule

1. This Markdown file is the canonical source of truth for project context.
2. The Word file named `Employee_Resignation_Prediction_AI_Project_Context_v4_0.docx` is a synchronized companion generated from this content.
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

- read Sections 0-7 and the embedded source policy in Section 9 before doing project-dependent work;
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
- [ ] The embedded normalized transcripts in Section 9 are available when written requirements matter.
- [ ] Original PDFs are requested only when exact visual appearance, annotations, or a newer/differently hashed revision must be verified.
- [ ] The dataset file is exactly `Extended_Employee_Performance_and_Productivity_Data.csv`.
- [ ] The dataset SHA-256 is `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb`.
- [ ] The dataset has 100,000 rows and 20 columns before transformation.
- [ ] The target is `Resigned`, with `True` as the positive class.
- [ ] The two-stage scope is preserved: resignation-risk evaluation followed by resource-constrained prioritization simulation.
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

- exact deadlines, lab times, viva arrangements, Q&A duration, submission platform, citation style, or page-count rules;
- team-member names, availability, ownership, repository URLs, or contribution percentages;
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
- Do not add a dashboard, application, API, deployment, neural network, external boosting framework, additional target, causal intervention model, or monetary optimization engine merely because it is technically possible.
- A target pivot after viewing formal holdout results is prohibited. A justified pivot before formal holdout evaluation requires a logged group or lecturer decision.
- Any change to the 20% primary capacity, value-proxy definition, ranking strategies, or evaluation metrics must be chosen from development evidence only and logged before formal-holdout access.
- If the Word companion is missing, outdated, or inconsistent, continue using this Markdown file and report that regeneration is required.

## 0.10 Mandatory AI startup acknowledgement

Before performing project-dependent work, an AI must provide a compact **Context acknowledgement** containing:

1. project objective and the two-stage build;
2. current phase and immediate milestone;
3. target column and positive class;
4. required models and primary prediction metric;
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
- The two files are sufficient for conceptual project understanding. Data-dependent execution still requires the exact CSV; implementation-dependent conclusions require the repository and saved results.

# 1. Machine-readable project memory

```yaml
document_control:
  context_schema_version: '4.0'
  project_context_version: 4.0.0
  synchronization_id: ERPC-20260801-V4.0
  generated_at: 2026-08-01 Asia/Colombo
  canonical_file: Employee_Resignation_Prediction_AI_Project_Context_v4_0.md
  word_companion_file: Employee_Resignation_Prediction_AI_Project_Context_v4_0.docx
  substantive_content_rule: Both files contain the same project information. Markdown is canonical; Word is a synchronized
    rendering/fallback.
  duplicate_input_rule: When both files are supplied, do not treat repeated statements as independent evidence and
    do not spend effort reconciling formatting-only differences.
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
project:
  name: Design and Evaluation of Machine Learning Models for Employee Resignation Risk and Resource-Constrained
    Retention Prioritization
  slug: Employee_Resignation_Prediction
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
    - DummyClassifier baseline
    - Logistic Regression and Random Forest training pipelines
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
    - real-world automated or individual employment action
    success_definition: Correct, reproducible, critically evaluated execution satisfies the project even if no model
      materially exceeds the dummy baseline or no prioritization strategy exceeds random selection.
live_state:
  status: active
  current_phase: proposal preparation, project setup, and approved scope update
  current_milestone: 15-minute proposal presentation during lab sessions in the week beginning 2026-08-03; exact
    date and time unconfirmed
  last_updated: '2026-08-01'
  updated_by: ChatGPT following the user-approved retention-prioritization scope expansion
  evidence:
  - SRC-001
  - SRC-002
  - SRC-004
  - SRC-006
  - SRC-007
  - SRC-008
  completed:
  - formal requirement extraction
  - dataset audit and data contract
  - initial methodology and scope decisions
  - shared GitHub repository created and published; canonical URL still not supplied
  - 01_data_audit notebook created with reusable CSV outputs and target-distribution figure
  - project scope expanded to include resource-constrained retention-prioritization simulation
  - context v4 generation
  active_next_actions:
  - replace the repository context files with synchronized v4 Markdown and Word copies
  - revise the proposal to explain the two-stage real-world problem and simulation
  - complete and peer-review the formal data audit and EDA notebooks
  - implement leakage-safe model, calibration, value-proxy, and prioritization workflows
  - obtain official administrative confirmations
  blockers:
  - team names and canonical repository URL not supplied
  - exact dates, submission rules, citation style, and page-limit inclusions not officially supplied
  - real intervention costs, outcomes, replacement costs, and business value are unavailable; monetary optimization
    claims are therefore blocked
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
  embedded_transcript: Section 9.2
- id: SRC-002
  name: Proposal Presentation.pdf
  sha256: 7cbf695c122b194880beb62a672b15eb56eaa470d43bf286a3794e9c10abb4d3
  authority: formal proposal instruction
  embedded_transcript: Section 9.3
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
embedded_source_policy:
  complete_normalized_transcripts_included: true
  transcript_use: Sufficient for understanding and applying the supplied written requirements when the original
    PDFs are absent.
  original_pdf_needed_when:
  - checking exact visual layout, signatures, stamps, annotations, or page appearance
  - verifying a newer or differently hashed revision
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
  requirement: Prepare a professional group report of no more than 15 pages and submit it as PDF before the viva.
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
  requirement: Proposal timing is tentatively during lab sessions in the week beginning 2026-08-03; exact date/time
    remains external.
  source: SRC-002
  status: VERIFIED with placeholder
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
    model_policy: exclude; retain only for traceability
  - name: Department
    raw_dtype: string/category
    nullable: false
    role: predictor
    model_policy: one-hot encode
  - name: Gender
    raw_dtype: string/category
    nullable: false
    role: sensitive predictor and audit attribute
    model_policy: include in full model; exclude in sensitivity model; one-hot encode
  - name: Age
    raw_dtype: integer
    nullable: false
    role: sensitive predictor and audit attribute
    model_policy: include in full model; scale for Logistic Regression; exclude in second sensitivity model
  - name: Job_Title
    raw_dtype: string/category
    nullable: false
    role: predictor
    model_policy: one-hot encode
  - name: Hire_Date
    raw_dtype: date-time text
    nullable: false
    role: audit field
    model_policy: parse for audit; exclude raw value from primary modelling
  - name: Years_At_Company
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain as tenure representation; scale for Logistic Regression
  - name: Education_Level
    raw_dtype: string/category
    nullable: false
    role: predictor
    model_policy: one-hot encode
  - name: Performance_Score
    raw_dtype: integer/ordinal
    nullable: false
    role: predictor
    model_policy: retain numeric; document ordinal interpretation
  - name: Monthly_Salary
    raw_dtype: integer/numeric
    nullable: false
    role: predictor
    model_policy: retain in primary model; unit must remain unspecified; test redundancy sensitivity if useful
  - name: Work_Hours_Per_Week
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain; scale for Logistic Regression
  - name: Projects_Handled
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain; scale for Logistic Regression
  - name: Overtime_Hours
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain; scale for Logistic Regression
  - name: Sick_Days
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain; scale for Logistic Regression
  - name: Remote_Work_Frequency
    raw_dtype: integer percentage
    nullable: false
    role: predictor
    model_policy: retain numeric; optional categorical sensitivity
  - name: Team_Size
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain; scale for Logistic Regression
  - name: Training_Hours
    raw_dtype: integer
    nullable: false
    role: predictor
    model_policy: retain; scale for Logistic Regression
  - name: Promotions
    raw_dtype: integer count
    nullable: false
    role: predictor
    model_policy: retain; scale for Logistic Regression or document categorical sensitivity
  - name: Employee_Satisfaction_Score
    raw_dtype: float/numeric
    nullable: false
    role: predictor
    model_policy: retain; scale for Logistic Regression
  - name: Resigned
    raw_dtype: boolean
    nullable: false
    role: target
    model_policy: map True=1 and False=0
  unknown_category_policy: OneHotEncoder(handle_unknown="ignore") for inference/reproducibility; report unexpected
    categories during validation.
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
  required_models:
  - Logistic Regression
  - Random Forest Classifier
  optional_model:
    name: Decision Tree Classifier
    condition: include only after required pipelines, tests, and proposal/report obligations are on schedule before
      model freeze
  logistic_search:
    solver: liblinear unless repository documents a compatible alternative
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
  random_forest_search:
    method: RandomizedSearchCV
    n_iter: 24
    n_estimators:
    - 200
    - 400
    - 600
    max_depth:
    - null
    - 8
    - 12
    - 20
    min_samples_split:
    - 2
    - 10
    - 30
    min_samples_leaf:
    - 1
    - 5
    - 20
    max_features:
    - sqrt
    - log2
    - 0.5
    class_weight:
    - null
    - balanced
    - balanced_subsample
    random_seed: 42
  model_selection_rule: 'Rank by mean CV Average Precision. Apply a one-standard-error rule: among configurations
    within one standard error of the best mean, choose the simplest/interpretable valid pipeline. Freeze before
    holdout use.'
  threshold_policy:
    reference_threshold: 0.5
    secondary_threshold: threshold maximizing F1 on pooled out-of-fold development predictions
    restriction: never select threshold from formal holdout
    reporting: report both thresholds and full precision/recall trade-off
  uncertainty:
    cv: mean, standard deviation, and fold values
    formal_holdout: 95% percentile bootstrap interval with 2000 stratified resamples and seed 42 when metric is
      defined
  calibration_rule: Brier score and reliability curves are mandatory. If the selected base model is materially miscalibrated
    on development out-of-fold predictions, compare a development-only calibrated variant and freeze the choice
    before holdout use.
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
  - jupyter
  - joblib
  - pytest
  - git
  exact_versions: TO BE GENERATED from requirements.txt or environment.yml at repository initialization
  version_rule: Record exact versions with every formal experiment and saved model.
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
  status: VERIFIED; schedule unresolved
- name: Employee_Resignation_Prediction_AI_Project_Context_v4_0.md
  format: Markdown
  status: canonical project context
- name: Employee_Resignation_Prediction_AI_Project_Context_v4_0.docx
  format: DOCX
  status: synchronized companion with same substantive content
fixed_decisions:
- Primary target remains Resigned with True as the positive class.
- 'The project has two linked stages: resignation-risk modelling and constrained retention-prioritization simulation.'
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
- Class imbalance is handled primarily through class weights and threshold analysis; no pre-split resampling.
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
- REPOSITORY-PATH-OR-URL
- EXACT-PROPOSAL-DATE-TIME
- ALL-MEMBERS-MUST-SPEAK
- Q_AND_A_TIMING
- FINAL-REPORT-DEADLINE-AND-TIMEZONE
- VIVA-DATE-AND-FORMAT
- SUBMISSION-PLATFORM
- OFFICIAL-FILE-NAMING-CONVENTION
- REQUIRED-CITATION-STYLE
- PAGE-LIMIT-INCLUSION-RULES
- OFFICIAL-INSTITUTIONAL-AI-OR-ACADEMIC-INTEGRITY-POLICY
- AI-GENERATED-TEXT-CITATION-RULE
- SUBMISSION-ARCHIVE-STRUCTURE
prohibited_assumptions:
- Do not claim the dataset contains verified real employees.
- Do not infer salary currency or units.
- Do not fabricate model results, deadlines, team identities, or lecturer decisions.
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
startup_acknowledgement:
  required_before_project_dependent_work: true
  fields:
  - project objective and two-stage scope
  - current phase
  - target and positive class
  - required models and primary prediction metric
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

**What is being created:** An academically defensible, reproducible two-stage study using the assigned employee dataset. Stage 1 evaluates resignation-risk prediction. Stage 2 simulates how a fixed HR capacity could prioritize records for review using risk-only and risk-plus-retention-value-proxy rankings.

**Why:** The assignment requires the group to apply AI/ML methods to a real-world problem and critically evaluate effectiveness and impact. The extended framing connects prediction to a realistic resource-allocation constraint without claiming a production HR or budget-allocation system.

**Current phase:** Proposal preparation, project setup, and approved scope update.

**Immediate objective:** Revise the proposal to explain the assigned dataset, resignation target, limited-capacity HR problem, two-stage solution, suitability, limitations, planned methodology, ethics, and planned AI-tool use.

**Primary prediction output:** Evidence on whether the available variables distinguish `Resigned=True` from `Resigned=False` out of sample.

**Primary prioritization output:** Evidence on whether selecting the top 20% by calibrated risk or by calibrated risk multiplied by a transparent retention-value proxy captures more actual positive records and more positive-class proxy value than repeated random selection or simpler rankings.

**Fixed methodology:** Stratified 80/20 split with seed 42; five-fold stratified CV on the 80% development partition; prior and most-frequent dummy baselines; Logistic Regression and Random Forest; PR-AUC as the primary prediction metric; probability calibration assessment; development-only definition and freezing of proxy/capacity/strategy; and one-time formal-holdout corroboration. The holdout is not described as historically pristine because preliminary audit split metadata was not retained.

**Capacity and proxy:** Primary capacity is 20%; 10% and 30% are sensitivities. The primary retention-value proxy is the equal-weight mean of development-fitted normalized `Performance_Score`, `Monthly_Salary`, and `Years_At_Company`. It is not actual employee worth. A no-salary proxy and a normalized multiplicative proxy are sensitivity analyses.

**Main constraints:** No real intervention cost, intervention success, replacement cost, revenue contribution, or verified employee-value data exist. Therefore the project is a prioritization simulation, not genuine financial optimization or proof that spending money on selected records would prevent resignation.

**Critical data facts:** 100,000 rows, 20 columns, no missing values, no duplicate rows, 10.01% positive target, redundant hire-date/tenure information, deterministic salary structure by job title and performance score, and preliminary audit-only evidence suggesting weak resignation signal.

**Next task:** Replace repository context files with v4, revise the proposal, complete formal EDA, and implement the model-calibration-prioritization workflow.

## 2.2 Critical terminology

| Term | Project-specific meaning |
| --- | --- |
| Positive class | `Resigned=True`. |
| PR-AUC / Average Precision | Primary prediction-ranking metric for the imbalanced positive class. |
| Formal holdout | The 20% stratified partition locked against tuning and used once after model, calibration, proxy, capacity, and strategy freeze. It carries a historical preliminary-audit exposure caveat. |
| Development CV | Five-fold StratifiedKFold performed only on the 80% development-training partition; primary model-comparison evidence. |
| Resignation-risk probability | A model output whose calibration must be assessed before it is used inside the priority score. It is not certainty. |
| Capacity | The fraction of records that can be selected in the academic simulation. Primary scenario: 20%. |
| Retention-value proxy | A transparent score constructed from available variables for academic comparison. It is not actual employee worth, financial contribution, or replacement cost. |
| Priority score | Calibrated resignation-risk probability multiplied by the selected retention-value proxy. |
| Precision@20% | Share of selected top-20% records that are actually positive. |
| Recall@20% | Share of all actual positive records captured in the selected top 20%. |
| Lift@20% | Precision@20% divided by overall positive prevalence; compares the selected group with random selection. |
| Value-weighted recall@20% | Share of total positive-class proxy value contained in the selected top 20%. It does not mean value was actually protected. |
| Predictive signal | Out-of-sample association materially exceeding a suitable baseline; not causality. |
| Fairness audit | Subgroup prediction and selection analysis; not a declaration of legal fairness or compliance. |

## 2.3 Items an AI must never assume

- that the employees are verified real individuals;
- that `Monthly_Salary` is in USD, LKR, or any other currency;
- that a high accuracy value indicates useful resignation prediction;
- that class weighting or resampling creates missing target signal;
- that correlation or model importance identifies why employees resign;
- that the retention-value proxy measures actual worth, contribution, replacement cost, or revenue;
- that the 20% scenario is an actual organization's budget;
- that selecting or supporting a record would prevent resignation;
- that the risk-times-value ranking is financially optimal;
- that a fairness check makes the strategy automatically fair or non-discriminatory;
- that every member must speak, or that Q&A is included, unless officially confirmed;
- that the tentative proposal week is an exact date;
- that a model should be deployed or used for real individual HR action.

# 3. Current operating state

## 3.1 Current milestone

**VERIFIED** - The proposal is a 15-minute PowerPoint presentation per group, delivered in class during lab sessions. The formal proposal sheet gives the week beginning 3 August 2026 as tentative; exact date and time remain unresolved.

## 3.2 Immediate objectives

1. Replace the repository context files with synchronized v4 Markdown and Word copies.
2. Revise the proposal around the two-stage real-world problem: resignation-risk estimation plus limited-capacity prioritization.
3. Complete and peer-review the formal data-audit and EDA workflow.
4. Freeze the model, calibration, value-proxy, capacity, and ranking-strategy protocol before formal evaluation.
5. Allocate six speaking roles and rehearse to 13-14 minutes.
6. Confirm official administrative placeholders with the lecturer or LMS.

## 3.3 Completed work

- **VERIFIED** - Formal assignment and proposal PDFs were extracted and reviewed.
- **VERIFIED** - The uploaded archive was decompressed and the CSV mechanically profiled.
- **VERIFIED** - File checksums, dimensions, schema, missingness, duplicates, class balance, ranges, and major deterministic relationships were recorded.
- **VERIFIED AUDIT ONLY** - A limited diagnostic indicated that resignation may contain little predictive signal. This is not a final experiment.
- **COMPLETED** - A shared GitHub repository was created and published; its canonical URL is still a placeholder in this context.
- **COMPLETED / LIVE** - `01_data_audit.ipynb` was created with dataset checks, reusable CSV summaries, and a target-distribution figure; final reproducibility cleanup and peer review remain required.
- **DECIDED** - On 2026-08-01, the user approved expanding the project from resignation prediction alone to a two-stage resignation-risk and constrained retention-prioritization simulation.
- **DECIDED** - `Resigned` remains the target; the new stage does not create a second target.
- **DECIDED** - Real employee worth, real budget optimization, and intervention effectiveness are not claimed because the dataset does not contain the required evidence.
- **COMPLETED** - Context v4 was generated to encode the new scope, protocol, risks, evaluation metrics, and presentation/report changes.

## 3.4 Work that may proceed now

- update and commit the v4 context files;
- final audit-notebook cleanup and peer review;
- proposal drafting and rehearsal;
- EDA planning and figure generation;
- leakage-safe preprocessing and model pipelines against development data only;
- calibration evaluation and calibrated-model comparison;
- development-only construction of retention-value proxies;
- implementation of repeated-random, risk-only, value-only, combined-priority, and threshold baselines;
- capacity sensitivity at 10%, 20%, and 30%;
- unit tests and result-schema design;
- AI-use logging and source-register maintenance.

## 3.5 Blocked or conditional work

- final submission scheduling and calendar planning depend on official dates;
- final report formatting depends on the official citation style and page-count rules;
- final model claims depend on formal experiments;
- final viva plan depends on the viva format and date;
- assigning named owners depends on team-member details.

## 3.6 Work not to prioritize now

- web or mobile applications;
- production deployment, APIs, databases, or cloud infrastructure;
- complex neural networks or external boosting frameworks without evidence of need;
- changing the target to obtain higher scores;
- causal intervention modelling without intervention data;
- assigning monetary value to employees;
- claiming a financially optimal budget allocation;
- elaborate fairness mitigation before confirming usable predictive and prioritization signal;
- drafting final conclusions before formal experiments.

## 3.7 Current risks

| Risk ID | Risk | Probability | Impact | Current control | Owner |
| --- | --- | --- | --- | --- | --- |
| R-001 | `Resigned` may be independent of available predictors. | High | High | Use baselines, pre-registered metrics, and a negative-result contingency. | Member 4 and Member 5 roles; names unresolved |
| R-002 | A weak risk model could make every downstream prioritization strategy unreliable. | High | High | Require stable lift over random selection; do not interpret value-only ranking as successful resignation prioritization. | Members 4-5 |
| R-003 | The retention-value proxy may be mistaken for actual employee worth. | High | High | Use explicit proxy terminology, publish the formula, and run sensitivity analyses. | Members 3 and 6 |
| R-004 | Salary determinism may indirectly double-count performance. | High | Medium | Use a no-salary proxy sensitivity and avoid raw multiplication as the primary proxy. | Members 2-3 |
| R-005 | The 20% capacity may be mistaken for a verified budget. | Medium | High | Label it a hypothetical equal-cost capacity and report 10%/30% sensitivity. | Member 1 |
| R-006 | Resource prioritization may amplify gender or age disparities. | Medium | High | Compare subgroup prediction and selection metrics and sensitive-feature variants. | Member 6 |
| R-007 | Accuracy may conceal failure on the 10.01% positive class. | High | High | Use PR-AUC and ranking-at-capacity metrics. | Member 5 |
| R-008 | Leakage through holdout use, calibration, proxy scaling, or strategy selection. | Medium | High | Fit and choose all learned or design elements on development data only; lock the holdout. | Members 2, 4, and 5 |
| R-009 | Dataset provenance may be misrepresented as real employee data. | Medium | High | Use cautious provenance wording and prohibit unsupported population claims. | Members 2 and 6 |
| R-010 | Six independently produced sections may become inconsistent. | Medium | High | Common split, proxy definitions, schemas, saved results, peer review, and final integration. | All members |
| R-011 | AI-assisted output may be used without validation or disclosure. | Medium | High | Maintain live usage records and require human/code verification. | Member 6 |

## 3.8 Next concrete actions

| Order | Action | Completion evidence | Status |
| --- | --- | --- | --- |
| 1 | Replace v3.1 context files in `docs/context/` with synchronized v4 files. | Both files committed and filenames updated. | TO BE GENERATED |
| 2 | Finish relative paths, checksum/schema assertions, conclusion, and peer review in `01_data_audit.ipynb`. | Fresh-kernel run succeeds and outputs match. | IN PROGRESS |
| 3 | Revise the proposal around the two-stage problem and solution. | Slides cover all formal topics and the new limited-capacity simulation. | TO BE GENERATED |
| 4 | Create `02_eda.ipynb`. | Approved figures and data-quality findings saved. | TO BE GENERATED |
| 5 | Implement and freeze split, preprocessing, models, calibration, value proxies, capacities, and ranking strategies. | Code, configuration, and split ID committed before holdout use. | TO BE GENERATED |
| 6 | Assign names to six role slots. | Team map updated. | PLACEHOLDER - HUMAN INPUT REQUIRED |
| 7 | Request official administrative confirmations. | LMS notice or lecturer response saved. | PLACEHOLDER - EXTERNAL INPUT REQUIRED |
| 8 | Rehearse and time the proposal. | Complete run is 13-14 minutes. | TO BE GENERATED |

# 4. Task router for AI assistants

| Future request | Must read | Must verify | May generate | Must not invent |
| --- | --- | --- | --- | --- |
| Understand the project | Sections 0-3 and 8.1-8.10 | Current version, two-stage scope, status, and placeholders | Concise summary with statuses | New objectives, dates, or results |
| Inspect data | Sections 5, 8.10-8.14, and 8.18-8.19 | CSV checksum, shape, schema | Audit code, tables, EDA plan | Units, provenance, missingness, or semantics not supported |
| Generate preprocessing code | Sections 8.15-8.19 | Feature policy, split, proxy scaling policy | Leakage-safe pipelines and validation tests | Full-data scaling, raw-date use, or silent row deletion |
| Generate modelling code | Sections 8.20-8.22 | Development/holdout separation, metrics, calibration | Dummy, Logistic Regression, Random Forest, optional Decision Tree | Holdout tuning or unapproved models as required scope |
| Generate prioritization code | Sections 8.16-8.24 | Frozen capacity, proxy definitions, calibration, strategies, ranking metrics | `prioritization.py`, random baselines, top-k evaluation | Actual employee worth, intervention benefit, or optimal-budget claims |
| Perform EDA | Sections 8.11, 8.18, and 8.19 | Exact file and target orientation | Reproducible charts and descriptive findings | Causal explanations or invented outliers |
| Tune hyperparameters/calibration | Sections 8.21-8.22 | Search space, CV, scoring, seed | Search code and results schema | Using formal holdout for tuning |
| Evaluate results | Sections 8.22-8.25 and 8.33 | Saved OOF and formal-holdout predictions | Prediction, calibration, ranking, uncertainty, error, fairness tables | A winner when differences are unstable |
| Write a report section | Sections 8.6-8.9 and relevant method/result sections | Formal report order, result provenance, page limit | Evidence-backed prose | Fabricated findings or omitted limitations |
| Create presentation content | Sections 8.28-8.29 | 15-minute duration and required topics | Ten-slide content, notes, figures | Mandatory slide count, Q&A, or exact date |
| Prepare viva questions | Sections 8.2, 8.13, 8.17-8.25, and 8.33 | Approved decisions and actual results | Questions and model answers | Results not generated |
| Debug code | Sections 5, 8.17-8.18, 8.25, and repository README | Environment, paths, split, failing test | Minimal corrections and tests | Method changes that hide failures |
| Update status | Sections 3, 7, 8.31-8.36 | Evidence for completion/change | Status, risk, AI-use, and change-log entries | Unverified completion claims |
| Recommend scope change | Sections 0.9, 6, 8.8, and 8.36 | Trigger, requirement, cost, authority | Decision proposal with trade-offs | Automatic unlogged implementation |
| Handle a missing file | Sections 5 and 7 | Manifest, checksum, exact filename | Missing-file report and recovery steps | Substituting a different file |

# 5. Companion-file manifest and handoff package

## 5.1 Required files for project-dependent work

| ID | Canonical filename or path | Required for | Integrity rule | Status |
| --- | --- | --- | --- | --- |
| CTX-MD | Employee_Resignation_Prediction_AI_Project_Context_v4_0.md | All project-dependent work | Canonical; version and date must match this document. | VERIFIED generated |
| CTX-DOCX | Employee_Resignation_Prediction_AI_Project_Context_v4_0.docx | Synchronized cross-format fallback and Word navigation | Must be regenerated from CTX-MD after substantive changes. | VERIFIED generated |
| SRC-001 | `Group Assignment Specification.pdf` or embedded Section 9.2 transcript | Requirements, marks, report and AI-use rules | PDF SHA-256 recorded; normalized complete transcript embedded. | VERIFIED |
| SRC-002 | `Proposal Presentation.pdf` or embedded Section 9.3 transcript | Proposal content, duration, mode, tentative timing | PDF SHA-256 recorded; normalized complete transcript embedded. | VERIFIED |
| SRC-003 | archive.zip | Original assigned archive | SHA-256 `abe4a9aa869fe553faec813a06ff93432900d72af2036058f8b84e3dd64db375`. | VERIFIED |
| SRC-004 | data/raw/Extended_Employee_Performance_and_Productivity_Data.csv | All data analysis and modelling | SHA-256 `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb`. | VERIFIED source; repository copy TO BE GENERATED |
| SRC-005 | Kaggle dataset page | Source citation and licence metadata | Use exact URL; do not treat description as original provenance evidence. | VERIFIED listing |
| CODE | Repository source and notebooks | Reproducibility and implementation | Path and commit must be recorded when created. | PLACEHOLDER |
| RESULTS | results/ and models/ artefacts | Report, viva, comparison | Must map to code commit, split ID, and environment. | TO BE GENERATED |

## 5.2 Optional files

- lecturer or LMS clarifications saved as PDF, screenshot, or text with date and source;
- slide deck and exported PDF;
- report source document and final submitted PDF;
- environment lock file;
- model cards or experiment cards;
- meeting notes and peer-review records;
- AI prompt/output evidence retained according to group policy.

## 5.3 Expected repository structure

```text
Employee_Resignation_Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── Extended_Employee_Performance_and_Productivity_Data.csv
│   └── processed/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modelling.ipynb
│   └── 04_prioritization.ipynb
├── src/
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── calibrate.py
│   ├── prioritize.py
│   ├── evaluate.py
│   └── fairness.py
├── tests/
├── models/
├── results/
├── reports/
├── presentations/
└── docs/
    └── context/
        ├── Employee_Resignation_Prediction_AI_Project_Context_v4_0.md
        └── Employee_Resignation_Prediction_AI_Project_Context_v4_0.docx
```

## 5.4 Missing or mismatched file protocol

1. Stop data-dependent work if SRC-004 is absent or its checksum differs.
2. Do not search for and substitute another copy automatically.
3. Compare filename, archive contents, shape, schema, and checksum.
4. Ask the group to restore the assigned file or explicitly approve a new artefact.
5. Record a new source ID and decision if the assigned artefact officially changes.
6. If CTX-DOCX is missing, continue using CTX-MD and regenerate Word.
7. If CTX-MD is missing but Word exists, treat the package as incomplete and request the canonical Markdown file.

## 5.5 Synchronization rule

- Markdown is authoritative.
- Any material change must be made in Markdown first.
- The Word companion must then be regenerated, visually inspected, and versioned identically.
- Both files must contain the same active decisions, placeholders, current state, and source register.

## 5.6 Handoff package levels

**Conceptual understanding handoff:** The two synchronized context files are sufficient for an AI to understand all currently known project requirements, decisions, dataset facts, methodology, limitations, placeholders, and update rules because the complete formal-source transcripts are embedded.

1. `Employee_Resignation_Prediction_AI_Project_Context_v4_0.md` - canonical source.
2. `Employee_Resignation_Prediction_AI_Project_Context_v4_0.docx` - equivalent synchronized fallback.

**Data-execution handoff:** Add the exact raw CSV or original archive. Data-dependent work must stop if its checksum cannot be verified.

**Implementation/result handoff:** Add the repository, environment lock, split record, results, saved models, and latest official administrative clarifications when they exist.

Original PDFs are optional for ordinary requirement parsing because their complete normalized text is embedded. They are required when exact page appearance, annotations, or a newer/differently hashed revision must be checked. A similarly named dataset, specification, report, or notebook is not an acceptable substitute.

# 6. Dependency map

## 6.1 Technical and procedural dependencies

```text
Formal requirements
    -> approved two-stage problem framing
        -> target and positive-class definition
            -> feature policy and leakage controls
                -> reproducible split
                    -> preprocessing pipelines
                        -> baseline and model tuning on development CV
                            -> out-of-fold probability generation and calibration assessment
                                -> development-only proxy normalization and capacity/strategy freeze
                                    -> formal holdout prediction and prioritization evaluation
                                        -> error, explainability, ranking, and subgroup analysis
                                            -> conclusions, report, and viva

Exact dataset identity and checksum
    -> schema and semantic audit
        -> data-quality limitations
            -> EDA, model, proxy, and ranking validity

Usable resignation-risk signal
    -> meaningful risk-based prioritization
        -> comparison with random, value-only, and risk-only baselines

Actual intervention cost/effect evidence
    -> genuine budget optimization
        -> NOT AVAILABLE; therefore blocked and outside scope

Official lecturer/LMS clarification
    -> exact presentation and submission schedule
    -> citation and page-count rules
    -> final submission checklist

Actual AI assistance
    -> live usage record
        -> final AI Tool Usage Declaration
```

## 6.2 Blocking table

| Dependency | Blocks | Current status | Can work continue? |
| --- | --- | --- | --- |
| Exact assigned CSV and checksum | All data-dependent work | VERIFIED available | Yes |
| Approved target and problem framing | Methodology and proposal | DECIDED | Yes |
| Feature and split protocol | Formal modelling | DECIDED | Yes |
| Official exact proposal date/time | Calendar and final rehearsal scheduling | PLACEHOLDER | Yes, preparation can continue |
| All-members-must-speak rule | Final speaking allocation | PLACEHOLDER | Yes, plan for all six as the conservative practice |
| Q&A timing | Precise presentation timing | PLACEHOLDER | Yes, retain 1-2 minute delivery buffer |
| Final report deadline and viva details | Final timeline | PLACEHOLDER | Yes, technical work can continue |
| Citation style and page-limit inclusions | Final report formatting | PLACEHOLDER | Yes, draft content can continue |
| Formal model experiments | Results and conclusions | TO BE GENERATED | No final claims yet |
| Named team map | Personal accountability record | PLACEHOLDER | Yes, numbered roles are active |

# 7. Update protocol and handoff rules

## 7.1 Required update sequence

For every material change:

1. Increment the version using semantic versioning:
   - patch for wording, status, or non-methodological corrections;
   - minor for approved methodological, schema, timeline, or deliverable changes;
   - major for a target, dataset, or project-scope change.
2. Update `last_verified_date` and the document-control table.
3. Update the YAML machine-readable memory.
4. Update Section 3 current operating state.
5. Update affected decision records and placeholders.
6. Update methodology, results, risks, references, and AI-use records as applicable.
7. Add a change-log entry with evidence and approver.
8. Regenerate the Word companion from Markdown.
9. Render and visually inspect the DOCX page by page.
10. Verify that both files show the same version, title, date, decisions, and placeholders.

## 7.2 Result update rule

A result may become **VERIFIED** only when its generating code, source data checksum, split definition, environment, and output artefact are identifiable. Copying a value from an unsaved notebook cell, slide, conversation, or prior audit is insufficient for the formal result record.

## 7.3 Decision update rule

Each decision change must record:

- decision ID;
- date;
- previous status/value;
- new status/value;
- rationale;
- evidence or authority;
- dependencies and affected sections;
- approver;
- reconsideration condition.

## 7.4 Reusable handoff instruction

> Read `Employee_Resignation_Prediction_AI_Project_Context_v4_0.md` first and treat it as canonical. Follow its authority hierarchy, status labels, fixed decisions, checksum rules, and anti-hallucination constraints. Before data-dependent work, verify the exact CSV checksum. Do not invent placeholders or results. Update the live state, decision log, result record, AI-use record, version, and synchronized Word companion after material changes.

# 8. Full canonical project specification

## 8.1 Document control

| Field | Value | Status |
| --- | --- | --- |
| Project title | Design and Evaluation of Machine Learning Models for Employee Resignation Risk and Resource-Constrained Retention Prioritization | DECIDED |
| Context version | 4.0.0 | VERIFIED generated |
| Canonical filename | Employee_Resignation_Prediction_AI_Project_Context_v4_0.md | DECIDED |
| Word companion | Employee_Resignation_Prediction_AI_Project_Context_v4_0.docx | DECIDED |
| Canonical format | Markdown UTF-8 | DECIDED |
| Last verified date | 2026-08-01 | VERIFIED |
| Current phase | Proposal preparation, project setup, and approved scope update | LIVE RECORD |
| Approval record | Initial Phase 1 recommendations approved 2026-07-28; title/build update approved 2026-07-29; retention-prioritization expansion approved 2026-08-01 | DECIDED |
| Document owner | IT2011 group; personal names unresolved | PLACEHOLDER |
| Repository status | GitHub repository published; canonical URL not supplied | LIVE RECORD / PLACEHOLDER |

## 8.2 Executive project summary

The group will use the assigned `Extended_Employee_Performance_and_Productivity_Data.csv` file in a two-stage human-resources and workforce-analytics study.

**Stage 1 - Resignation-risk modelling:** Evaluate whether employee-related attributes can distinguish records marked `Resigned=True` from records marked `Resigned=False` using supervised binary classification.

**Stage 2 - Resource-constrained retention prioritization:** Under a hypothetical equal-cost capacity where HR can review or support only a limited share of records, compare several selection strategies. The primary scenario selects the top 20%; 10% and 30% are sensitivity scenarios. Strategies include repeated random selection, risk-only ranking, value-proxy-only ranking, a combined risk-times-value-proxy score, and a reference probability-threshold approach.

The value measure is an explicitly labelled **retention-value proxy**, not actual employee worth. The primary proxy is the equal-weight average of development-fitted normalized `Performance_Score`, `Monthly_Salary`, and `Years_At_Company`. Because `Monthly_Salary` is deterministically linked to job title and performance in this dataset, the project also tests a no-salary proxy and a normalized multiplicative proxy as sensitivity analyses.

The project is deliberately framed as an academic evaluation and simulation. It does not claim that a real retention intervention would prevent resignation, that the top 20% represents a real organization's budget, or that the combined ranking is financially optimal. If the resignation model has no usable signal, the project must state that the prioritization stage is unsupported.

The approved implementation uses a locked 20% stratified formal holdout, five-fold stratified development cross-validation, dummy baselines, Logistic Regression, Random Forest, calibration assessment, development-only proxy construction, pre-registered capacity scenarios, ranking-at-capacity metrics, and one-time formal-holdout corroboration. Average Precision/PR-AUC remains the primary prediction metric because only 10.01% of records are positive.

### 8.2.1 Exact assessed build

The group is building an **end-to-end, reproducible Python ML and prioritization study**, not a deployed software product. The minimum complete build consists of:

1. immutable preservation and validation of the assigned CSV;
2. reproducible data audit and EDA with visualizations;
3. leakage-safe preprocessing pipelines;
4. dummy baselines plus Logistic Regression and Random Forest;
5. hyperparameter tuning and five-fold stratified development cross-validation;
6. out-of-fold probability generation, calibration assessment, and a frozen calibrated or uncalibrated probability policy;
7. transparent development-only retention-value-proxy construction and sensitivity proxies;
8. primary top-20% capacity evaluation plus 10% and 30% sensitivity;
9. repeated-random, risk-only, value-only, combined risk-times-value, and probability-threshold strategy comparison;
10. prediction, calibration, ranking-at-capacity, uncertainty, error, interpretability, ethics, and subgroup analysis;
11. one-time formal-holdout corroboration after all design choices are frozen; and
12. saved code, configurations, predictions, metrics, figures, fitted artefacts, proposal, report, AI-use, and viva evidence.

A GUI, website, mobile app, dashboard, API, database, cloud deployment, causal intervention model, or production HR decision tool is outside scope unless an official instruction adds it. The build remains academically successful when the workflow is correct and critically evaluated, even when the formal evidence shows that resignation prediction or constrained prioritization is not useful.

## 8.3 Formal problem statement

**DECIDED**

> Evaluate whether demographic, employment, workload, performance, and satisfaction variables provide usable out-of-sample signal for `Resigned=True`; then simulate whether limited HR retention capacity can be prioritized more effectively than simple baselines by ranking records using calibrated resignation risk and a transparent retention-value proxy.

The primary capacity is 20% of records, interpreted as a hypothetical equal-cost review/support constraint. The project compares repeated random selection, risk-only ranking, value-only ranking, combined risk-times-value ranking, and a reference threshold strategy. It reports 10% and 30% capacity sensitivity when feasible.

The project must not imply that:

- the dataset represents a verified real organization;
- the model identifies causes of resignation;
- the value proxy measures actual employee worth or financial contribution;
- a selected retention intervention would prevent resignation;
- the 20% scenario is a verified real budget;
- the ranking is financially optimal;
- the system is safe or lawful for employment decisions;
- useful prediction or prioritization is guaranteed;
- a result near chance or random-selection performance is a project failure.

## 8.4 Objectives and project questions

### 8.4.1 Objectives

1. Verify and document the exact assigned dataset, schema, quality, and provenance limitations.
2. Perform reproducible EDA with appropriate visualizations.
3. Build non-informative baselines and at least two defensible supervised classifiers.
4. Tune models using development-only stratified cross-validation and leakage-safe pipelines.
5. Assess probability calibration because model probabilities feed the prioritization score.
6. Determine whether prediction performance exceeds simple baselines meaningfully.
7. Construct a transparent retention-value proxy and sensitivity alternatives without calling them actual employee worth.
8. Compare repeated random, risk-only, value-only, combined risk-times-value, and threshold strategies at a 20% capacity.
9. Test capacity sensitivity at 10% and 30% without tuning those values on the formal holdout.
10. Analyse prediction errors, selection outcomes, interpretability, subgroup behaviour, and ethical risks.
11. Produce a clear proposal, concise final report, reproducible code, and defensible viva explanation.
12. Maintain transparent records of generative-AI assistance and human verification.

### 8.4.2 Project questions

- PQ-01: Do the available predictors provide out-of-sample resignation signal beyond a dummy baseline?
- PQ-02: How do Logistic Regression and Random Forest compare under the same split, preprocessing, calibration, and metric framework?
- PQ-03: How do class weighting, calibration, and threshold choice affect minority-class precision and recall?
- PQ-04: At a fixed 20% capacity, does top-k risk ranking capture more actual positive records than repeated random selection?
- PQ-05: Does the combined risk-times-value-proxy strategy capture more positive-class proxy value than risk-only, value-only, and random strategies?
- PQ-06: Are prioritization conclusions stable at 10%, 20%, and 30% capacities and across the alternative proxy definitions?
- PQ-07: Does excluding `Gender`, or excluding both `Gender` and `Age`, materially change prediction performance or selection disparities?
- PQ-08: Are group selection rates, recall, false-positive rates, and representation materially different across gender and age groups?
- PQ-09: Which data-generation artefacts, proxy assumptions, and provenance limitations constrain interpretation?
- PQ-10: If no useful signal or lift exists, what evidence supports rejecting the prioritization concept and what additional data would be required?

## 8.5 Intended users, audience, and stakeholders

| Stakeholder | Interest | Permitted use | Not permitted |
| --- | --- | --- | --- |
| Module lecturers and assessors | Compliance, technical quality, critical evaluation, ethics | Assess proposal, implementation, report, and viva | Assume unreported work or invented results |
| Six-member student group | Learning, collaboration, marks, reproducibility | Develop and explain the two-stage study | Delegate ownership entirely to AI |
| Future AI assistants | Continuity and task execution | Use this context and verified artefacts | Override decisions or fabricate missing information |
| Hypothetical HR analyst audience | Understand prediction and resource-prioritization methodology | Interpret aggregate academic simulation findings | Treat proxy scores as actual worth or use rankings for real employee actions |
| People represented by the concept of employee data | Potential impact of risk scoring and prioritization | Benefit from ethical analysis and non-deployment safeguards | Be profiled, ranked, dismissed, denied opportunities, or surveilled by a real system based on this project |

## 8.6 Assignment requirements

| Requirement ID | Formal requirement | Project response | Status |
| --- | --- | --- | --- |
| REQ-01 | Use the assigned dataset; selection is not required. | Use exact SRC-004 and prohibit substitution. | VERIFIED / DECIDED |
| REQ-02 | Identify a real-world domain aligned with the dataset. | Human resources and workforce analytics; resignation-risk and constrained retention-prioritization simulation. | DECIDED |
| REQ-03 | Describe features, target, size, quality, and suitability. | Full audit, target imbalance, proxy-relevant variables, and limitations. | DECIDED |
| REQ-04 | Clean/preprocess data and perform EDA with visualizations. | Leakage-safe preprocessing; EDA includes target, salary structure, proxy components, and subgroup distributions. | DECIDED |
| REQ-05 | Use at least two suitable AI/ML algorithms. | Logistic Regression and Random Forest; optional Decision Tree. | DECIDED |
| REQ-06 | Implement in Python. | Python 3.11 and scikit-learn stack. | DECIDED |
| REQ-07 | Include hyperparameter tuning and model pipeline steps. | Development-only tuning, calibration assessment, proxy scaling, and reproducible prioritization code. | DECIDED |
| REQ-08 | Use relevant metrics and k-fold cross-validation. | PR-AUC primary, secondary classification metrics, five-fold stratified CV, and top-k ranking metrics. | DECIDED |
| REQ-09 | Compare models and justify best performer. | Compare prediction pipelines and downstream prioritization strategies with negative-result contingency. | DECIDED |
| REQ-10 | Discuss ethics and bias mitigation. | Sensitive-feature models, subgroup prediction/selection metrics, proxy and allocation-risk analysis, prohibited-use statement. | DECIDED |
| REQ-11 | Provide AI Tool Usage Declaration and Transparency. | Preliminary plan plus actual-use live record. | DECIDED / LIVE RECORD |
| REQ-12 | Professional final report, maximum 15 pages, PDF before viva. | Updated report plan includes prediction and prioritization without exceeding formal section order. | VERIFIED / PLACEHOLDER |
| REQ-13 | 15-minute proposal PowerPoint in lab. | Ten-slide working plan explains the dataset, real-world constraint, two-stage solution, limitations, and planned AI use. | VERIFIED / DECIDED |

## 8.7 Assessment and acceptance criteria

| Area | Marks | Acceptance criteria for this project |
| --- | --- | --- |
| Problem framing and dataset understanding | 10 | Exact source/link; correct target and balance; defensible HR problem; clear limited-capacity scenario; proxy and provenance limitations; planned AI use. |
| Preprocessing and EDA | 25 | Reproducible validation; justified transformations; no unnecessary imputation/deletion; target and proxy-component visuals; meaningful patterns and artefacts. |
| Model design and implementation | 25 | Dummy, Logistic Regression, Random Forest; pipelines; tuning; calibration assessment; reusable prioritization implementation; reproducible seeds/configuration. |
| Evaluation and comparison | 30 | Five-fold development CV; PR-AUC; formal-holdout corroboration; top-k ranking metrics; random/risk/value/combined comparisons; capacity/proxy sensitivity; honest no-signal conclusion. |
| Group report | 10 | Professional, max 15 pages, required sections, ethical and allocation-risk analysis, AI transparency, reflections, references, consistent visuals. |

**Project-level acceptance rule:** A low-scoring prediction model or a prioritization strategy that does not beat random selection is not automatically a failed project. The project succeeds when the group executes the approved protocol correctly, reports evidence honestly, explains limitations, and critically evaluates whether the proposed use should be rejected.

## 8.8 Scope boundaries

### In scope

- exact dataset and provenance audit;
- binary classification of `Resigned`;
- descriptive EDA and data-quality analysis;
- preprocessing, calibration, and validation pipelines;
- DummyClassifier, Logistic Regression, Random Forest, optional Decision Tree;
- class weighting and threshold analysis;
- development-only construction of transparent retention-value proxies;
- top-20% equal-cost capacity simulation with 10% and 30% sensitivity;
- repeated-random, risk-only, value-only, combined risk-times-value, and threshold strategy comparison;
- precision@k, recall@k, lift@k, captured-positive count, and value-weighted recall@k;
- five-fold stratified development CV and one-time formal-holdout corroboration;
- explainability, error analysis, calibration, subgroup prediction analysis, and subgroup selection analysis;
- proposal, final report, viva, reproducibility artefacts, and AI-tool disclosure.

### Out of scope unless officially required

- data collection from a real organization;
- deployment, API, web application, dashboard, database, cloud hosting, or monitoring service;
- real monetary budget optimization;
- estimation of retention-intervention effectiveness;
- replacement-cost, revenue-contribution, or true employee-value modelling;
- automated or real individual employment decisions;
- causal inference;
- external personal or sensitive datasets;
- changing the target to maximize performance;
- broad multi-target analytics;
- neural networks, SVMs, or boosting frameworks without a documented need;
- legal compliance or fairness certification.

## 8.9 Correct terminology and prohibited claims

| Use | Avoid |
| --- | --- |
| "records marked as resigned" | "employees who definitely resigned for these reasons" |
| "resignation-risk estimate" | "certainty that an employee will resign" |
| "retention-value proxy" | "employee worth", "true value", or "financial contribution" |
| "hypothetical equal-cost 20% capacity" | "the company's actual HR budget" |
| "priority score" | "optimal spending decision" |
| "records selected for simulated review/support" | "employees HR must save" |
| "potential positive-class proxy value covered" | "value protected" or "money saved" |
| "predictive association" or "out-of-sample signal" | "cause" or "driver" without causal design |
| "likely constructed or synthetic-like patterns" | "confirmed synthetic dataset" without authoritative evidence |
| "academic simulation" | "production HR or budget-allocation system" |
| "Average Precision/PR-AUC" | "accuracy proves success" |
| "subgroup disparity audit" | "legally fair", "bias-free", or automatically "not unfair" |
| "model or strategy did not exceed its baseline meaningfully" | "the project failed" |
| "salary values; unit unverified" | "USD salary" or "LKR salary" without source evidence |
| "feature importance" or "coefficient association" | "reason for resignation" |

## 8.10 Source material and provenance

| Source ID | Source | Authority and use | Verified notes |
| --- | --- | --- | --- |
| SRC-001 | `Group Assignment Specification.pdf` and Section 9.2 transcript | Highest authority for assignment requirements and marks. | Two pages; checksum recorded; complete normalized transcript embedded. |
| SRC-002 | `Proposal Presentation.pdf` and Section 9.3 transcript | Highest authority for proposal content and duration. | One page; checksum recorded; complete normalized transcript embedded; 15 minutes per group; week beginning 3 August 2026 is tentative. |
| SRC-003 | `archive.zip` | Uploaded assigned dataset package. | Contains one CSV; archive checksum recorded. |
| SRC-004 | `Extended_Employee_Performance_and_Productivity_Data.csv` | Authoritative project artefact for all data facts. | 100,000 x 20; mechanically inspected; checksum recorded. |
| SRC-005 | [Kaggle dataset page](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data) | Third-party listing for source link and licence metadata. | Listing reports CC0: Public Domain. It is not evidence of original organizational provenance. |
| SRC-006 | Conversation approval on 2026-07-28 | Authority for user/group decisions. | All Phase 1 recommendations approved. |
| SRC-007 | Conversation decision on 2026-07-29 | Authority for the previous assignment-aligned project title and clarified build definition. | User selected "Design and Evaluation of Machine Learning Models for Employee Resignation Prediction" and requested synchronized AI-only context files. |

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

Environment contract: target Python 3.11.x. Exact package versions are **TO BE GENERATED** in `requirements.txt` or `environment.yml` at repository initialization and must be saved with every formal experiment.

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
| C-001 | Final report maximum is 15 pages. | VERIFIED | Prioritize high-mark sections and concise visuals. |
| C-002 | At least two algorithms and k-fold CV are required. | VERIFIED | Use Logistic Regression, Random Forest, and five-fold CV. |
| C-003 | Exact deadlines and citation style are not supplied. | PLACEHOLDER | Obtain official confirmation. |
| C-004 | No deployment is required. | VERIFIED absence plus DECIDED scope | Do not prioritize application development. |

## 8.15 Deliberately unresolved placeholders

| Placeholder ID | Required answer | How to obtain it | Blocks current work? |
| --- | --- | --- | --- |
| GROUP-NUMBER | Official group number. | Group or LMS supplies it. | No |
| LAB-BATCH-OR-SESSION | Lab batch/session identifier. | Timetable or lecturer supplies it. | No |
| TEAM-MEMBER-NAMES | Names and student IDs mapped to Member 1-6 roles. | Group supplies identities and confirms ownership. | No |
| MODULE-COORDINATOR-OR-LECTURER | Name/title used on formal documents or clarification records. | Official module information. | No |
| REPOSITORY-PATH-OR-URL | Canonical shared repository path or URL. | Create repository and paste path. | No for documentation; yes for collaborative implementation |
| EXACT-PROPOSAL-DATE-TIME | Confirmed lab date and time. | Official LMS or lecturer/module coordinator. | No |
| ALL-MEMBERS-MUST-SPEAK | Whether each member must present. | Official lecturer clarification. | No; plan for all six meanwhile |
| Q_AND_A_TIMING | Whether questions are within or after 15 minutes. | Official lecturer clarification. | No |
| FINAL-REPORT-DEADLINE | Exact date/time and timezone. | Official LMS or assignment portal. | No |
| VIVA-DATE-AND-FORMAT | Date, duration, attendees, question style. | Official LMS or lecturer. | No |
| SUBMISSION-PLATFORM | Where files are submitted. | Official LMS or lecturer. | No |
| OFFICIAL-FILE-NAMING-CONVENTION | Required report/slides/archive names. | Official submission guidance. | No |
| REQUIRED-CITATION-STYLE | APA, Harvard, IEEE, or other. | Official rubric, lecturer, or LMS. | No for technical work |
| PAGE-LIMIT-INCLUSION-RULES | Whether title, references, appendices count. | Official lecturer clarification. | No for early drafting |
| OFFICIAL-INSTITUTIONAL-AI-OR-ACADEMIC-INTEGRITY-POLICY | Institution/module rules beyond the supplied disclosure note. | Official policy/LMS. | No; required before final submission |
| AI-GENERATED-TEXT-CITATION-RULE | Whether AI-assisted wording requires a separate citation beyond the declaration. | Official policy/lecturer. | No; required before final submission |
| SUBMISSION-ARCHIVE-STRUCTURE | Required contents/folder structure of any submitted code archive. | Submission guidance. | No |

## 8.16 Final methodology and technical decisions

| Decision ID | Active decision | Status |
| --- | --- | --- |
| D-001 | Primary target is `Resigned`; `True` is the positive class. | DECIDED |
| D-002 | Project title is "Design and Evaluation of Machine Learning Models for Employee Resignation Risk and Resource-Constrained Retention Prioritization". | DECIDED |
| D-003 | Project has two stages: resignation-risk modelling and constrained retention-prioritization simulation. | DECIDED |
| D-004 | No application or production deployment is required. | DECIDED |
| D-005 | Provenance is undocumented and likely constructed; do not claim verified real employee data. | DECIDED |
| D-006 | Exclude `Employee_ID`; audit then exclude raw `Hire_Date`; retain `Years_At_Company`. | DECIDED |
| D-007 | Use an 80/20 stratified split with seed 42; five-fold stratified development CV is primary evidence; holdout is one-time corroboration with historical exposure caveat. | DECIDED |
| D-008 | Use DummyClassifier, Logistic Regression, and Random Forest; optional Decision Tree. | DECIDED |
| D-009 | Use Average Precision/PR-AUC as the primary prediction metric. | DECIDED |
| D-010 | Use class weights and threshold analysis; no pre-split SMOTE. | DECIDED |
| D-011 | Assess probability calibration; use calibrated probabilities in priority scoring when the development evidence justifies calibration. | DECIDED |
| D-012 | Primary prioritization capacity is 20%; 10% and 30% are fixed sensitivity scenarios. | DECIDED |
| D-013 | Treat each selected record as one equal-cost capacity unit; do not claim genuine monetary budget optimization. | DECIDED |
| D-014 | Primary retention-value proxy is the equal-weight mean of development-fitted min-max normalized performance, salary, and tenure. | DECIDED |
| D-015 | Use no-salary and normalized multiplicative proxies as sensitivity analyses. | DECIDED |
| D-016 | Combined priority score is calibrated resignation probability multiplied by the selected retention-value proxy. | DECIDED |
| D-017 | Compare repeated random, risk-only, value-only, combined risk-times-value, and reference threshold strategies. | DECIDED |
| D-018 | Evaluate precision@capacity, recall@capacity, lift@capacity, captured-positive count, and value-weighted recall@capacity. | DECIDED |
| D-019 | Accept and report a defensible no-signal or no-lift outcome. | DECIDED |
| D-020 | Academic record-level ranking is permitted only inside this controlled simulation; real individual employment action is prohibited. | DECIDED |
| D-021 | Use coefficients, permutation importance, component decomposition, and error analysis; no causal claims. | DECIDED |
| D-022 | Use full, no-Gender, and optional no-Gender/no-Age models plus subgroup selection audits. | DECIDED |
| D-023 | Use reproducible Python/scikit-learn code, immutable raw data, saved split/proxy metadata, and machine-readable results. | DECIDED |
| D-024 | Plan ten proposal slides and 13-14 minutes of rehearsed content. | DECIDED |
| D-025 | Use six numbered responsibility slots with peer review until names are supplied. | DECIDED |
| D-026 | Record ChatGPT as confirmed planned and actual AI assistance. | DECIDED |
| D-027 | Administrative items remain external placeholders. | DECIDED handling |
| D-028 | Preserve embedded formal transcripts, preliminary audit diagnostics, and historical holdout-exposure limitation. | DECIDED |
| D-029 | Markdown is canonical; Word is synchronized; startup acknowledgement and stable/live precedence are required. | DECIDED |

## 8.17 End-to-end workflow and experimental protocol

1. **Acquire and verify**
   - restore the exact CSV;
   - compute SHA-256 and assert shape/schema;
   - preserve the immutable raw file.
2. **Audit semantics and quality**
   - verify types, missingness, duplicates, ranges, categories, target balance, and deterministic relationships.
3. **Create and lock the split**
   - map `Resigned=True` to 1;
   - perform a stratified 80/20 split with `random_state=42`;
   - save row IDs/split indices and split metadata;
   - lock the formal holdout against tuning, calibration choice, proxy design, capacity choice, and strategy choice;
   - disclose the historical preliminary-audit exposure caveat.
4. **Training-only EDA and design**
   - use development data for target-conditioned EDA;
   - pre-register preprocessing, searches, calibration policy, proxy formulas, capacities, strategies, and metrics.
5. **Build preprocessing pipelines**
   - exclude ID and raw date;
   - one-hot encode categorical variables;
   - scale numerical variables for Logistic Regression;
   - fit all learned steps inside development folds.
6. **Establish prediction baselines**
   - `DummyClassifier(strategy="prior")` for prevalence probability/PR-AUC reference;
   - `DummyClassifier(strategy="most_frequent")` for majority threshold metrics.
7. **Tune candidate models**
   - Logistic Regression via GridSearchCV;
   - Random Forest via bounded RandomizedSearchCV;
   - optional Decision Tree only if required work is on schedule.
8. **Generate development out-of-fold probabilities**
   - create OOF probabilities for candidate pipelines;
   - compare PR-AUC, fold stability, precision-recall behaviour, Brier score, and reliability.
9. **Choose and freeze calibration**
   - retain uncalibrated probabilities when sufficiently reliable;
   - otherwise compare a development-only calibrated variant;
   - freeze the choice before holdout use.
10. **Construct retention-value proxies on development data**
    - fit min-max scaling for proxy components on development-training data only;
    - compute primary equal-weight normalized performance-salary-tenure proxy;
    - compute no-salary and normalized multiplicative sensitivities;
    - save formulas, fitted ranges, and version IDs.
11. **Evaluate prioritization strategies on development OOF predictions**
    - repeated random selection at the same capacity;
    - top-k by risk;
    - top-k by value proxy;
    - top-k by risk-times-value proxy;
    - reference threshold strategy with actual selected fraction reported;
    - evaluate 20% primary capacity and 10%/30% sensitivities.
12. **Freeze the complete design**
    - preprocessing, model, hyperparameters, calibration, threshold, proxy formula, capacities, strategies, split ID, code commit, and package versions.
13. **Evaluate once on the formal holdout**
    - compute prediction, calibration, and prioritization metrics;
    - use bootstrap intervals where defined;
    - compare against repeated-random distributions without retuning.
14. **Analyse errors and disparities**
    - prediction false positives/negatives;
    - selected and unselected positives;
    - subgroup prediction and selection metrics;
    - full/no-Gender/no-Gender-no-Age sensitivity;
    - proxy-component and strategy decomposition.
15. **Interpret and decide**
    - determine whether prediction exceeds baselines;
    - determine whether risk-based prioritization shows stable lift over random selection;
    - reject operational use when evidence is weak, unstable, or ethically problematic.
16. **Package and report**
    - save reproducible artefacts;
    - generate tables/figures from saved results;
    - update AI-use, decision, risk, and result records.

## 8.18 Architecture, repository, data handling, and leakage controls

### 8.18.1 Component responsibilities

| Component | Responsibility | Required outputs |
| --- | --- | --- |
| `data_validation.py` | Checksum, schema, ranges, categories, duplicates, target balance, date and consistency checks | Validation report and fail-fast assertions |
| `preprocessing.py` | Feature lists, ColumnTransformer, model-specific pipelines | Serializable preprocessing objects |
| `train.py` | Split loading, CV, tuning, deterministic seeds, experiment metadata | Fitted pipelines and CV result tables |
| `calibrate.py` | OOF reliability assessment and optional development-only calibration | Calibration metrics, reliability figures, calibrated pipeline choice |
| `prioritize.py` | Proxy scaling/formulas, capacities, strategy ranking, random baselines, top-k metrics | Ranked outputs and strategy-comparison tables |
| `evaluate.py` | Formal-holdout prediction, calibration, ranking metrics, curves, error slices | Machine-readable metrics and figures |
| `fairness.py` | Sensitive-feature sensitivity and subgroup prediction/selection analysis | Fairness table with counts and caveats |
| Notebooks | Transparent audit, EDA, modelling, and prioritization interpretation | Re-executable narrative analysis |
| `tests/` | Integrity, leakage, transformation, proxy, ranking, serialization, and metric tests | Passing test report |
| `results/` | Canonical experiment outputs | JSON/CSV tables linked to commit and split |
| `models/` | Full fitted pipelines, calibrator when applicable, and metadata | joblib artefact and model card |

### 8.18.2 Leakage and integrity controls

- The formal holdout is unavailable to hyperparameter search, feature selection, calibration choice, threshold selection, proxy-formula selection, proxy weighting, capacity selection, strategy selection, or target-conditioned EDA after repository initialization.
- Historical preliminary-audit exposure remains disclosed.
- One-hot categories and scalers are fitted inside each development CV fold.
- Proxy min-max parameters are fitted on development-training data only and frozen before application to validation or formal holdout.
- `Employee_ID` is excluded from predictors and used only for traceability.
- Raw `Hire_Date` is excluded from primary modelling because it duplicates tenure and contains a fixed-time artefact.
- Formal-holdout predictions and rankings are saved only after the complete design is frozen.
- Top-k ties use a documented deterministic rule, such as descending score then ascending `Employee_ID`.
- Repeated-random baselines use fixed seeds and the same selected count as the compared strategy.
- Report tables are generated from saved result files, not manually typed.
- Raw CSV is never overwritten; processed data and proxy scores receive new filenames and metadata.
- A checksum mismatch invalidates prior results until re-audited.

## 8.19 Preprocessing and EDA plan

### 8.19.1 Preprocessing

- Assert exact schema before processing.
- Convert target Boolean to binary integer with explicit mapping.
- Parse `Hire_Date` for audit only.
- Create audit-only `Implied_Hire_Age = Age - Years_At_Company`; do not use it as a predictor without separate approval.
- Drop `Employee_ID` and raw `Hire_Date` from primary predictors.
- One-hot encode `Department`, `Gender`, `Job_Title`, and `Education_Level` with unknown-category handling.
- Treat remaining approved predictors as numerical.
- Standardize numerical features inside the Logistic Regression pipeline; scaling is not required for trees.
- Do not impute because the verified file has no missing values. Fail validation and log a new decision if a later file introduces missingness.
- Do not automatically delete bounded statistical outliers.
- Preserve class distribution; compare class weights inside model search.
- Fit proxy normalization separately from model preprocessing, using development-training data only.
- Do not include `Resigned` or model outputs in the retention-value proxy.
- Save proxy component ranges, formula version, and score provenance.

### 8.19.2 EDA questions and figures

| EDA item | Recommended output | Interpretation rule |
| --- | --- | --- |
| Dataset overview | Shape, schema, sample, type summary | Do not treat ID as a meaningful feature |
| Target distribution | Count and percentage bar chart | Annotate 89.99% majority baseline |
| Categorical predictors | Count plots and target-rate plots | Show sample sizes; avoid causal labels |
| Numerical predictors | Histograms/boxplots and target-stratified summaries | Use consistent axes; note bounded regularity |
| Date and tenure | Date histogram and tenure comparison | Highlight fixed time and redundancy |
| Salary structure | Job-title x performance salary table/heatmap | Describe deterministic mapping and unknown unit |
| Age-tenure consistency | Implied-hire-age distribution | Report implausible values; do not silently remove |
| Proxy components | Performance, salary, tenure distributions and pairwise relationships | Explain deterministic/redundant structure before proxy use |
| Proxy sensitivity | Primary/no-salary/multiplicative proxy distributions and rank correlations | Do not call a proxy "better" based only on wider spread |
| Correlations | Numeric heatmap plus target correlations | Correlation near zero supports no-signal concern but is not decisive |
| Target signal by subgroup | Target rates with sample sizes/intervals where suitable | Avoid causal or population claims |
| Scatter plots | Sampled points or density plots | Avoid unreadable overplotting |

## 8.20 Model and algorithm selection

| Model | Role | Justification | Key limitations |
| --- | --- | --- | --- |
| DummyClassifier | Mandatory baseline | Shows whether candidate models exceed prevalence/majority behaviour. | Not a learned solution; metrics depend on strategy. |
| Logistic Regression | Required linear model | Interpretable, efficient for one-hot tabular data, supports class weights and probability outputs. | Assumes additive linear log-odds; interactions must be engineered or missed. |
| Random Forest Classifier | Required non-linear ensemble | Captures interactions and non-linearity, handles mixed transformed features, provides robust tabular comparison. | Can default to majority predictions; probability calibration and importance interpretation require care. |
| Decision Tree Classifier | Optional interpretable non-linear reference | Easy to visualize and explain if time permits. | High variance and prone to overfitting. |
| Neural network, SVM, boosting libraries | Not selected | No evidence they are necessary for this dataset or assignment. | Higher complexity, runtime, and explanation burden; cannot create absent signal. |

## 8.21 Tuning and configuration plan

### 8.21.1 Global configuration

- Python target: 3.11.
- Random seed: 42 for split, CV shuffling, stochastic models, and primary random-baseline generation.
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Primary prediction scorer: `average_precision`.
- Use parallel jobs where supported and record runtime/environment.

### 8.21.2 Logistic Regression search

- solver: `liblinear` or another documented compatible solver;
- penalty: `l1`, `l2` where compatible;
- `C`: 0.01, 0.1, 1, 10;
- `class_weight`: `None`, `balanced`;
- `max_iter`: 2000;
- record convergence warnings.

### 8.21.3 Random Forest search

Use bounded `RandomizedSearchCV`:

- `n_estimators`: 200, 400, 600;
- `max_depth`: `None`, 8, 12, 20;
- `min_samples_split`: 2, 10, 30;
- `min_samples_leaf`: 1, 5, 20;
- `max_features`: `sqrt`, `log2`, 0.5;
- `class_weight`: `None`, `balanced`, `balanced_subsample`;
- `n_iter=24`, `random_state=42`.

Exact sampled combinations and selected values are **TO BE GENERATED**.

### 8.21.4 Threshold and calibration selection

- Report the default 0.5 threshold as a reference.
- Select any alternative threshold from pooled out-of-fold development predictions only.
- Predefined secondary threshold: maximize F1 on development OOF predictions.
- Report the full precision-recall trade-off.
- Evaluate Brier score and reliability curves for the selected model.
- When development evidence shows material miscalibration, compare a `CalibratedClassifierCV` variant using development data only.
- Freeze the uncalibrated/calibrated choice before formal-holdout access.
- Do not imply that F1 or calibration represents a verified business cost ratio.

### 8.21.5 Retention-value proxy configuration

Primary proxy components:

- `Performance_Score`;
- `Monthly_Salary`;
- `Years_At_Company`.

Primary formula:

```text
ValueProxy_primary = mean(
    minmax(Performance_Score),
    minmax(Monthly_Salary),
    minmax(Years_At_Company)
)
```

Rules:

- fit min/max values on development-training data only;
- apply frozen transformations to OOF/validation/holdout records;
- use equal weights because no verified business weights exist;
- do not call the result actual worth;
- record formula version and fitted ranges.

Sensitivity proxies:

1. no-salary proxy: mean of normalized performance and tenure;
2. normalized multiplicative proxy: product of epsilon-adjusted normalized performance, salary, and tenure.

### 8.21.6 Capacity and strategy configuration

- Primary selected fraction: 20%.
- Sensitivities: 10% and 30%.
- The capacity values are pre-registered and never tuned on formal-holdout performance.
- Each selected record consumes one equal-cost capacity unit.
- Combined score:

```text
PriorityScore = CalibratedResignationProbability * RetentionValueProxy
```

Strategies:

1. repeated random selection at the same `k`;
2. top-k by resignation risk;
3. top-k by retention-value proxy;
4. top-k by combined priority score;
5. reference threshold strategy, with the resulting selected fraction reported.

- Use deterministic tie-breaking.
- Use at least 1,000 repeated random selections for a stable random-reference distribution when computationally feasible.
- Do not optimize proxy formula, weights, or capacity using the formal holdout.

## 8.22 Evaluation and comparison framework

### 8.22.1 Prediction and calibration metrics

| Metric | Role | Reason and caution |
| --- | --- | --- |
| Average Precision / PR-AUC | Primary model-selection metric | Focuses on positive-class ranking; prevalence reference is 0.1001. |
| ROC-AUC | Secondary ranking metric | Threshold-independent but can appear optimistic under imbalance. |
| Precision | Threshold metric | Correctness of predicted positives. |
| Recall | Threshold metric | Share of actual positives captured. |
| F1 | Threshold metric | Balances precision and recall but hides their trade-off. |
| Balanced accuracy | Threshold metric | Balances class sensitivity. |
| Confusion matrix | Required diagnostic | Shows false positives and false negatives. |
| Accuracy | Context only | Majority-only baseline already achieves 0.8999. |
| Brier score | Required calibration metric | Measures squared probability error. |
| Reliability curve | Required calibration diagnostic | Checks whether estimated probabilities correspond to observed rates. |

### 8.22.2 Prioritization metrics

At each capacity `k`:

| Metric | Definition | Caution |
| --- | --- | --- |
| Selected count | Number of records selected | Must equal the intended top-k count except threshold strategy |
| Precision@k | Actual positives selected / selected count | Also called positive rate in the selected set |
| Recall@k | Actual positives selected / all actual positives | Measures captured positive cases |
| Lift@k | Precision@k / overall positive prevalence | Compare with random selection at the same capacity |
| Captured-positive count | Number of actual positives selected | Easy operational interpretation for the simulation |
| Value-weighted recall@k | Positive-class proxy value selected / total positive-class proxy value | Proxy value, not actual protected value |
| Random percentile or interval | Strategy result relative to repeated random selections | Use fixed seeds and equal selected counts |

### 8.22.3 Model and strategy selection rule

1. Rank model configurations by mean development-CV Average Precision.
2. Inspect fold variability, precision-recall behaviour, Brier score, and reliability.
3. Reject unstable or failed configurations.
4. Apply the one-standard-error rule to select a simpler valid pipeline when performance is statistically indistinguishable.
5. Freeze model, calibration, threshold, proxy formulas, capacities, strategies, split ID, package versions, and code commit before formal-holdout evaluation.
6. Use the formal holdout only as one-time corroboration and disclose historical audit exposure.
7. Identify the numerical best model even when all models are weak.
8. Call resignation signal potentially useful only under the predefined PR-AUC rule.
9. Call prioritization potentially useful only when risk-based strategies show stable lift over repeated random selection and do not depend solely on the value proxy.
10. Treat value-only performance as a descriptive baseline, not evidence that resignation was predicted.
11. If models remain close to dummy/random references or prioritization is unstable, conclude that the dataset does not support the proposed use.

### 8.22.4 Uncertainty

- Report prediction fold values, CV mean, standard deviation, and standard error.
- Use 2,000 stratified bootstrap resamples with seed 42 for formal-holdout prediction/ranking intervals where defined.
- Use repeated random-selection distributions for random prioritization baselines.
- Report sensitivity across 10%, 20%, and 30% capacity and across proxy definitions.
- Do not infer practical significance from tiny differences on a large constructed-like dataset.

## 8.23 Explainability and interpretability plan

- Logistic Regression: report selected coefficients, direction of association, and odds ratios only with representation/scaling explained.
- Random Forest: use permutation importance on held-out validation or formal holdout; do not rely only on impurity importance.
- Probability output: report calibration evidence before describing scores as resignation probabilities.
- Priority score: decompose each aggregate result into risk and proxy contributions; do not present individual employee narratives.
- Proxy analysis: show how performance, salary, and tenure affect the proxy and how ranks change under the no-salary and multiplicative sensitivities.
- Report importance and ranking stability across folds or sensitivity settings when feasible.
- Analyse false positives, false negatives, selected positives, selected negatives, and missed high-proxy positives.
- Never translate model importance or proxy components into causal reasons for resignation.
- Avoid SHAP unless the group can explain it correctly and the selected model shows usable signal.

## 8.24 Ethics, fairness, privacy, security, and risk analysis

### 8.24.1 Intended and prohibited use

**DECIDED intended use:** academic evaluation of resignation prediction and transparent simulation of limited-capacity prioritization on the assigned public dataset.

**DECIDED permitted record-level use:** ranking dataset records only for internal experiment evaluation, error analysis, and subgroup auditing.

**DECIDED prohibited real-world use:** hiring, dismissal, promotion, discipline, compensation, surveillance, denial of support, or any real individual employment action.

### 8.24.2 Ethical risks

| Risk | Why it matters | Control |
| --- | --- | --- |
| Direct discrimination | Gender and age may influence risk and selection. | Compare full/no-Gender/no-Gender-Age models and subgroup selection outcomes. |
| Proxy discrimination | Department, title, salary, tenure, and performance may encode structural inequality. | Discuss proxies and audit both risk and value components. |
| Worth labelling | A proxy score may be interpreted as a person's true worth. | Use explicit proxy terminology and prohibit individual narratives. |
| False positives | A non-resigning record may be selected as high priority. | Report precision@k and selected-negative counts; no real action. |
| False negatives | Actual positive records may be missed. | Report recall@k and missed-positive patterns. |
| Unequal allocation | A combined score may direct simulated resources away from certain groups. | Report selection rate, representation, recall, and value-weighted share by group. |
| Automation bias | Users may over-trust a ranking. | Prominent academic/non-production statement and uncertainty reporting. |
| Causal overreach | High risk or importance may be mistaken for a reason to intervene. | Association language only; no intervention-effect claim. |
| Budget overclaim | Top 20% may be mistaken for real financial optimization. | Call it an equal-cost capacity scenario; no monetary outcomes. |
| Provenance uncertainty | Unknown generation undermines representativeness. | Treat as likely constructed and avoid population claims. |
| No signal | Complex prioritization may appear credible despite random risk predictions. | Require prediction baselines and lift over random selection. |

### 8.24.3 Fairness and subgroup analysis

- Define `Gender` groups as Female, Male, Other.
- Use age bands 22-29, 30-39, 40-49, and 50-60 after verifying counts.
- For prediction, report sample size, prevalence, precision, recall, false-positive rate, and predicted-positive rate when minimum denominators are met.
- For prioritization, report selection rate, representation ratio, precision@capacity, recall@capacity, selected-positive count, and value-weighted selection share.
- Compare disparities for risk-only, value-only, and combined strategies to identify whether differences arise from the model or proxy.
- Compare full, no-Gender, and optional no-Gender/no-Age models.
- Do not declare a strategy fair solely because metrics are similar.
- Do not call a detected difference unfair without context, and do not call the absence of a detected difference proof of fairness.
- Do not apply advanced mitigation unless usable signal and persistent disparities justify it.

### 8.24.4 Security and privacy controls

- Store only the public assigned artefact and project outputs; do not add private employee data.
- Do not publish credentials, personal contact details, private LMS content, or team private data.
- Use `.gitignore` for environments, caches, and local secrets.
- Avoid publishing record-level ranked outputs in the public report or slides; use aggregate tables and anonymized examples.
- Validate input schema before any demonstration.
- Document the model and prioritization strategy as academic and non-production.

## 8.25 Quality assurance and testing plan

| Test ID | Test | Pass condition |
| --- | --- | --- |
| T-001 | Raw-file checksum | Matches SRC-004 exactly. |
| T-002 | Shape and schema | 100,000 rows; 20 expected columns. |
| T-003 | Target mapping | Only False/True; True maps to 1. |
| T-004 | Missingness and duplicates | Zero missing and zero exact duplicates reproduced. |
| T-005 | Identifier exclusion | `Employee_ID` absent from model features. |
| T-006 | Raw-date exclusion | `Hire_Date` absent from primary model features. |
| T-007 | Split reproducibility | Same row IDs from seed 42 and saved split logic. |
| T-008 | Stratification | Development and holdout prevalence remain close to 10.01%. |
| T-009 | Pipeline leakage | Encoders/scalers fitted only within development folds. |
| T-010 | CV configuration | Five stratified shuffled folds with seed 42. |
| T-011 | Metric orientation | `Resigned=True` positive in every metric. |
| T-012 | Baseline presence | Dummy results appear in comparison. |
| T-013 | Formal-holdout lock | No holdout access before complete design freeze. |
| T-014 | Serialization | Saved pipeline reloads and reproduces predictions. |
| T-015 | Input validation | Missing, extra, wrong-type, and unseen inputs handled explicitly. |
| T-016 | Result provenance | Every table maps to commit, split, environment, and output file. |
| T-017 | Report consistency | Figures and values match saved results. |
| T-018 | AI-use disclosure | Material AI-supported work is recorded and human-verified. |
| T-019 | Calibration provenance | Calibration is assessed/fit only with development data. |
| T-020 | Proxy leakage | Proxy scaling uses development-training data only and excludes target/model outputs. |
| T-021 | Proxy formula | Primary and sensitivity formulas reproduce expected values from test records. |
| T-022 | Capacity count | Top-k strategies select exactly the required count. |
| T-023 | Tie handling | Ties resolve deterministically and reproducibly. |
| T-024 | Random baseline | Random strategies use equal selected count, fixed seeds, and repeated draws. |
| T-025 | Ranking metrics | Precision@k, recall@k, lift@k, and value-weighted recall pass unit tests. |
| T-026 | No operational overclaim | Generated reports label values/probabilities/proxies/capacity correctly. |

### Review gates

- **Gate A - Proposal ready:** required content, two-stage framing, source, limitations, and 13-14 minute rehearsal complete.
- **Gate B - Data ready:** checksum/schema tests and audit notebook pass peer review.
- **Gate C - Modelling ready:** split locked, pipelines tested, searches and calibration plan frozen.
- **Gate D - Prioritization ready:** proxy formulas, capacities, random baselines, metrics, and fairness checks are tested and frozen.
- **Gate E - Results ready:** formal holdout used once; prediction and ranking metrics saved; exposure caveat and negative-result rules applied.
- **Gate F - Submission ready:** report/slides align with verified results; AI declaration, citations, and formatting checked.

## 8.26 Deliverable specifications

| Deliverable | Required content | Acceptance evidence |
| --- | --- | --- |
| Proposal PowerPoint | Dataset source/link, problem domain, target/quality, limited-capacity problem, two-stage solution, suitability, limitations, planned AI use | 15-minute maximum; rehearsed working plan |
| Code repository | Raw-data validation, EDA, pipelines, tuning, calibration, proxy construction, prioritization, evaluation, fairness, tests | Fresh environment reproduces outputs |
| Model artefact | Full fitted prediction pipeline plus calibrator when applicable, metadata, and threshold | Reload verification and model card |
| Prioritization artefact | Proxy configuration, capacity settings, strategy rankings, aggregate metrics, and metadata | Reproducible from saved predictions without re-tuning |
| Final report PDF | Nine required sections, max 15 pages, prediction and ranking results, ethics, AI declaration, references | Submission-ready PDF and consistency QA |
| Viva package | Project summary, two-stage methodology, results, limitations, ethics, member understanding | Question bank and shared rehearsal |
| Context Markdown | Complete AI-first project memory | UTF-8, valid YAML, current decisions/status |
| Context Word | Synchronized companion | Visual QA, headings, tables, page numbers |

## 8.27 Demonstration or prototype specification

No deployed prototype is required. If the group creates a minimal local demonstration, it may show aggregate or clearly fictional example records only. It must:

- load the saved prediction pipeline and calibrator when applicable;
- accept schema-valid example inputs;
- label outputs as academic estimates, not employment recommendations;
- show the chosen capacity and proxy formula;
- describe priority as a simulation score rather than actual worth;
- display calibration and uncertainty cautions;
- refuse missing required fields and unsupported inputs cleanly;
- avoid storing inputs;
- avoid displaying real-looking employee watchlists;
- include prominent non-production and non-employment-decision disclaimers.

**Status:** Optional and not to be prioritized before required work.

## 8.28 Final report plan

The formal maximum is 15 pages. Exact inclusion rules for title page, references, and appendices remain unresolved.

| Required section | Content plan | Priority |
| --- | --- | --- |
| 1. Introduction and Problem Statement | HR domain, resignation target, limited-capacity retention problem, two-stage evaluation, no-guarantee/no-deployment position. | High |
| 2. Dataset Description | Source/link, exact file, shape, schema, target balance, proxy components, provenance and deterministic artefacts. | High |
| 3. Preprocessing and EDA | Validation, feature policy, imbalance, proxy-component analysis, figures, quality findings. | Very high |
| 4. Model Design and Implementation | Baselines, Logistic Regression, Random Forest, pipelines, tuning, calibration, value proxies, capacities, ranking strategies, seed/split ID. | Very high |
| 5. Evaluation and Comparison | Development CV, formal holdout, calibration, top-k metrics, random/risk/value/combined comparisons, capacity/proxy sensitivity, uncertainty, errors. | Very high |
| 6. Ethical Considerations and Bias Mitigation | Prohibited real use, worth-proxy risks, subgroup prediction/selection audit, sensitive-feature sensitivity, provenance. | High |
| 7. AI Tool Usage Declaration and Transparency | Actual tools, tasks, extent, verification, ownership. | Required |
| 8. Reflections and Lessons Learned | Negative findings if applicable, proxy limitations, teamwork, modelling limits, future data needs. | Medium |
| 9. References | Formal PDFs, Kaggle listing, software/documentation, methods sources. | Required |

### Report writing rules

- Generate all tables and plots from saved results.
- Prioritize marks-heavy technical sections.
- State target prevalence and baselines prominently.
- Separate prediction quality from prioritization quality.
- Separate risk-only performance from any value-only effect.
- Never call the proxy actual employee worth or report simulated value as money saved.
- Separate audit-only diagnostics from formal results.
- Do not include guessed results or generic descriptions unrelated to implementation.
- Use the officially confirmed citation style.

## 8.29 Proposal presentation plan

### 8.29.1 Working slide structure

| Slide | Topic | Owner role | Timing guide |
| --- | --- | --- | --- |
| 1 | Title, group, module, one-sentence two-stage objective | Member 1 | 1:00 |
| 2 | Dataset source, Kaggle link, exact file and licence | Member 1 | 1:10 |
| 3 | Dataset dimensions, feature groups, target and imbalance | Member 2 | 1:35 |
| 4 | Real-world problem: resignation risk plus limited HR capacity | Member 3 | 1:30 |
| 5 | Proposed two-stage solution and comparison strategies | Member 3 or 4 | 1:45 |
| 6 | Dataset suitability for prediction and constrained prioritization | Member 4 | 1:15 |
| 7 | Quality and limitations: provenance, weak signal risk, date/tenure, deterministic salary, no intervention data | Member 5 | 2:00 |
| 8 | Planned preprocessing, models, calibration, proxy, capacity, and evaluation metrics | Member 5 | 2:00 |
| 9 | Ethics, subgroup checks, prohibited use, and planned AI-tool usage | Member 6 | 1:35 |
| 10 | Expected evidence, success/no-signal conditions, and next actions | Member 6 | 1:00 |

Target rehearsed duration: 13-14 minutes. The document does not specify a Q&A block or mandatory slide count.

### 8.29.2 Slide content rules

- Cite and link the Kaggle source directly.
- Keep the target `Resigned` explicit; do not imply a second ML target.
- Use a clear class-distribution visual.
- Explain the top 20% as a hypothetical equal-cost capacity, not actual budget.
- Call the value measure a retention-value proxy and show the formula at a high level.
- Present the combined ranking as a strategy to evaluate, not a guaranteed superior solution.
- Make the no-signal contingency visible: poor prediction makes risk-based prioritization unreliable.
- Present limitations as evidence of critical analysis.
- Do not present preliminary audit-only model scores as final results.
- Every member should understand the full pipeline.

## 8.30 Team responsibility map

| Role slot | Primary responsibility | Required peer review | Name |
| --- | --- | --- | --- |
| Member 1 | Coordination, requirements, problem framing, final integration | Member 6 | PLACEHOLDER |
| Member 2 | Dataset audit, schema, preprocessing, validation, proxy-data contract | Member 3 | PLACEHOLDER |
| Member 3 | EDA, proxy-component analysis, charts, presentation visuals | Member 2 | PLACEHOLDER |
| Member 4 | Dummy baseline, Logistic Regression, OOF probabilities, calibration | Member 5 | PLACEHOLDER |
| Member 5 | Random Forest, tuning, prioritization implementation, formal evaluation | Member 4 | PLACEHOLDER |
| Member 6 | Ethics, fairness, subgroup selection audit, AI-use record, report consistency, references | Member 1 | PLACEHOLDER |

Team rules:

- Members 4 and 5 use the same split, preprocessing, calibration, metrics, and result schema.
- Members 2, 3, and 5 use the same proxy definitions and saved scaling metadata.
- No member independently changes the target, formal holdout, positive label, proxy formula, capacity, strategy set, or primary metrics without a logged decision.
- Every major artefact receives at least one peer review.
- All six members must be able to explain prediction, calibration, prioritization, limitations, and ethics.
- Retain contribution evidence through commits, review notes, meeting records, and output ownership.

## 8.31 Timeline and milestones

| Milestone | Required completion condition | Date/status |
| --- | --- | --- |
| Context v4 and scope update | Synchronized v4 files generated and committed | COMPLETED 2026-08-01; repository commit TO BE GENERATED |
| Repository setup | Approved layout exists and raw checksum passes | COMPLETED; URL PLACEHOLDER |
| Audit notebook | Fresh-kernel execution, relative paths, assertions, outputs, peer review | IN PROGRESS |
| Proposal draft | Slides cover formal requirements and two-stage framing | TO BE GENERATED |
| Proposal rehearsal | Full delivery is 13-14 minutes and roles balanced | TO BE GENERATED before presentation |
| Proposal delivery | In-class PowerPoint during lab | Tentative week beginning 2026-08-03; exact date PLACEHOLDER |
| EDA gate | Audit tests pass and EDA/proxy diagnostics are peer reviewed | TO BE GENERATED |
| Model-development gate | Baselines and candidates tuned with development CV; calibration choice frozen | TO BE GENERATED |
| Prioritization-design gate | Proxy formulas, capacities, strategies, metrics, and tests frozen | TO BE GENERATED |
| Final evaluation gate | Frozen complete design evaluated once on formal holdout | TO BE GENERATED |
| Report complete | Nine sections, max 15 pages, consistency QA | TO BE GENERATED; deadline PLACEHOLDER |
| Viva readiness | All members explain methods/results/ethics | TO BE GENERATED; date PLACEHOLDER |

## 8.32 Current project status

- Project definition and two-stage scope: **DECIDED**.
- Dataset identity and mechanical audit: **VERIFIED**.
- Shared repository: **COMPLETED**, URL **PLACEHOLDER**.
- Audit notebook: **IN PROGRESS / FUNCTIONAL**, final reproducibility cleanup and peer review pending.
- Proposal specification: **VERIFIED**.
- Proposal slides: **TO BE GENERATED / REVISED FOR V4**.
- Formal EDA: **TO BE GENERATED**.
- Formal model and calibration experiments: **TO BE GENERATED**.
- Prioritization implementation and strategy evaluation: **TO BE GENERATED**.
- Formal holdout results and selected solution: **TO BE GENERATED**.
- Ethics/fairness execution: **TO BE GENERATED**.
- AI-use plan: **DECIDED**.
- Actual AI use: **LIVE RECORD**.
- Administrative schedule and formatting details: **PLACEHOLDER - EXTERNAL INPUT REQUIRED**.

## 8.33 Results templates - TO BE GENERATED

### 8.33.1 Experiment register

| Experiment ID | Code commit | Dataset checksum | Split ID | Pipeline/model | Calibration | Proxy/capacity | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-001 | TO BE GENERATED | SRC-004 | TO BE GENERATED | DummyClassifier | N/A | N/A | TO BE GENERATED |
| EXP-002 | TO BE GENERATED | SRC-004 | TO BE GENERATED | Logistic Regression | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| EXP-003 | TO BE GENERATED | SRC-004 | TO BE GENERATED | Random Forest | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| EXP-004 | TO BE GENERATED | SRC-004 | TO BE GENERATED | Decision Tree optional | TO BE GENERATED | TO BE GENERATED | OPTIONAL |

### 8.33.2 Cross-validation comparison

| Model | Best parameters | Mean PR-AUC | Std PR-AUC | Mean ROC-AUC | Mean F1 | Brier | Calibration note | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | N/A | TO BE GENERATED |
| Logistic Regression | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| Random Forest | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |

### 8.33.3 Formal-holdout prediction evaluation

| Selected pipeline | Calibration | Threshold | PR-AUC | ROC-AUC | Balanced accuracy | Precision | Recall | F1 | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |

### 8.33.4 Prioritization strategy comparison at 20%

| Strategy | Proxy | Selected N | Precision@20% | Recall@20% | Lift@20% | Captured positives | Value-weighted recall@20% | Random percentile/interval |
| --- | --- | ---:| ---:| ---:| ---:| ---:| ---:| --- |
| Repeated random | N/A | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | 1.0 expected | TO BE GENERATED | TO BE GENERATED | Reference |
| Risk only | N/A | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| Value only | Primary | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| Risk x value | Primary | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| Threshold reference | N/A | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |

### 8.33.5 Capacity and proxy sensitivity

| Capacity | Proxy version | Strategy | Precision@k | Recall@k | Lift@k | Value-weighted recall@k | Stability note |
| ---:| --- | --- | ---:| ---:| ---:| ---:| --- |
| 10% | Primary / no-salary / multiplicative | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| 20% | Primary / no-salary / multiplicative | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| 30% | Primary / no-salary / multiplicative | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |

### 8.33.6 Confusion matrix

| Actual / Predicted | Predicted False | Predicted True |
|---|---:|---:|
| Actual False | TO BE GENERATED | TO BE GENERATED |
| Actual True | TO BE GENERATED | TO BE GENERATED |

### 8.33.7 Fairness and subgroup template

| Model/strategy | Group | N | Positive prevalence | Selection rate | Precision@capacity | Recall@capacity | False-positive rate | Representation ratio | Caveat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Risk only | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| Value only | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| Risk x value | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |

### 8.33.8 Explainability and error analysis template

- Selected model and calibration: **TO BE GENERATED**
- Stable important predictors: **TO BE GENERATED**
- Proxy formula and component influence: **TO BE GENERATED**
- Rank correlation across proxies: **TO BE GENERATED**
- False-positive and false-negative patterns: **TO BE GENERATED**
- Selected-negative and missed-positive patterns: **TO BE GENERATED**
- Calibration finding: **TO BE GENERATED**
- Causal claims permitted: **None**
- Actual employee-worth claim permitted: **None**

### 8.33.9 Final conclusion template

- Evidence of useful predictive signal: **TO BE GENERATED**
- Best justified prediction model or baseline: **TO BE GENERATED**
- Evidence of lift over random prioritization: **TO BE GENERATED**
- Best supported prioritization strategy, if any: **TO BE GENERATED**
- Capacity/proxy sensitivity: **TO BE GENERATED**
- Real-world deployment recommendation: **TO BE GENERATED**, with default presumption against use unless strong, stable, ethical evidence exists.
- Claims about intervention success or money saved: **Not permitted from this dataset**
- Main dataset and proxy limitations: **TO BE GENERATED from verified/formal findings**
- Recommended future data: **TO BE GENERATED**
- Lessons learned: **TO BE GENERATED**

## 8.34 AI-assistant rules for code and writing

### Code rules

- Use functions, type hints, docstrings, validation, explicit seeds, and deterministic paths.
- Prefer scikit-learn `Pipeline` and `ColumnTransformer`.
- Never fit preprocessing, calibration, or proxy scaling on the full dataset before evaluation.
- Keep target mapping explicit and tested.
- Keep risk score, value proxy, and combined priority score as separate named fields.
- Save machine-readable prediction, calibration, ranking, random-baseline, proxy, capacity, and experiment metadata.
- Use deterministic top-k tie-breaking.
- Do not suppress warnings without explanation.
- Do not alter the raw CSV.
- Avoid unapproved dependencies.
- Include tests for major pipeline and ranking components.

### Writing rules

- Tie claims to source IDs, code outputs, or saved results.
- Keep **VERIFIED** facts, **DECIDED** methods, and **TO BE GENERATED** results separate.
- Use precise language about imbalance, calibration, uncertainty, proxy meaning, and no-signal outcomes.
- Never call the proxy actual employee worth or the 20% scenario a verified budget.
- Never claim that selected support would prevent resignation or save money.
- Separate risk-only, value-only, and combined-strategy findings.
- Do not present feature importance or proxy components as causality.
- Do not conceal dataset artefacts or weak results.
- Match values across report, slides, README, model card, and context.

## 8.35 AI-tool plan and actual-use live record

### 8.35.1 Preliminary AI-tool plan

| Tool | Planned tasks | Estimated extent | Verification and ownership |
| --- | --- | --- | --- |
| ChatGPT | Requirement interpretation, context architecture, dataset-audit assistance, problem-framing review, code/debugging assistance, EDA and visualization review, report and slide editing, viva preparation | Moderate; update if actual use changes | Group checks formal sources, executes code, reviews outputs, rewrites where necessary, and retains responsibility. |

Do not list Copilot, Gemini, Claude, or another tool until the group actually plans to use it. Add each tool as a separate record with version, tasks, extent, verification, and retained evidence.

### 8.35.2 Actual AI-tool usage - LIVE RECORD

| Record ID | Date | Tool/version | Task supported | Extent | Verification performed | Human ownership/status |
| --- | --- | --- | --- | --- | --- | --- |
| AI-001 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Interpreted assignment/proposal requirements and planned presentation/report structure. | Moderate | Checked against formal PDFs. | User reviewed; group responsible. |
| AI-002 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Audited archive/CSV and identified schema, checksums, class balance, deterministic relationships, and provenance limitations. | Moderate | Code-based inspection and source review. | User approved recommendations. |
| AI-003 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Recommended target, methodology, models, metrics, ethics, team roles, repository, and no-signal contingency. | Moderate | Tied to rubric and audit evidence. | User approved. |
| AI-004 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Generated canonical Markdown and synchronized Word context files. | Moderate | YAML/content checks and DOCX rendering. | Group owns maintenance. |
| AI-005 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Expanded context to v3 with embedded transcripts, data contract, holdout caveat, evaluation/fairness rules, and synchronization protocol. | Extensive | Validated YAML/headings/checksums and rendered Word. | User requested update. |
| AI-006 | 2026-07-29 | ChatGPT - GPT-5.6 Thinking | Updated title and exact assessed-build definition in v3.1. | Moderate | Content checks and Word rendering. | User approved. |
| AI-007 | 2026-07-28 to 2026-08-01 | ChatGPT - GPT-5.6 Thinking | Guided Jupyter audit notebook, result exports, chart creation, GitHub setup, and collaboration workflow. | Extensive | User executed code and supplied screenshots/results. | User performed steps; group retains responsibility. |
| AI-008 | 2026-08-01 | ChatGPT - GPT-5.6 Thinking | Evaluated and formalized the new resignation-risk plus constrained retention-prioritization idea; defined capacity, proxy, strategy, metrics, safeguards, and updated synchronized context v4 files. | Extensive | Rechecked official assignment text, preserved dataset evidence, validated Markdown/YAML, generated and rendered Word companion. | User explicitly requested the scope update; group owns approval and implementation. |

Future records must include actual tool/version if known, precise tasks, extent, verification, retained evidence, and group responsibility.

## 8.36 Decision and change log

| Decision ID | Date | Status | Topic | Active decision | Evidence/authority | Approver | Reconsideration condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q-001 to Q-022 | 2026-07-28 | DECIDED | Initial target, methodology, evaluation, ethics, handoff, and administrative handling | Retained except where explicitly superseded by Q-024 to Q-030 | Formal requirements, dataset evidence, user approval | User | Higher-authority instruction or logged change |
| Q-023 | 2026-07-29 | SUPERSEDED IN PART | Project title and exact assessed build | Previous title focused on resignation prediction alone; non-deployment scope retained | Explicit user decision | User | Superseded by Q-024 title/scope expansion |
| Q-024 | 2026-08-01 | DECIDED | Project scope and title | Use "Design and Evaluation of Machine Learning Models for Employee Resignation Risk and Resource-Constrained Retention Prioritization"; retain `Resigned` target and add a constrained prioritization simulation | User request and assignment alignment | User | Higher-authority instruction or explicit logged change |
| Q-025 | 2026-08-01 | DECIDED | Capacity | Primary top-20% equal-cost capacity; 10% and 30% sensitivity; do not call it a verified real budget | User concept plus methodological review | User | Verified real cost/capacity evidence or approved redesign |
| Q-026 | 2026-08-01 | DECIDED | Retention-value proxy | Primary equal-weight normalized performance-salary-tenure proxy; no-salary and normalized multiplicative sensitivity | User concept plus dataset artefact review | User | Approved proxy redesign before holdout use |
| Q-027 | 2026-08-01 | DECIDED | Prioritization strategies | Compare repeated random, risk-only, value-only, risk-times-value, and threshold reference | Methodological comparison need | User | Approved pre-holdout protocol change |
| Q-028 | 2026-08-01 | DECIDED | Ranking evaluation | Use precision@k, recall@k, lift@k, captured-positive count, value-weighted recall@k, random-reference distribution, and capacity sensitivity | Resource-constrained ranking methodology | User | Higher-authority or approved pre-holdout change |
| Q-029 | 2026-08-01 | DECIDED | Operational and causal limits | Academic simulation only; no actual worth, real budget optimization, intervention-effect, money-saved, or real employment-action claims | Dataset limitations and ethics | User | New verified data plus formal approval |
| Q-030 | 2026-08-01 | DECIDED | Calibration and downstream validity | Assess calibration; weak prediction or no lift invalidates risk-based prioritization claims | Probability use and no-signal integrity | User | Approved methodological change before holdout |

### Change log

| Version | Date | Change | Rationale | Status |
|---|---|---|---|---|
| 2.0.0 | 2026-07-28 | Created AI-first canonical context and synchronized Word companion. | Complete handoff and fixed methodology. | SUPERSEDED |
| 3.0.0 | 2026-07-28 | Embedded formal transcripts; expanded live state/data contract; documented exposure; fixed evaluation, fairness, and synchronization rules. | Make handoff self-contained and execution-safe. | SUPERSEDED |
| 3.1.0 | 2026-07-29 | Adopted the assignment-aligned title "Design and Evaluation of Machine Learning Models for Employee Resignation Prediction" and exact-build definition. | Clarify required ML workflow and non-deliverables. | SUPERSEDED |
| 4.0.0 | 2026-08-01 | Expanded the project to two stages: resignation-risk modelling and resource-constrained retention-prioritization simulation; added capacity/proxy/strategy/calibration/ranking metrics, sensitivity, ethics, repository, report, presentation, test, result, decision, and AI-use updates; regenerated synchronized Word companion. | Connect the ML target to a realistic limited-resource problem while preserving methodological and ethical limits. | ACTIVE |

## 8.37 Remaining questions requiring human or external input

| ID | Question | Recommended handling | Why not auto-decided | How to confirm | Blocks current work? |
| --- | --- | --- | --- | --- | --- |
| EXT-001 | Exact proposal date and lab time | Use official value; retain tentative week meanwhile. | Only lecturer/LMS can set schedule. | LMS/lecturer. | No |
| EXT-002 | Must all six members speak? | Plan for all six until confirmed. | Formal document silent. | Lecturer. | No |
| EXT-003 | Is Q&A within or after 15 minutes? | Keep 1-2 minute delivery buffer. | Formal document silent. | Lecturer. | No |
| EXT-004 | Final report deadline and timezone | Use official portal value only. | Not supplied. | LMS/portal. | No |
| EXT-005 | Viva date, format, and duration | Use official value only. | Not supplied. | LMS/lecturer. | No |
| EXT-006 | Submission platform and file naming | Use official instructions. | Not supplied. | LMS/portal. | No |
| EXT-007 | Citation style | Request official style. | Unsafe to assume. | Rubric/LMS/lecturer. | No for development |
| EXT-008 | Whether title/references/appendices count in 15 pages | Use official clarification. | Assessment authority decision. | Lecturer. | No |
| EXT-009 | Group number, lab batch, names/IDs, role mapping | Fill after group agreement and official confirmation. | Cannot be inferred. | Group/LMS. | No |
| EXT-009A | Lecturer/coordinator identity | Use official module information. | Not in supplied PDFs. | LMS/module page. | No |
| EXT-010 | Canonical repository URL | Paste the published GitHub URL. | Repository exists but URL was not supplied. | Group/GitHub. | No |
| EXT-011 | Institutional AI/academic-integrity policy and AI-text citation rule | Apply official policy. | Not in supplied evidence. | Institution/LMS. | No for development; yes before submission |
| EXT-012 | Required submission archive structure | Use official guidance. | Not supplied. | LMS/portal. | No |
| EXT-013 | Does the lecturer accept the expanded retention-prioritization framing for the proposal? | Present it as an extension of the same `Resigned` target and ask for confirmation if practical. | The assignment permits aligned real-world framing but does not mention this specific extension. | Lecturer/module coordinator. | No for drafting; confirmation recommended before final scope lock |

## 8.38 References and source-file register

### Formal and project sources

1. **SRC-001:** Faculty of Computing. *IT2011 - Artificial Intelligence and Machine Learning: Group Assignment Specification*. Year 2 Semester 1, 2026. Local file: `Group Assignment Specification.pdf`. SHA-256: `7bcd43f095c4e76d89d941ee6c28da228260b6cebb70ae56c623e6d55867b481`.
2. **SRC-002:** Faculty of Computing. *IT2011 - Artificial Intelligence and Machine Learning: Proposal Presentation*. Year 2 Semester 1, 2026. Local file: `Proposal Presentation.pdf`. SHA-256: `7cbf695c122b194880beb62a672b15eb56eaa470d43bf286a3794e9c10abb4d3`.
3. **SRC-003:** Uploaded assigned archive, `archive.zip`. SHA-256: `abe4a9aa869fe553faec813a06ff93432900d72af2036058f8b84e3dd64db375`.
4. **SRC-004:** `Extended_Employee_Performance_and_Productivity_Data.csv`. SHA-256: `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb`.
5. **SRC-005:** Kaggle. *Employee Performance and Productivity Data*. Dataset page: [Kaggle dataset page](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data). Listing observed as CC0: Public Domain.
6. **SRC-006:** User approval in project conversation, 2026-07-28: approval of initial recommendations.
7. **SRC-007:** User title/build update decision, 2026-07-29: use "Design and Evaluation of Machine Learning Models for Employee Resignation Prediction" and clarify the assessed build.
8. **SRC-008:** User scope-expansion decision, 2026-08-01: retain `Resigned` and add resource-constrained retention prioritization using risk and a transparent value proxy.

### Reference maintenance rules

- Add formal method and software references when the implementation is written.
- Prefer official scikit-learn documentation and primary methodological sources.
- Record access dates for web sources in the final reference style.
- Do not cite this context file as evidence for a fact when the underlying formal source or dataset can be cited directly.
- Replace provisional citation formatting with the officially confirmed style before final submission.

# 9. Embedded official source transcripts

## 9.1 Transcription and authority policy

The following are complete normalized-text transcriptions of the supplied formal PDFs. Line wrapping, bullet glyphs, and page decoration have been normalized for reliable AI parsing; wording and page boundaries are preserved substantively.

- Source file hashes are recorded in Sections 1 and 8.38.
- These transcripts are sufficient for understanding and applying the supplied written requirements when the original PDFs are unavailable.
- The original PDFs remain authoritative for exact visual layout, stamps, signatures, handwritten annotations, or verification of a newer/differently hashed revision.
- If a new official instruction conflicts with these transcripts, apply the authority hierarchy and record the change.

## 9.2 SRC-001 - Group Assignment Specification

### Page 1 of 2

**Faculty of Computing**  
**Year 2 Semester 1 (2026)**  
**IT2011 - Artificial Intelligence and Machine Learning**  
**Group Assignment Specification**

**Group Size:** 6 Members

**Assignment Title:**  
Design, Implement, and Evaluate AI/ML Solutions for a Real-World Problem, Using a given Dataset

**Objective**  
To design and implement an end-to-end AI/ML solution for a real-world problem using suitable supervised or unsupervised learning algorithms, applied to a dataset assigned to your group, and to critically evaluate the performance and ethical implications of the chosen approach.

**Learning Outcomes Addressed:**

- LO3: Apply AI/ML techniques to solve real-world problems.
- LO4: Critically evaluate the effectiveness and impact of AI/ML algorithms.

**Assignment Components:**

**Step 1: Assigned Dataset Understanding and Problem Framing (10 Marks)**

- Each group will be provided with a specific dataset by the module coordinators. Dataset selection is not required this year; your task is to understand, describe, and justify the suitability of your assigned dataset.
- Identify a real-world problem domain (for example, healthcare, finance, agriculture, education) that aligns with the assigned dataset.
- Describe and justify the dataset's suitability (features, labels/target variable, size, and quality).
- Briefly state which AI tools, if any, you plan to use during the project and for what purpose (see the AI Tool Usage note below).

**Step 2: Data Preprocessing and Analysis (25 Marks)**

- Clean and preprocess the assigned dataset (for example, missing values, outliers, encoding).
- Perform exploratory data analysis with visualizations.
- Provide insights and patterns observed in the data.

### Page 2 of 2

**Step 3: Model Design and Implementation (25 Marks)**

- Select and justify at least two AI/ML algorithms (for example, SVM, Decision Trees, K-Means, Neural Networks) suitable for the assigned dataset.
- Implement models using Python (Pytorch, TensorFlow, etc.).
- Include hyperparameter tuning and model pipeline steps.

**Step 4: Model Evaluation and Comparison (30 Marks)**

- Use relevant performance metrics (Accuracy, F1 Score, Confusion Matrix, etc.).
- Perform k-fold cross-validation.
- Compare models and justify the best performer.

**Step 5: Group Report (10 Marks)**

**a) Prepare a professional report (max 15 pages):**

1. Introduction and Problem Statement
2. Dataset Description (assigned dataset: source, structure, and characteristics)
3. Preprocessing & EDA
4. Model Design and Implementation
5. Evaluation and Comparison
6. Ethical Considerations and Bias Mitigation
7. AI Tool Usage Declaration and Transparency (new)
8. Reflections and Lessons Learned
9. References

**b)** Submit your report in PDF format before the viva session.

**c)** Use of visuals, teamwork, and clarity will be evaluated.

**AI Tool Usage Declaration - what to include**

We would like each group to be transparent about how generative AI tools (for example, ChatGPT, Claude, Copilot, Gemini, etc.) were used during the project. In the dedicated report section, include a short table or write-up covering, for each tool used:

- The tool name and version, if known.
- The specific aspects it supported - for example, code debugging, EDA/visualization ideas, hyperparameter search, writing/editing the report, generating explanations, brainstorming problem framing, etc.
- The extent of use for each aspect (for example, minimal, moderate, extensive).
- How the group verified, tested, or took ownership of AI-assisted outputs before including them.

Groups that used no AI tools should simply state this clearly. Honest disclosure is what is being assessed here, not the amount of AI use itself.

## 9.3 SRC-002 - Proposal Presentation

### Page 1 of 1

**Faculty of Computing**  
**Year 2 Semester 1 (2026)**  
**IT2011 - Artificial Intelligence and Machine Learning**  
**Proposal Presentation**

**1. Assigned Dataset**

- Your group will be assigned a specific open-source dataset by the module coordinators (dataset selection is not required).
- Note the dataset source and provide a direct link/citation to it in your slides.
- Review the dataset's structure (features, target variable, size, and format) before the presentation.

**2. Problem Domain Identification**

- Clearly define the real-world problem you intend to address using the assigned dataset (for example, healthcare, finance, agriculture, education).
- Ensure strong alignment between the assigned dataset and the problem domain you propose.

**3. Dataset Justification**

- Provide a clear analysis of the assigned dataset's suitability for your chosen problem.
- Discuss features, target variable, size, and quality.
- Note any limitations of the dataset and how your group intends to address them (optional, but encouraged).

**4. Planned AI Tool Usage**

- Briefly outline which generative AI tools, if any, your group plans to use over the course of the project and for which tasks (for example, coding assistance, EDA interpretation, report drafting).
- This is a preliminary plan - the final AI Tool Usage Declaration in the group report should reflect what was actually used.

**Proposal Presentation Details**

- Format: Slide-based presentation (PowerPoint)
- Duration: 15 minutes per group
- Mode: In-class during Lab Sessions
- Date and Time: During lab sessions of the 1st Week of August (starting week - 3rd August 2026; tentative)
- Slides Should Include: Assigned Dataset (source + link); Problem Domain (real-world context); Dataset Justification (features, labels, size, quality); Planned AI Tool Usage (brief)

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

