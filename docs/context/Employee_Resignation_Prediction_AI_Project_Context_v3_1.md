---
title: "Design and Evaluation of Machine Learning Models for Employee Resignation Prediction"
subtitle: "Canonical project specification, live state, and AI handoff memory"
author: "IT2011 Project Group"
date: "2026-07-29"
version: "3.1.0"
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

This file is the persistent AI-first project memory for **Design and Evaluation of Machine Learning Models for Employee Resignation Prediction**. It is intended to let a new AI assistant, team member, reviewer, or future project session understand the project without relying on conversational memory.

**VERIFIED** - The project is an IT2011 group assignment for six members. It requires an end-to-end AI/ML solution, at least two suitable algorithms, Python implementation, hyperparameter tuning, pipelines, k-fold cross-validation, model comparison, ethical evaluation, a 15-minute proposal presentation, and a final report of no more than 15 pages.

**DECIDED** - The primary task is binary classification of `Resigned`, framed as evaluating whether the available variables contain usable predictive signal. The project does not assume that a useful operational model will result.

**DECIDED** - The assessed build is an end-to-end Python machine-learning workflow and reproducibility package: data validation, preprocessing, EDA, baseline construction, at least two trained classifiers, hyperparameter tuning, k-fold evaluation, model comparison, ethical analysis, saved results, proposal material, and report evidence. A website, mobile application, dashboard, API, database, cloud deployment, or production HR system is not required unless a higher-authority instruction changes the scope.

## 0.2 Canonical-source rule

1. This Markdown file is the canonical source of truth for project context.
2. The Word file named `Employee_Resignation_Prediction_AI_Project_Context_v3_1.docx` is a synchronized companion generated from this content.
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
- [ ] The formal 20% holdout is unavailable to current tuning and threshold selection after repository initialization.
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
- model scores, selected hyperparameters, confidence intervals, fairness findings, feature rankings, or conclusions before execution;
- claims that a model is suitable for employment decisions;
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
- A scope-changing proposal must identify the reason, benefits, cost, risks, affected deliverables, and approval authority.
- Do not add a dashboard, application, API, deployment, neural network, external boosting framework, or additional target merely because it is technically possible.
- A target pivot after viewing formal holdout results is prohibited. A justified pivot before formal holdout evaluation requires a logged group or lecturer decision.
- If the Word companion is missing, outdated, or inconsistent, continue using this Markdown file and report that regeneration is required.

## 0.10 Mandatory AI startup acknowledgement

Before performing project-dependent work, an AI must provide a compact **Context acknowledgement** containing:

1. project objective;
2. current phase and immediate milestone;
3. target column and positive class;
4. required models and primary metric;
5. formal holdout restriction and the historical preliminary-audit exposure caveat;
6. principal provenance or data-quality limitation;
7. prohibited operational use;
8. unresolved placeholders relevant to the requested task;
9. files or evidence currently available; and
10. the next action it will perform.

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
  context_schema_version: '3.1'
  project_context_version: 3.1.0
  synchronization_id: ERPC-20260729-V3.1
  generated_at: 2026-07-29 Asia/Colombo
  canonical_file: Employee_Resignation_Prediction_AI_Project_Context_v3_1.md
  word_companion_file: Employee_Resignation_Prediction_AI_Project_Context_v3_1.docx
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
  name: Design and Evaluation of Machine Learning Models for Employee Resignation Prediction
  slug: Employee_Resignation_Prediction
  type: academic AI/ML classification and critical evaluation project
  domain: human resources and workforce analytics
  module:
    code: IT2011
    name: Artificial Intelligence and Machine Learning
    faculty: Faculty of Computing
    academic_period: Year 2 Semester 1 (2026)
  team_size: 6
  primary_objective: Evaluate whether the assigned employee dataset contains usable out-of-sample signal for distinguishing
    records marked Resigned=True from Resigned=False.
  problem_statement: Use supervised binary classification to evaluate predictive association without claiming causality,
    operational readiness, legal compliance, or verified real-employee provenance.
  build_definition:
    primary_build: End-to-end Python machine-learning workflow for binary classification of Resigned.
    required_components:
    - immutable raw-data preservation and checksum/schema validation
    - reproducible preprocessing and exploratory data analysis
    - DummyClassifier baseline
    - Logistic Regression and Random Forest training pipelines
    - hyperparameter tuning within five-fold stratified development cross-validation
    - threshold analysis and one-time formal-holdout corroboration
    - model comparison, error analysis, interpretability, ethics, and subgroup audit
    - saved code, configurations, results, figures, model artefacts, proposal content, report evidence, and viva preparation
    explicitly_not_required:
    - website or mobile application
    - dashboard or graphical user interface
    - API, database, cloud service, or production deployment
    - automated or individual employment decision support
    success_definition: Correct, reproducible, critically evaluated execution satisfies the project even if no model materially exceeds the dummy baseline.
live_state:
  status: active
  current_phase: proposal preparation and project setup
  current_milestone: 15-minute proposal presentation during lab sessions in the week beginning 2026-08-03; exact
    date and time unconfirmed
  last_updated: '2026-07-29'
  updated_by: ChatGPT with user-approved project decisions
  evidence:
  - SRC-001
  - SRC-002
  - SRC-004
  - SRC-006
  - SRC-007
  completed:
  - formal requirement extraction
  - dataset audit and data contract
  - methodology and scope decisions
  - context v3 generation
  - assignment-aligned title and exact build-definition update
  active_next_actions:
  - create shared repository
  - preserve raw CSV and verify checksum
  - reproduce data audit in code
  - draft and rehearse proposal
  - obtain official administrative confirmations
  blockers:
  - team names and repository URL not supplied
  - exact dates/submission rules/citation style not officially supplied
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
  authority: explicit user decision to use the new title and regenerate the two-file AI handoff
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
  calibration_rule: Brier score and reliability curve are mandatory if outputs are described as probabilities or
    risk estimates; otherwise they are optional diagnostics and no calibration claim is allowed.
  usable_signal_rule: The best model may be identified even when weak. Call signal potentially useful only when
    CV Average Precision exceeds the prior baseline consistently, the absolute mean improvement is at least 0.01,
    uncertainty is not dominated by baseline performance, and no serious subgroup failure invalidates the claim.
    This is a project rule, not a universal industry threshold.
  no_signal_rule: If models remain near baseline or unstable, report a defensible negative result and reject operational
    use.
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
  insufficient_sample_rule: When thresholds are not met, report counts/prevalence only and label rate estimates
    unstable or not reported.
  sensitivity_models:
  - full model
  - no-Gender model
  - no-Gender/no-Age model if required model work is complete
  interpretation: Subgroup similarity does not prove fairness or legal compliance.
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
- name: Employee_Resignation_Prediction_AI_Project_Context_v3_1.md
  format: Markdown
  status: canonical project context
- name: Employee_Resignation_Prediction_AI_Project_Context_v3_1.docx
  format: DOCX
  status: synchronized companion with same substantive content
fixed_decisions:
- Primary task is binary resignation classification framed as evaluation of predictive signal.
- Assessed build is the end-to-end Python ML workflow and evaluation package, not a website, dashboard, API, or deployed HR product.
- No application, dashboard, API, or deployment is required unless official instructions change scope.
- Employee_ID and raw Hire_Date are excluded from primary modelling; Years_At_Company is retained.
- Class imbalance is handled primarily through class weights and threshold analysis; no pre-split resampling.
- Five-fold development CV is primary comparison evidence; the formal holdout is corroborative and carries the historical
  audit-exposure caveat.
- A valid negative result is acceptable and must be reported honestly.
- Project is academic only and must not support individual employment decisions.
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
startup_acknowledgement:
  required_before_project_dependent_work: true
  fields:
  - project objective
  - current phase
  - target and positive class
  - required models and primary metric
  - formal holdout restriction and historical exposure caveat
  - principal provenance/data-quality limitation
  - prohibited operational use
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

**What is being created:** An academically defensible, reproducible binary-classification study using the assigned employee dataset, plus a proposal presentation, implementation, model comparison, ethics/fairness analysis, final report, and viva preparation.

**Why:** The assignment requires the group to apply AI/ML methods to a real-world problem and critically evaluate effectiveness and impact.

**Current phase:** Proposal preparation and project setup.

**Immediate objective:** Present the assigned dataset, resignation-prediction problem, suitability, limitations, planned methodology, and planned AI-tool use in a 15-minute group PowerPoint.

**Core inputs:** The embedded formal-source transcripts, exact uploaded CSV, Kaggle listing, approved decisions in this context file, and later code/results. Original PDFs remain useful for exact visual or revision verification but are not required merely to understand the written requirements.

**Primary output:** Evidence on whether the available variables distinguish `Resigned=True` from `Resigned=False` out of sample.

**Fixed methodology:** Stratified 80/20 split with seed 42; five-fold stratified CV on the 80% development partition as primary comparison evidence; prior and most-frequent dummy baselines; Logistic Regression and Random Forest; PR-AUC as primary metric; and one-time formal-holdout corroboration after model/threshold freeze. Because preliminary audit split metadata was not retained, the holdout is not described as historically pristine.

**Main constraints:** Six members; at least two algorithms; Python; tuning; pipeline; k-fold CV; maximum 15-page final report; no unsupported operational or causal claims.

**Critical data facts:** 100,000 rows, 20 columns, no missing values, no duplicate rows, 10.01% positive target, redundant hire-date/tenure information, and deterministic salary structure by job title and performance score.

**Current blocker:** Exact administrative schedule and submission rules are not yet confirmed. Technical preparation is not blocked.

**Next task:** Create the repository, reproduce the data audit, draft the ten-slide proposal, and obtain official administrative confirmations.

## 2.2 Critical terminology

| Term | Project-specific meaning |
| --- | --- |
| Positive class | `Resigned=True`. |
| PR-AUC / Average Precision | Primary ranking metric for the imbalanced positive class. |
| Formal holdout | The 20% stratified partition locked against future tuning and used once after model/threshold freeze. It is corroborative and carries a historical preliminary-audit exposure caveat. |
| Development CV | Five-fold StratifiedKFold performed only on the 80% development-training partition; this is the primary model-comparison evidence. |
| Predictive signal | Out-of-sample association that materially exceeds a suitable baseline; not causality. |
| Likely constructed data | An inference based on deterministic and highly regular patterns; not an authoritatively verified provenance statement. |
| Fairness audit | Subgroup performance analysis and sensitive-feature sensitivity tests; not a declaration of legal fairness or compliance. |

## 2.3 Items an AI must never assume

- that the employees are real individuals;
- that `Monthly_Salary` is in USD, LKR, or any other currency;
- that a high accuracy value indicates useful resignation prediction;
- that class weighting or resampling creates missing target signal;
- that correlation or model importance identifies why employees resign;
- that every member must speak, or that Q&A is included, unless officially confirmed;
- that the current tentative proposal week is an exact date;
- that a model should be deployed or used for individual HR action.

# 3. Current operating state

## 3.1 Current milestone

**VERIFIED** - The proposal is a 15-minute PowerPoint presentation per group, delivered in class during lab sessions. The formal proposal sheet gives the week beginning 3 August 2026 as tentative; exact date and time remain unresolved.

## 3.2 Immediate objectives

1. Convert the approved project framing into a ten-slide proposal.
2. Reproduce the dataset audit from code in the shared repository.
3. Allocate six speaking roles and rehearse to 13-14 minutes.
4. Confirm official administrative placeholders with the lecturer or LMS.
5. Freeze the initial methodology before formal model experimentation.

## 3.3 Completed work

- **VERIFIED** - Formal assignment and proposal PDFs were extracted and visually inspected.
- **VERIFIED** - The uploaded archive was decompressed and the CSV mechanically profiled.
- **VERIFIED** - File checksums, dimensions, schema, missingness, duplicates, class balance, ranges, and major deterministic relationships were recorded.
- **VERIFIED AUDIT ONLY** - A limited holdout diagnostic indicated that resignation may contain little predictive signal. This is not a final experiment.
- **DECIDED** - Phase 1 decisions Q-001 through Q-017 were approved by the user on 2026-07-28. Context v3 adds Q-018 through Q-022 to make the AI handoff self-contained and methodologically explicit.
- **PLACEHOLDER** - The administrative values governed by Q-017 remain subject to official confirmation, as approved.
- **SUPERSEDED COMPLETION RECORD** - Canonical context v3.0.0 and synchronized Word companion were generated and revalidated on 2026-07-28.
- **DECIDED / COMPLETED** - On 2026-07-29, the project title was changed to **Design and Evaluation of Machine Learning Models for Employee Resignation Prediction**, and the AI handoff was updated to state exactly what must be built and what is out of scope.

## 3.4 Work that may proceed now

- repository setup and raw-data preservation;
- data-validation code and reproducible audit notebook;
- proposal drafting and rehearsal;
- EDA planning and figure templates;
- pipeline implementation against training data only;
- unit tests and result-schema design;
- AI-use logging and source register maintenance.

## 3.5 Blocked or conditional work

- final submission scheduling and calendar planning depend on official dates;
- final report formatting depends on the official citation style and page-count rules;
- final model claims depend on formal experiments;
- final viva plan depends on the viva format and date;
- assigning named owners depends on team-member details.

## 3.6 Work not to prioritize now

- web or mobile applications;
- production deployment, APIs, databases, or cloud infrastructure;
- complex neural networks or large external libraries;
- changing the target to obtain higher scores;
- drafting final conclusions before experiments;
- elaborate fairness mitigation before confirming that the model has usable signal.

## 3.7 Current risks

| Risk ID | Risk | Probability | Impact | Current control | Owner |
| --- | --- | --- | --- | --- | --- |
| R-001 | `Resigned` may be independent of available predictors. | High | High | Use baseline comparison, pre-registered metrics, and negative-result contingency. | Member 4 and Member 5 roles; names unresolved |
| R-002 | Dataset provenance may be misrepresented as real employee data. | Medium | High | Use approved cautious provenance statement and prohibit unsupported claims. | Member 2 and Member 6 roles |
| R-003 | Accuracy may conceal failure on the 10.01% positive class. | High | High | Use PR-AUC primary and report minority-class metrics. | Member 5 role |
| R-004 | Leakage through holdout use or preprocessing outside folds. | Medium | High | Lock formal holdout against future tuning; use development CV as primary evidence; fit learned steps inside pipelines/folds; disclose historical audit exposure. | Members 2, 4, and 5 roles |
| R-005 | Scope creep into an application or excessive model portfolio. | Medium | Medium | Maintain approved scope and decision log. | Member 1 role |
| R-006 | Administrative requirements remain unconfirmed. | Medium | Medium | Obtain LMS or lecturer confirmation; retain placeholders. | Member 1 role |
| R-007 | Six independently produced sections become inconsistent. | Medium | High | Common split, schemas, result files, peer review, and final integrator. | All members |
| R-008 | AI-assisted output is used without validation or disclosure. | Medium | High | Maintain live usage record and require human/code verification. | Member 6 role |

## 3.8 Next concrete actions

| Order | Action | Completion evidence | Status |
|---|---|---|---|
| 1 | Create shared repository using the approved layout. | Repository exists; path entered in this file. | TO BE GENERATED |
| 2 | Copy the verified CSV unchanged to `data/raw/`. | Checksum matches SRC-004. | TO BE GENERATED |
| 3 | Implement `01_data_audit.ipynb` or equivalent script. | Reproduces schema and audit assertions. | TO BE GENERATED |
| 4 | Draft the ten proposal slides. | Slides contain all formal required topics. | TO BE GENERATED |
| 5 | Assign names to six role slots. | Team map updated. | PLACEHOLDER - HUMAN INPUT REQUIRED |
| 6 | Request official administrative confirmations. | LMS notice or lecturer response saved. | PLACEHOLDER - EXTERNAL INPUT REQUIRED |
| 7 | Rehearse and time the proposal. | Complete run is 13-14 minutes. | TO BE GENERATED |

# 4. Task router for AI assistants

| Future request | Must read | Must verify | May generate | Must not invent |
| --- | --- | --- | --- | --- |
| Understand the project | Sections 0-3 and 8.1-8.10 | Current version, status, and unresolved placeholders | A concise summary with statuses | New objectives, dates, or results |
| Inspect data | Sections 5, 8.10-8.14, and 8.18-8.19 | CSV checksum, row/column count, schema | Audit code, tables, EDA plan | Units, provenance, missingness, or semantic meanings not supported |
| Generate preprocessing code | Sections 8.15-8.18 | Approved feature policy and split protocol | scikit-learn pipelines and validation tests | Pre-split resampling, raw-date use, or silent row deletion |
| Generate modelling code | Sections 8.20-8.22 | Development/formal-holdout separation and metric definitions | Dummy, Logistic Regression, Random Forest, optional Decision Tree | Formal-holdout tuning or unapproved models as required scope |
| Perform EDA | Sections 8.11, 8.18, and 8.19 | Source file and target orientation | Reproducible charts and descriptive findings | Causal explanations or invented outliers |
| Tune hyperparameters | Sections 8.21-8.22 | Search space, CV folds, scoring, random seed | Grid/RandomizedSearchCV code and results schema | Using the formal holdout for tuning or presenting audit-only diagnostics as formal results |
| Evaluate results | Sections 8.22-8.25 and 8.33 | Saved out-of-fold and formal-holdout predictions | Metric tables, uncertainty, error analysis, fairness analysis | A winner when models are indistinguishable |
| Write a report section | Sections 8.6-8.9 and the relevant method/result section | Formal report order, result provenance, page limit | Concise evidence-backed prose with citations | Fabricated findings or omitted limitations |
| Create presentation content | Sections 8.28-8.29 | 15-minute duration and required topics | Ten-slide content, speaker notes, figures | Mandatory slide count, Q&A, or exact date |
| Prepare viva questions | Sections 8.2, 8.13, 8.17-8.25, and 8.33 | Approved decisions and actual results | Questions, model answers, challenge scenarios | Results that have not been generated |
| Debug code | Sections 5, 8.17, 8.26, and repository README | Environment, file paths, split indices, failing test | Minimal corrections and tests | Changing methodology to hide failures |
| Update project status | Sections 3, 7, 8.31-8.36 | Evidence for completion or change | Status, risk, AI-use, and change-log entries | Unverified completion claims |
| Recommend scope change | Sections 0.9, 6, 8.8, and 8.36 | Trigger, requirement, cost, and authority | A decision proposal with trade-offs | Automatic implementation of the change |
| Handle a missing file | Sections 5 and 7 | Manifest, checksum, exact expected filename | A missing-file report and recovery steps | Substituting a different file |

# 5. Companion-file manifest and handoff package

## 5.1 Required files for project-dependent work

| ID | Canonical filename or path | Required for | Integrity rule | Status |
| --- | --- | --- | --- | --- |
| CTX-MD | Employee_Resignation_Prediction_AI_Project_Context_v3_1.md | All project-dependent work | Canonical; version and date must match this document. | VERIFIED generated |
| CTX-DOCX | Employee_Resignation_Prediction_AI_Project_Context_v3_1.docx | Synchronized cross-format fallback and Word navigation | Must be regenerated from CTX-MD after substantive changes. | VERIFIED generated |
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
│   └── 03_modelling.ipynb
├── src/
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── fairness.py
├── tests/
├── models/
├── results/
├── reports/
├── presentations/
└── docs/
    └── context/
        ├── Employee_Resignation_Prediction_AI_Project_Context_v3_1.md
        └── Employee_Resignation_Prediction_AI_Project_Context_v3_1.docx
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

1. `Employee_Resignation_Prediction_AI_Project_Context_v3_1.md` - canonical source.
2. `Employee_Resignation_Prediction_AI_Project_Context_v3_1.docx` - equivalent synchronized fallback.

**Data-execution handoff:** Add the exact raw CSV or original archive. Data-dependent work must stop if its checksum cannot be verified.

**Implementation/result handoff:** Add the repository, environment lock, split record, results, saved models, and latest official administrative clarifications when they exist.

Original PDFs are optional for ordinary requirement parsing because their complete normalized text is embedded. They are required when exact page appearance, annotations, or a newer/differently hashed revision must be checked. A similarly named dataset, specification, report, or notebook is not an acceptable substitute.

# 6. Dependency map

## 6.1 Technical and procedural dependencies

```text
Formal requirements
    -> approved problem framing
        -> target and positive-class definition
            -> feature policy and leakage controls
                -> reproducible split
                    -> preprocessing pipelines
                        -> baseline and model tuning on training CV
                            -> threshold selection on training-fold predictions
                                -> formal holdout corroboration
                                    -> error, explainability, and fairness analysis
                                        -> conclusions, report, and viva

Exact dataset identity and checksum
    -> schema and semantic audit
        -> data-quality limitations
            -> EDA and modelling validity

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

> Read `Employee_Resignation_Prediction_AI_Project_Context_v3_1.md` first and treat it as canonical. Follow its authority hierarchy, status labels, fixed decisions, checksum rules, and anti-hallucination constraints. Before data-dependent work, verify the exact CSV checksum. Do not invent placeholders or results. Update the live state, decision log, result record, AI-use record, version, and synchronized Word companion after material changes.

# 8. Full canonical project specification

## 8.1 Document control

| Field | Value | Status |
| --- | --- | --- |
| Project title | Design and Evaluation of Machine Learning Models for Employee Resignation Prediction | DECIDED |
| Context version | 3.1.0 | VERIFIED generated |
| Canonical filename | Employee_Resignation_Prediction_AI_Project_Context_v3_1.md | DECIDED |
| Word companion | Employee_Resignation_Prediction_AI_Project_Context_v3_1.docx | DECIDED |
| Canonical format | Markdown UTF-8 | DECIDED |
| Last verified date | 2026-07-29 | VERIFIED |
| Current phase | Proposal preparation and project setup | LIVE RECORD |
| Approval record | Phase 1 recommendations approved on 2026-07-28; assignment-aligned project title and build definition approved on 2026-07-29 | DECIDED |
| Document owner | IT2011 group; personal names unresolved | PLACEHOLDER |
| Repository path | Not yet supplied | PLACEHOLDER |

## 8.2 Executive project summary

The group will use the assigned `Extended_Employee_Performance_and_Productivity_Data.csv` file to evaluate whether employee-related attributes can distinguish records marked as resigned from records marked as not resigned. The work is a supervised binary-classification study in the human-resources and workforce-analytics domain.

The project is deliberately framed as an evaluation rather than a promise of an effective resignation-risk system. Mechanical audit findings and preliminary audit-only diagnostics suggest that the target may contain limited predictive signal. The project therefore values reproducibility, baseline comparison, honest negative findings, data-provenance limitations, and critical ethical analysis over headline accuracy.

The approved implementation uses a locked 20% stratified formal holdout, five-fold stratified cross-validation on the 80% training partition, a DummyClassifier baseline, Logistic Regression, Random Forest, and optionally a Decision Tree if time permits. Average Precision/PR-AUC is the primary selection metric because only 10.01% of records are positive.

### 8.2.1 Exact assessed build

The group is building an **end-to-end, reproducible Python ML study**, not a deployed software product. The minimum complete build consists of:

1. immutable preservation and validation of the assigned CSV;
2. a reproducible data-audit and EDA workflow with visualizations;
3. leakage-safe preprocessing pipelines;
4. dummy baselines plus at least Logistic Regression and Random Forest;
5. hyperparameter tuning and five-fold stratified development cross-validation;
6. threshold analysis, one-time formal-holdout corroboration, and comparison using imbalance-appropriate metrics;
7. saved code, configurations, predictions, metrics, figures, and fitted pipeline artefacts;
8. interpretability, error, ethics, bias, and subgroup analysis; and
9. proposal, final-report, AI-use-declaration, and viva evidence generated from verified outputs.

A GUI, website, mobile app, dashboard, API, database, cloud deployment, or production HR decision tool is outside the assessed scope unless an official instruction explicitly adds it. The build remains academically successful when the workflow is correct and critically evaluated, even if the formal evidence shows that the dataset cannot support useful resignation prediction.

## 8.3 Formal problem statement

**DECIDED**

> Evaluate whether the available demographic, employment, workload, performance, and satisfaction variables can distinguish employees recorded as `Resigned=True` from employees recorded as `Resigned=False`, using reproducible supervised-learning pipelines, development cross-validation, and one-time formal-holdout corroboration with the historical audit-exposure limitation disclosed.

The project must not imply that:

- the dataset represents a verified real organization;
- the model identifies causes of resignation;
- the model is safe or lawful for employment decisions;
- a useful predictor is guaranteed;
- a result near chance is a project failure.

## 8.4 Objectives and project questions

### 8.4.1 Objectives

1. Verify and document the exact assigned dataset, schema, quality, and provenance limitations.
2. Perform reproducible EDA with appropriate visualizations.
3. Build a non-informative baseline and at least two defensible supervised classifiers.
4. Tune models using training-only stratified cross-validation and leakage-safe pipelines.
5. Compare models using metrics appropriate for class imbalance.
6. Determine whether performance exceeds simple baselines in a practically meaningful way.
7. Analyse errors, interpretability, subgroup behaviour, and ethical risks.
8. Produce a clear proposal, concise final report, and defensible viva explanation.
9. Maintain transparent records of generative-AI assistance and human verification.

### 8.4.2 Project questions

- PQ-01: Do the available predictors provide out-of-sample resignation signal beyond a dummy baseline?
- PQ-02: How do Logistic Regression and Random Forest compare under the same split, preprocessing, and metric framework?
- PQ-03: How does class weighting and threshold selection affect minority-class recall, precision, and F1?
- PQ-04: Are model rankings stable across training folds?
- PQ-05: Does excluding `Gender`, or excluding both `Gender` and `Age`, materially change performance or subgroup disparities?
- PQ-06: Which data-generation artefacts and provenance limitations constrain interpretation?
- PQ-07: If no useful signal exists, what evidence supports rejecting deployment and what data would be needed for a better study?

## 8.5 Intended users, audience, and stakeholders

| Stakeholder | Interest | Permitted use | Not permitted |
| --- | --- | --- | --- |
| Module lecturers and assessors | Compliance, technical quality, critical evaluation, ethics | Assess proposal, implementation, report, and viva | Assume unreported work or invented results |
| Six-member student group | Learning, collaboration, marks, reproducibility | Develop and explain the project | Delegate ownership entirely to AI |
| Future AI assistants | Continuity and task execution | Use this context and verified artefacts | Override decisions or fabricate missing information |
| Hypothetical HR analyst audience | Understand methodological implications | Interpret aggregate academic findings | Use outputs for individual employment actions |
| People represented by the concept of employee data | Potential impact of risk scoring | Benefit from ethical analysis and non-deployment safeguards | Be profiled, ranked, dismissed, or surveilled by this project |

## 8.6 Assignment requirements

| Requirement ID | Formal requirement | Project response | Status |
| --- | --- | --- | --- |
| REQ-01 | Use the assigned dataset; selection is not required. | Use exact SRC-004 and prohibit substitution. | VERIFIED / DECIDED |
| REQ-02 | Identify a real-world domain aligned with the dataset. | Human resources and workforce analytics; resignation evaluation. | DECIDED |
| REQ-03 | Describe features, target, size, quality, and suitability. | Full data audit and schema in Sections 8.10-8.13. | DECIDED |
| REQ-04 | Clean/preprocess data and perform EDA with visualizations. | Leakage-safe preprocessing and EDA plan in Sections 8.18-8.19. | DECIDED |
| REQ-05 | Use at least two suitable AI/ML algorithms. | Logistic Regression and Random Forest; optional Decision Tree. | DECIDED |
| REQ-06 | Implement in Python. | Python 3.11 and scikit-learn stack. | DECIDED |
| REQ-07 | Include hyperparameter tuning and model pipeline steps. | Training-only Grid/RandomizedSearchCV within pipelines. | DECIDED |
| REQ-08 | Use relevant metrics and k-fold cross-validation. | PR-AUC primary, secondary metrics, five-fold stratified CV. | DECIDED |
| REQ-09 | Compare models and justify best performer. | Pre-registered model-selection rule and negative-result contingency. | DECIDED |
| REQ-10 | Discuss ethics and bias mitigation. | Sensitive-feature tests, subgroup metrics, prohibited-use statement. | DECIDED |
| REQ-11 | Provide AI Tool Usage Declaration and Transparency. | Preliminary plan plus actual-use live record. | DECIDED / LIVE RECORD |
| REQ-12 | Professional final report, maximum 15 pages, PDF before viva. | Report plan in Section 8.28; exact deadline unresolved. | VERIFIED / PLACEHOLDER |
| REQ-13 | 15-minute proposal PowerPoint in lab. | Ten-slide, 13-14 minute rehearsed presentation plan. | VERIFIED / DECIDED |

## 8.7 Assessment and acceptance criteria

| Area | Marks | Acceptance criteria for this project |
| --- | --- | --- |
| Problem framing and dataset understanding | 10 | Exact source and link; correct target and class balance; defensible domain; quality and provenance limitations; planned AI use. |
| Preprocessing and EDA | 25 | Reproducible validation; justified transformations; no unnecessary imputation/deletion; visual evidence; meaningful patterns and limitations. |
| Model design and implementation | 25 | At least Logistic Regression and Random Forest; justified baseline; pipelines; tuning; reproducible code and seeds. |
| Evaluation and comparison | 30 | Five-fold stratified development CV; PR-AUC primary; full minority-class metrics; formal holdout corroboration; baseline comparison; honest best-model or no-signal conclusion. |
| Group report | 10 | Professional, max 15 pages, required sections, ethical analysis, AI transparency, reflections, references, consistent visuals. |

**Project-level acceptance rule:** A low-scoring model is not automatically a failed project. The project succeeds when the group executes the approved protocol correctly, reports evidence honestly, explains limitations, and critically evaluates whether deployment should be rejected.

## 8.8 Scope boundaries

### In scope

- exact dataset and provenance audit;
- binary classification of `Resigned`;
- descriptive EDA and quality analysis;
- preprocessing and validation pipelines;
- DummyClassifier, Logistic Regression, Random Forest, optional Decision Tree;
- class weighting and threshold analysis;
- five-fold stratified development cross-validation and one-time formal-holdout corroboration;
- explainability, error analysis, calibration where probabilities are interpreted;
- subgroup/fairness analysis;
- proposal, final report, viva, and reproducibility artefacts;
- AI-tool disclosure and human verification.

### Out of scope unless officially required

- data collection from a real organization;
- deployment, API, web application, dashboard, database, cloud hosting, or monitoring service;
- automated HR decisions or individual risk reports;
- causal inference or intervention-effect estimation;
- external personal or sensitive datasets;
- changing the target to maximize performance;
- broad multi-target analytics programme;
- neural networks, SVMs, or boosting frameworks without a documented need;
- legal compliance certification.

## 8.9 Correct terminology and prohibited claims

| Use | Avoid |
| --- | --- |
| "records marked as resigned" | "employees who definitely resigned for these reasons" |
| "predictive association" or "out-of-sample signal" | "cause" or "driver" without causal design |
| "likely constructed or synthetic-like patterns" | "confirmed synthetic dataset" unless authoritatively verified |
| "academic evaluation" | "production HR system" |
| "Average Precision/PR-AUC" | "accuracy proves success" |
| "subgroup performance audit" | "legally fair" or "bias-free" |
| "model did not exceed baseline meaningfully" | "the project failed" |
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
| SRC-007 | Conversation decision on 2026-07-29 | Authority for the active project title and clarified build definition. | User selected "Design and Evaluation of Machine Learning Models for Employee Resignation Prediction" and requested synchronized AI-only context files. |

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
| A-006 | The current dataset is static. | SUPPORTED by Kaggle listing | Record checksum; re-audit if changed. |
| A-007 | Computing resources are suitable for conventional tabular models. | DECIDED | Avoid excessive models and use modest searches. |
| C-001 | Final report maximum is 15 pages. | VERIFIED | Prioritize high-mark sections and concise visuals. |
| C-002 | At least two algorithms and k-fold CV are required. | VERIFIED | Use Logistic Regression, Random Forest, and five-fold CV. |
| C-003 | Exact deadlines and citation style are not supplied. | PLACEHOLDER | Obtain official confirmation. |
| C-004 | No deployment is required. | VERIFIED absence plus DECIDED scope | Do not prioritize app development. |

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
| D-001 | Primary task is binary classification of `Resigned`, framed as evaluation of predictive signal. | DECIDED |
| D-002 | Project title is "Design and Evaluation of Machine Learning Models for Employee Resignation Prediction". | DECIDED |
| D-003 | No application or deployment in assessed scope. | DECIDED |
| D-004 | Provenance described as undocumented and likely constructed, not verified real employee data. | DECIDED |
| D-005 | Exclude `Employee_ID`; audit then exclude raw `Hire_Date`; retain `Years_At_Company`. | DECIDED |
| D-006 | Use an 80/20 stratified split with seed 42; five-fold stratified CV on development training is primary evidence; formal holdout is one-time corroboration with the historical exposure caveat. | DECIDED |
| D-007 | Use DummyClassifier, Logistic Regression, and Random Forest; optional Decision Tree. | DECIDED |
| D-008 | Use Average Precision/PR-AUC as primary metric. | DECIDED |
| D-009 | Use class weights and threshold analysis; no pre-split SMOTE. | DECIDED |
| D-010 | Accept and report a defensible no-signal outcome. | DECIDED |
| D-011 | Academic use only; fairness sensitivity with and without sensitive features. | DECIDED |
| D-012 | Use coefficients, permutation importance, and error analysis; no causal claims. | DECIDED |
| D-013 | Use reproducible Python/scikit-learn repository and immutable raw data. | DECIDED |
| D-014 | Plan ten proposal slides and 13-14 minutes of rehearsed content. | DECIDED |
| D-015 | Use six numbered responsibility slots with peer review until names are supplied. | DECIDED |
| D-016 | Record ChatGPT as confirmed planned and actual AI assistance. | DECIDED |
| D-017 | Administrative items remain external placeholders. | DECIDED handling |
| D-018 | Embed complete normalized transcripts of both formal PDFs; originals are needed only for visual/revision verification. | DECIDED |
| D-019 | Preserve preliminary audit diagnostics and exposure limitation; never claim a historically pristine holdout. | DECIDED |
| D-020 | Use the one-standard-error model tie-break, fixed threshold policy, 2,000-resample bootstrap rule, and explicit usable-signal rule. | DECIDED |
| D-021 | Use fixed age bands and minimum subgroup denominators for fairness reporting. | DECIDED |
| D-022 | Require startup acknowledgement, stable/live-state precedence, and Markdown/Word synchronization. | DECIDED |

## 8.17 End-to-end workflow and experimental protocol

1. **Acquire and verify**
   - restore the exact CSV from SRC-003;
   - compute SHA-256 and assert shape/schema;
   - preserve immutable raw file.
2. **Audit semantics and quality**
   - parse types;
   - verify missingness, duplicates, ranges, categories, target balance;
   - document deterministic relationships and implausible age/tenure combinations.
3. **Create split**
   - map `Resigned=True` to 1;
   - perform stratified 80/20 split with `random_state=42`;
   - save row IDs or split indices;
   - lock the formal holdout against all future tuning and record its split ID;
   - label the holdout as formal corroboration, not historically pristine, because prior audit split metadata is unavailable.
4. **Training-only EDA and design**
   - use training data for any target-conditioned exploratory analysis;
   - pre-register preprocessing and model searches;
   - avoid target-driven feature engineering using formal-holdout data.
5. **Build preprocessing pipelines**
   - exclude ID and raw date;
   - one-hot encode categorical variables;
   - scale numerical variables for Logistic Regression;
   - fit all learned steps within training folds.
6. **Establish baseline**
   - `DummyClassifier(strategy="prior")` for the constant prevalence-probability/PR-AUC reference;
   - `DummyClassifier(strategy="most_frequent")` for majority-class accuracy and threshold-metric reference.
7. **Tune candidate models**
   - Logistic Regression via GridSearchCV;
   - Random Forest via bounded RandomizedSearchCV;
   - optional Decision Tree only if required pipelines, tests, and proposal/report obligations are on schedule before model freeze.
8. **Select model and threshold**
   - compare mean and variability of training CV Average Precision;
   - inspect precision-recall trade-offs from out-of-fold predictions;
   - apply the one-standard-error rule: among valid configurations within one standard error of the best mean CV Average Precision, choose the simpler and more interpretable pipeline;
   - use 0.5 as the reference threshold and a secondary threshold maximizing F1 on pooled out-of-fold development predictions.
9. **Freeze pipeline**
   - record hyperparameters, features, threshold, code commit, package versions, and split ID.
10. **Evaluate once on formal holdout**
    - compute all approved metrics; use 2,000 stratified bootstrap resamples with seed 42 for 95% percentile intervals where the metric is defined;
    - generate confusion matrix, precision-recall curve, ROC curve, and calibration check if probabilities are used.
11. **Analyse errors and fairness**
    - false-positive and false-negative patterns;
    - sensitive-feature sensitivity models;
    - subgroup metrics with sample-size warnings.
12. **Interpret and decide**
    - compare against baseline and project questions;
    - reject deployment if performance is weak or unstable;
    - document limitations and future data requirements.
13. **Package and report**
    - save reproducible artefacts;
    - generate report tables/figures from result files;
    - update AI-use, decision, risk, and result records.

## 8.18 Architecture, repository, data handling, and leakage controls

### 8.18.1 Component responsibilities

| Component | Responsibility | Required outputs |
| --- | --- | --- |
| `data_validation.py` | Checksum, schema, ranges, categories, duplicates, target balance, date and consistency checks | Validation report and fail-fast assertions |
| `preprocessing.py` | Feature lists, ColumnTransformer, model-specific pipelines | Serializable preprocessing objects |
| `train.py` | Split loading, CV, tuning, deterministic seeds, experiment metadata | Fitted pipelines and CV result tables |
| `evaluate.py` | Formal-holdout metrics, curves, confusion matrix, error slices | Machine-readable metrics and figures |
| `fairness.py` | Sensitive-feature sensitivity and subgroup metrics | Fairness table with sample sizes and caveats |
| Notebooks | Transparent audit, EDA, and interpretation | Re-executable narrative analysis |
| `tests/` | Integrity, leakage, transformation, serialization, and metric tests | Passing test report |
| `results/` | Canonical experiment outputs | JSON/CSV tables linked to commit and split |
| `models/` | Full fitted pipelines and metadata | joblib artefact and model card |

### 8.18.2 Leakage and integrity controls

- The formal holdout is unavailable to hyperparameter search, feature selection, threshold selection, and target-conditioned EDA after repository initialization. Historical preliminary-audit exposure remains disclosed.
- One-hot categories and scalers are fitted inside each CV fold.
- Resampling, if ever tested, occurs only inside an imbalanced-learn pipeline within folds.
- `Employee_ID` is excluded from predictors and used only for traceability.
- Raw `Hire_Date` is excluded from primary modelling because it duplicates tenure and contains a fixed time artefact.
- Formal-holdout predictions are saved only after preprocessing, hyperparameters, threshold, split ID, code commit, and package versions are frozen.
- Report tables are generated from saved result files, not manually typed.
- Raw CSV is never overwritten; processed data receives new filenames and metadata.
- A checksum mismatch invalidates prior results until re-audited.

## 8.19 Preprocessing and EDA plan

### 8.19.1 Preprocessing

- Assert exact schema before processing.
- Convert target Boolean to binary integer with explicit mapping.
- Parse `Hire_Date` for audit only.
- Create audit-only `Implied_Hire_Age = Age - Years_At_Company`; do not use it in the primary model unless separately approved.
- Drop `Employee_ID` and raw `Hire_Date` from primary predictors.
- One-hot encode `Department`, `Gender`, `Job_Title`, and `Education_Level` with unknown-category handling.
- Treat remaining approved variables as numerical.
- Standardize numerical features in Logistic Regression pipeline; do not require scaling for trees.
- Do not impute because the verified file has no missing values. Fail validation if a later file introduces missingness, then document a new decision.
- Do not automatically delete statistical outliers from bounded variables.
- Preserve class distribution; compare class weights in model search.

### 8.19.2 EDA questions and figures

| EDA item | Recommended output | Interpretation rule |
| --- | --- | --- |
| Dataset overview | Shape, schema, head sample, type summary | Do not expose ID as meaningful feature |
| Target distribution | Count and percentage bar chart | Annotate 89.99% majority baseline |
| Categorical predictors | Count plots and target-rate plots | Show sample sizes; avoid causal labels |
| Numerical predictors | Histograms/boxplots and target-stratified summaries | Use consistent axes; note bounded regularity |
| Date and tenure | Date histogram and tenure comparison | Highlight fixed time and redundancy |
| Salary structure | Job title x performance salary table/heatmap | Describe deterministic mapping and unknown unit |
| Age-tenure consistency | Distribution of implied hire age | Report implausible values; do not silently remove |
| Correlations | Numeric correlation heatmap plus target correlations | Correlation near zero supports no-signal concern but is not decisive |
| Target signal by subgroup | Target rates with confidence intervals where suitable | Do not overinterpret small differences in 100,000 generated-like records |
| Scatter plots | Use sampled points or density plots | Avoid unreadable overplotting |

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
- Random seed: 42 for split, CV shuffling, and stochastic models.
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Primary scorer: `average_precision`.
- Use parallel jobs where supported and record hardware/runtime.

### 8.21.2 Logistic Regression search

- solver: `liblinear` or another solver justified by the implementation;
- penalty: `l1`, `l2` where solver-compatible;
- `C`: 0.01, 0.1, 1, 10;
- `class_weight`: `None`, `balanced`;
- `max_iter`: 2000;
- record convergence warnings rather than suppressing them silently.

### 8.21.3 Random Forest search

Use a bounded `RandomizedSearchCV` rather than an excessive full grid. Candidate dimensions may include:

- `n_estimators`: 200 to 600;
- `max_depth`: `None`, 8, 12, 20;
- `min_samples_split`: 2, 10, 30;
- `min_samples_leaf`: 1, 5, 20;
- `max_features`: `sqrt`, `log2`, 0.5;
- `class_weight`: `None`, `balanced`, `balanced_subsample`.

Use `n_iter=24` and `random_state=42`. The exact sampled combinations, runtime, and selected values are **TO BE GENERATED** and must be recorded.

### 8.21.4 Threshold selection

- Default threshold 0.5 is reported as a reference.
- Any alternative threshold is selected from out-of-fold training predictions.
- The predefined secondary threshold maximizes F1 on pooled out-of-fold development predictions.
- Report the full precision-recall trade-off; do not imply that F1 represents a verified business cost ratio.
- No business cost ratio is available; do not invent one.
- The formal holdout is not used to select the threshold.

## 8.22 Evaluation and comparison framework

### 8.22.1 Primary and secondary metrics

| Metric | Role | Reason and caution |
| --- | --- | --- |
| Average Precision / PR-AUC | Primary selection metric | Focuses on positive-class ranking under imbalance; prevalence reference is 0.1001. |
| ROC-AUC | Secondary ranking metric | Threshold-independent but can look less severe under imbalance. |
| Precision | Secondary threshold metric | Measures correctness of predicted resignations. |
| Recall | Secondary threshold metric | Measures captured positive records. |
| F1 | Secondary threshold metric | Balances precision and recall but hides the individual trade-off. |
| Balanced accuracy | Secondary threshold metric | Balances sensitivity across classes. |
| Confusion matrix | Required diagnostic | Shows false positives and false negatives explicitly. |
| Accuracy | Context only | Majority-only baseline already achieves 0.8999. |
| Brier score/calibration curve | Conditional but rule-bound | Mandatory if outputs are described as probabilities or risk estimates; otherwise optional and no calibration claim is allowed. |

### 8.22.2 Model-selection rule

1. Rank candidate configurations by mean training-CV Average Precision.
2. Inspect fold variability and precision-recall behaviour.
3. Reject configurations that have unstable folds, convergence failures, or implausible complexity without benefit.
4. Apply a one-standard-error rule: among valid configurations whose mean CV Average Precision is within one standard error of the best mean, choose the simpler and more interpretable pipeline.
5. Freeze preprocessing, hyperparameters, class weighting, threshold, split ID, package versions, and code commit before formal-holdout evaluation.
6. Use the formal-holdout result only as one-time corroboration, not to continue model selection; disclose the historical audit-exposure limitation.
7. Identify the numerical best performer even when all models are weak, because the assignment requires comparison.
8. Call signal potentially useful only when mean CV Average Precision exceeds the prior baseline consistently, the absolute mean improvement is at least 0.01, uncertainty is not dominated by baseline performance, and no serious subgroup failure invalidates the claim. This is a project-defined rule, not a universal standard.
9. If candidates remain close to dummy/random references or unstable, conclude that the available data does not support useful prediction under this design.

### 8.22.3 Uncertainty

- Report each fold value, CV mean, standard deviation, and the standard error used by the one-standard-error rule.
- For formal-holdout metrics, use 2,000 stratified bootstrap resamples, seed 42, and 95% percentile intervals where the metric is defined.
- Do not infer significance solely from tiny metric differences on a large constructed-like dataset.

## 8.23 Explainability and interpretability plan

- Logistic Regression: report selected coefficients, direction of association, and odds ratios only with the representation and scaling explained.
- Random Forest: use permutation importance on held-out validation or the formal holdout; do not rely solely on impurity importance.
- Optional partial dependence: limited to justified numerical features and accompanied by non-causal caveats.
- Report importance stability across CV folds when feasible.
- Analyse false positives and false negatives without creating individual employee narratives.
- Never translate an importance ranking into a claim that the feature causes resignation.
- Avoid SHAP unless the group can explain it correctly and the model demonstrates usable signal.

## 8.24 Ethics, fairness, privacy, security, and risk analysis

### 8.24.1 Intended and prohibited use

**DECIDED intended use:** academic evaluation and aggregate methodological learning.

**DECIDED prohibited use:** hiring, dismissal, promotion, discipline, compensation, surveillance, or any individual employee decision.

### 8.24.2 Ethical risks

| Risk | Why it matters | Control |
| --- | --- | --- |
| Direct discrimination | Gender and age may influence predictions. | Compare full model with models excluding Gender and excluding Gender plus Age. |
| Proxy discrimination | Department, job title, salary, and tenure may encode protected or structural differences. | Discuss proxies and inspect subgroup performance. |
| False positives | A record could be incorrectly labelled high resignation risk. | No individual use; report false-positive rate and confusion matrix. |
| False negatives | Actual positive records may be missed. | Report recall and error patterns; do not claim comprehensive detection. |
| Automation bias | Users may over-trust a numerical score. | Explicit non-deployment statement and calibration/uncertainty discussion. |
| Causal overreach | Feature importance may be mistaken for a reason to intervene. | Use association language only. |
| Provenance uncertainty | Unknown collection/generation undermines representativeness. | Treat as likely constructed and avoid population claims. |
| Privacy misrepresentation | Public availability does not prove ethical real-world collection. | Do not claim consent or anonymization; note IDs appear generated. |
| Model instability/no signal | Complex outputs may look credible despite random performance. | Baseline, CV uncertainty, and reject-deployment contingency. |

### 8.24.3 Fairness analysis

- Define `Gender` groups exactly as observed: Female, Male, Other.
- Use fixed age bands `22-29`, `30-39`, `40-49`, and `50-60`; report counts and confirm they remain valid for the exact assigned file before analysis.
- Report sample size, positive prevalence, precision, recall, false-positive rate, and predicted-positive rate only when total `n >= 200`, positive `n >= 30`, and negative `n >= 30`; otherwise report counts/prevalence and label rate estimates unstable or not reported.
- Compare full model, no-Gender model, and optionally no-Gender/no-Age model.
- Do not declare a model fair solely because metrics are similar.
- Do not apply advanced mitigation unless disparities and usable predictive signal justify it.

### 8.24.4 Security and privacy controls

- Store only the public assigned artefact and project outputs; do not add private employee data.
- Do not publish credentials, personal contact details, or private LMS content in the repository.
- Use `.gitignore` for environments, caches, and local secrets.
- Validate model input schema before inference demonstrations.
- Document the model as academic and non-production.

## 8.25 Quality assurance and testing plan

| Test ID | Test | Pass condition |
| --- | --- | --- |
| T-001 | Raw-file checksum | Matches SRC-004 exactly. |
| T-002 | Shape and schema | 100,000 rows; 20 expected columns in exact or documented order. |
| T-003 | Target mapping | Only False/True; True maps to 1. |
| T-004 | Missingness and duplicates | Audit reproduces zero missing and zero exact duplicates for assigned file. |
| T-005 | Identifier exclusion | `Employee_ID` absent from model feature matrix. |
| T-006 | Raw-date exclusion | `Hire_Date` absent from primary model feature matrix. |
| T-007 | Split reproducibility | Same row IDs produced from seed 42 and saved split logic. |
| T-008 | Stratification | Development-training and formal-holdout prevalence remain close to 10.01%. |
| T-009 | Pipeline leakage | Encoders/scalers fitted only within training folds. |
| T-010 | CV configuration | Exactly five stratified shuffled folds with seed 42. |
| T-011 | Metric orientation | `Resigned=True` treated as positive in every metric. |
| T-012 | Baseline presence | Dummy results appear in comparison table. |
| T-013 | Formal-holdout lock | No formal-holdout access before model freeze; access event recorded. |
| T-014 | Serialization | Saved pipeline reloads and reproduces predictions. |
| T-015 | Input validation | Missing, extra, wrong-type, or unseen category inputs handled explicitly. |
| T-016 | Result provenance | Every final table maps to code commit, split ID, environment, and output file. |
| T-017 | Report consistency | Figures and metric values match saved results. |
| T-018 | AI-use disclosure | Every material AI-supported task is recorded and human-verified. |

### Review gates

- **Gate A - Proposal ready:** required content present, source linked, limitations visible, 13-14 minute rehearsal completed.
- **Gate B - Data ready:** checksum/schema tests pass and audit notebook is peer reviewed.
- **Gate C - Modelling ready:** split locked, pipelines tested, search plan frozen.
- **Gate D - Results ready:** formal holdout evaluated once, metrics saved, exposure caveat retained, and negative-result rule applied.
- **Gate E - Submission ready:** report and slides align with verified results, AI declaration complete, citations and formatting checked.

## 8.26 Deliverable specifications

| Deliverable | Required content | Acceptance evidence |
| --- | --- | --- |
| Proposal PowerPoint | Dataset source/link, domain, structure, target, size, quality, suitability, limitations, planned AI use | 15-minute maximum per group; ten-slide working plan; rehearsal record |
| Code repository | Raw-data checksum, audit, EDA, pipelines, tuning, evaluation, fairness, tests | Fresh environment can reproduce outputs |
| Model artefact | Entire fitted pipeline plus metadata and threshold | Reload verification and model card |
| Final report PDF | Nine required sections, max 15 pages, visuals, ethics, AI declaration, references | Submission-ready PDF and consistency QA |
| Viva package | Project summary, methodology, results, limitations, member understanding | Question bank and shared rehearsal |
| Context Markdown | Complete AI-first project memory | UTF-8, valid YAML, current decisions/status |
| Context Word | Synchronized human-readable companion | Visual QA, headings, tables, page numbers, TOC |

## 8.27 Demonstration or prototype specification

No deployed prototype is required. If the group produces a minimal local demonstration, it must:

- load the saved full pipeline;
- accept only schema-valid aggregate/example inputs;
- label output as an academic model estimate, not a decision recommendation;
- display the selected threshold and a caution that probability calibration may be limited;
- refuse missing required fields and unsupported categories cleanly;
- avoid storing inputs;
- include a prominent non-production and non-employment-decision disclaimer.

**Status:** Optional and not to be prioritized before required work.

## 8.28 Final report plan

The formal maximum is 15 pages. Exact inclusion rules for title page, references, and appendices remain unresolved.

| Required section | Content plan | Priority |
| --- | --- | --- |
| 1. Introduction and Problem Statement | Domain, target, evaluation framing, objectives, no-guarantee/no-deployment position. | High |
| 2. Dataset Description | Source/link, exact file, shape, schema, target balance, provenance and deterministic artefacts. | High |
| 3. Preprocessing and EDA | Validation, feature policy, class imbalance, figures, quality findings. | Very high |
| 4. Model Design and Implementation | Baseline, Logistic Regression, Random Forest, pipelines, tuning, seed, split ID, and exposure caveat. | Very high |
| 5. Evaluation and Comparison | Development-CV and formal-holdout metrics, curves, uncertainty, errors, selected model or no-signal outcome. | Very high |
| 6. Ethical Considerations and Bias Mitigation | Prohibited use, subgroup audit, sensitive-feature sensitivity, provenance and proxy risks. | High |
| 7. AI Tool Usage Declaration and Transparency | Actual tools, tasks, extent, verification, ownership. | Required |
| 8. Reflections and Lessons Learned | Negative findings if applicable, teamwork, modelling limits, future data needs. | Medium |
| 9. References | Formal PDFs, Kaggle listing, software/documentation, methods sources. | Required |

### Report writing rules

- Generate tables and plots from saved results.
- Prioritize marks-heavy technical sections.
- State target prevalence and baseline prominently.
- Separate audit-only context diagnostics from formal results.
- Do not include guessed results or generic model descriptions unrelated to actual implementation.
- Use the citation style confirmed through the official placeholder.

## 8.29 Proposal presentation plan

### 8.29.1 Working slide structure

| Slide | Topic | Owner role | Timing guide |
| --- | --- | --- | --- |
| 1 | Title, group, module, one-sentence objective | Member 1 | 1:00 |
| 2 | Dataset source, Kaggle link, exact file and licence | Member 1 | 1:15 |
| 3 | Dataset dimensions, schema and feature groups | Member 2 | 1:30 |
| 4 | Target definition, positive class, and 10.01% imbalance | Member 3 | 1:30 |
| 5 | Real-world HR problem and cautious evaluation framing | Member 3 or 4 | 1:30 |
| 6 | Why the dataset is suitable for supervised comparison | Member 4 | 1:15 |
| 7 | Quality and limitations: provenance, date/tenure, deterministic salary | Member 5 | 2:00 |
| 8 | Planned preprocessing, split, baseline and models | Member 5 | 1:45 |
| 9 | Ethics, fairness and planned AI-tool usage | Member 6 | 1:30 |
| 10 | Conclusion, expected evidence and immediate next actions | Member 6 | 1:00 |

Target rehearsed duration: 13-14 minutes, leaving 1-2 minutes for transitions or lecturer interaction. The document does not specify a Q&A block or mandatory slide count.

### 8.29.2 Slide content rules

- Cite and link the Kaggle source directly.
- Show no more than one dense table per slide.
- Use a clear class-distribution visual.
- Present limitations as evidence of critical analysis, not as an apology.
- Do not present audit-only diagnostic model scores unless the group chooses to explain that they are preliminary and not final.
- Every member should understand every slide even if speaking ownership is divided.

## 8.30 Team responsibility map

| Role slot | Primary responsibility | Required peer review | Name |
| --- | --- | --- | --- |
| Member 1 | Coordination, requirements, problem framing, final integration | Member 6 | PLACEHOLDER |
| Member 2 | Dataset audit, schema, preprocessing, validation | Member 3 | PLACEHOLDER |
| Member 3 | EDA, charts, data-quality analysis, presentation visuals | Member 2 | PLACEHOLDER |
| Member 4 | Dummy baseline and Logistic Regression | Member 5 | PLACEHOLDER |
| Member 5 | Random Forest, tuning, comparison, formal-holdout evaluation | Member 4 | PLACEHOLDER |
| Member 6 | Ethics, fairness, AI-use record, report consistency, references | Member 1 | PLACEHOLDER |

Team rules:

- Members 4 and 5 must use the same split, preprocessing policy, metrics, and result schema.
- No member independently changes the target, formal holdout, positive label, or metric without a logged decision.
- Every major artefact receives at least one peer review.
- All six members must be able to explain the full pipeline in the viva.
- Contribution evidence should be retained through commits, review notes, meeting records, and output ownership.

## 8.31 Timeline and milestones

| Milestone | Required completion condition | Date/status |
| --- | --- | --- |
| Context and decisions frozen | Context v3 approved and generated | COMPLETED 2026-07-28 |
| Repository setup | Approved layout exists and raw checksum passes | TO BE GENERATED |
| Proposal draft | Ten slides cover all formal requirements | TO BE GENERATED |
| Proposal rehearsal | Full delivery is 13-14 minutes and roles are balanced | TO BE GENERATED before presentation |
| Proposal delivery | In-class PowerPoint during lab | Tentative week beginning 2026-08-03; exact date PLACEHOLDER |
| Data/EDA gate | Audit tests pass and EDA is peer reviewed | TO BE GENERATED |
| Model-development gate | Baseline and candidates tuned using training CV | TO BE GENERATED |
| Final evaluation gate | Frozen pipeline evaluated once on formal holdout | TO BE GENERATED |
| Report complete | Nine sections, max 15 pages, consistency QA | TO BE GENERATED; deadline PLACEHOLDER |
| Viva readiness | All members can explain methods/results/ethics | TO BE GENERATED; date PLACEHOLDER |

## 8.32 Current project status

- Project definition: **DECIDED**.
- Dataset identity and audit: **VERIFIED**.
- Proposal specification: **VERIFIED**.
- Proposal slides: **TO BE GENERATED**.
- Repository and formal code: **TO BE GENERATED**.
- Formal EDA: **TO BE GENERATED**.
- Formal model experiments: **TO BE GENERATED**.
- Final results and selected solution: **TO BE GENERATED**.
- Ethics/fairness execution: **TO BE GENERATED**.
- AI-use plan: **DECIDED**.
- Actual AI use: **LIVE RECORD**.
- Administrative schedule and formatting details: **PLACEHOLDER - EXTERNAL INPUT REQUIRED**.

## 8.33 Results templates - TO BE GENERATED

### 8.33.1 Experiment register

| Experiment ID | Code commit | Dataset checksum | Split ID | Pipeline/model | CV design | Primary score | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-001 | TO BE GENERATED | SRC-004 | TO BE GENERATED | DummyClassifier | 5-fold stratified CV | TO BE GENERATED | TO BE GENERATED |
| EXP-002 | TO BE GENERATED | SRC-004 | TO BE GENERATED | Logistic Regression | 5-fold stratified CV | TO BE GENERATED | TO BE GENERATED |
| EXP-003 | TO BE GENERATED | SRC-004 | TO BE GENERATED | Random Forest | 5-fold stratified CV | TO BE GENERATED | TO BE GENERATED |
| EXP-004 | TO BE GENERATED | SRC-004 | TO BE GENERATED | Decision Tree optional | 5-fold stratified CV | TO BE GENERATED | OPTIONAL |

### 8.33.2 Cross-validation comparison

| Model | Best parameters | Mean PR-AUC | Std PR-AUC | Mean ROC-AUC | Mean F1 | Mean recall | Mean precision | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| Logistic Regression | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| Random Forest | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |

### 8.33.3 Formal-holdout evaluation

| Selected pipeline | Threshold | PR-AUC | ROC-AUC | Accuracy | Balanced accuracy | Precision | Recall | F1 | Brier score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |

### 8.33.4 Confusion matrix

| Actual / Predicted | Predicted False | Predicted True |
|---|---:|---:|
| Actual False | TO BE GENERATED | TO BE GENERATED |
| Actual True | TO BE GENERATED | TO BE GENERATED |

### 8.33.5 Fairness and subgroup template

| Model variant | Group | N | Positive prevalence | Predicted-positive rate | Precision | Recall | False-positive rate | Caveat |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Full model | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| No Gender | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |
| No Gender/Age | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED | TO BE GENERATED |

### 8.33.6 Explainability and error analysis template

- Selected model: **TO BE GENERATED**
- Stable important features: **TO BE GENERATED**
- Unstable or misleading importances: **TO BE GENERATED**
- False-positive patterns: **TO BE GENERATED**
- False-negative patterns: **TO BE GENERATED**
- Calibration finding: **TO BE GENERATED**
- Causal claims permitted: **None**

### 8.33.7 Final conclusion template

- Evidence of useful predictive signal: **TO BE GENERATED**
- Best justified model or baseline: **TO BE GENERATED**
- Deployment recommendation: **TO BE GENERATED**, with default presumption against deployment unless strong, stable, ethical evidence exists.
- Main dataset limitations: **TO BE GENERATED from verified and formal findings**
- Recommended future data: **TO BE GENERATED**
- Lessons learned: **TO BE GENERATED**

## 8.34 AI-assistant rules for code and writing

### Code rules

- Use functions, type hints, docstrings, validation, explicit random seeds, and deterministic file paths.
- Prefer scikit-learn `Pipeline` and `ColumnTransformer`.
- Never fit preprocessing on the full dataset before CV.
- Keep target mapping explicit and tested.
- Save machine-readable metrics and experiment metadata.
- Do not suppress warnings without explanation.
- Do not alter the raw CSV.
- Avoid adding dependencies not listed in an approved change.
- Include minimal tests for every major pipeline component.

### Writing rules

- Tie claims to source IDs, code outputs, or saved results.
- Keep **VERIFIED** facts separate from **DECIDED** methods and **TO BE GENERATED** results.
- Use precise language about imbalance, uncertainty, and no-signal outcomes.
- Do not write generic textbook paragraphs when project-specific evidence is available.
- Do not present feature importance as causality.
- Do not conceal dataset artefacts to make the project appear stronger.
- Match all values across report, slides, README, model card, and context.

## 8.35 AI-tool plan and actual-use live record

### 8.35.1 Preliminary AI-tool plan

| Tool | Planned tasks | Estimated extent | Verification and ownership |
| --- | --- | --- | --- |
| ChatGPT | Requirement interpretation, context architecture, dataset-audit assistance, problem-framing review, code/debugging assistance, EDA and visualization review, report and slide editing, viva preparation | Moderate; update if actual use changes | Group checks formal sources, executes code, reviews outputs, rewrites where necessary, and retains responsibility. |

Do not list Copilot, Gemini, Claude, or another tool until the group actually plans to use it. Add each tool as a separate record with version, tasks, extent, verification, and retained evidence.

### 8.35.2 Actual AI-tool usage - LIVE RECORD

| Record ID | Date | Tool/version | Task supported | Extent | Verification performed | Human ownership/status |
| --- | --- | --- | --- | --- | --- | --- |
| AI-001 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Interpreted assignment and proposal requirements; clarified presentation duration, report requirement, slide planning, and six-member division. | Moderate | Checked against the formal PDFs. | User reviewed discussion; group remains responsible. |
| AI-002 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Mechanically audited the uploaded archive/CSV and identified schema, checksums, class balance, deterministic relationships, and provenance limitations. | Moderate | Used code-based inspection; PDF pages visually inspected; source URL checked. | User approved all recommendations. |
| AI-003 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Recommended target, methodology, models, metrics, ethics, team roles, repository, proposal structure, and no-signal contingency. | Moderate | Recommendations tied to rubric and audit evidence; user explicitly approved all. | DECIDED by user approval. |
| AI-004 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Generated the canonical Markdown and synchronized Word project-context files. | Moderate | YAML validation, content checks, DOCX rendering and visual QA. | Files remain subject to group use and update. |
| AI-005 | 2026-07-28 | ChatGPT - GPT-5.6 Thinking | Upgraded the context to v3: embedded complete formal-source transcripts, added structured live state and exact data contract, documented preliminary audit exposure, fixed evaluation/fairness rules, synchronized Markdown/Word, and packaged the files. | Extensive | Validated YAML and headings, checked source checksums/transcripts, compared substantive DOCX text to Markdown, rendered and visually inspected every Word page. | User requested the update; group remains responsible for maintaining future live records. |
| AI-006 | 2026-07-29 | ChatGPT - GPT-5.6 Thinking | Updated the active project title to "Design and Evaluation of Machine Learning Models for Employee Resignation Prediction", clarified the exact assessed build and non-deliverables, synchronized the Markdown and Word context files, and revalidated the handoff. | Moderate | Applied deterministic content checks, parsed the machine-readable YAML, rendered the Word file, and visually inspected all pages. | User approved the title and requested the updated AI-only handoff; the group owns future maintenance. |

Future records must include the actual tool version if known, precise tasks, extent, verification, retained evidence, and group responsibility. Honest disclosure is assessed, not the volume of AI use.

## 8.36 Decision and change log

| Decision ID | Date | Status | Topic | Current value | Evidence/authority | Approved by | Reconsider if |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q-001 | 2026-07-28 | DECIDED | Primary problem and target | Binary resignation classification framed as evaluation of signal | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-002 | 2026-07-28 | SUPERSEDED | Project name | Evaluating Machine Learning for Employee Resignation Prediction | Phase 1 audit, formal requirements, and dataset evidence | User approval | Superseded by Q-023 on 2026-07-29 |
| Q-003 | 2026-07-28 | DECIDED | Scope | Analytical assignment workflow only; no deployment | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-004 | 2026-07-28 | DECIDED | Provenance wording | Undocumented and likely constructed; no real-data claim | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-005 | 2026-07-28 | DECIDED | Feature policy | Drop ID; audit/drop raw date; retain tenure; pipeline encoding/scaling | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-006 | 2026-07-28 | DECIDED | Split/CV | 80/20 stratified; seed 42; 5-fold development CV primary; formal holdout one-time corroboration with historical exposure caveat | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-007 | 2026-07-28 | DECIDED | Models | Dummy, Logistic Regression, Random Forest; optional Decision Tree | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-008 | 2026-07-28 | DECIDED | Metrics | PR-AUC primary with approved secondary metrics | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-009 | 2026-07-28 | DECIDED | Imbalance | Class weights and threshold analysis; no pre-split SMOTE | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-010 | 2026-07-28 | DECIDED | No-signal contingency | Report a defensible negative result and reject deployment | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-011 | 2026-07-28 | DECIDED | Ethics/fairness | Academic use only; sensitive-feature and subgroup analysis | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-012 | 2026-07-28 | DECIDED | Explainability | Coefficients, permutation importance, error analysis; no causal claims | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-013 | 2026-07-28 | DECIDED | Technology/repository | Python/scikit-learn reproducible structure | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-014 | 2026-07-28 | DECIDED | Proposal/report plan | Ten slides, 13-14 minutes; formal report order | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-015 | 2026-07-28 | DECIDED | Team map | Six role slots with peer review; names placeholders | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-016 | 2026-07-28 | DECIDED | AI tools | Record ChatGPT planned and actual use | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-017 | 2026-07-28 | DECIDED | Administrative details | Preserve official-only items as placeholders | Phase 1 audit, formal requirements, and dataset evidence | User approval | Higher-authority instruction or new verified evidence |
| Q-018 | 2026-07-28 | DECIDED | Self-contained source requirements | Embed complete normalized transcripts of both formal PDFs; original PDFs required only for visual/revision verification | User request for two-file AI handoff; verified PDFs | User request to create updated files | A newer official revision is supplied |
| Q-019 | 2026-07-28 | DECIDED | Preliminary audit exposure | Preserve audit metrics and disclose missing split metadata; do not claim a historically pristine holdout | Methodological integrity review | User request to apply improvements | Reproducible prior audit row IDs/code become available |
| Q-020 | 2026-07-28 | DECIDED | Evaluation precision | Development CV is primary; one-standard-error tie-break; formal holdout is one-time corroboration; exact threshold/bootstrap rules fixed | Methodological review and assignment requirements | User request to apply improvements | Higher-authority requirement or justified logged experiment-plan change |
| Q-021 | 2026-07-28 | DECIDED | Fairness reporting thresholds | Fixed age bands and minimum subgroup denominators | Reproducibility and instability control | User request to apply improvements | Exact file changes or higher-authority instruction |
| Q-022 | 2026-07-28 | DECIDED | AI handoff structure | Markdown canonical; Word same substantive content; startup acknowledgement and stable/live-state precedence required | User's AI-only handoff objective | User request to create updated files | Handoff format changes |
| Q-023 | 2026-07-29 | DECIDED | Project title and exact assessed build | Use "Design and Evaluation of Machine Learning Models for Employee Resignation Prediction". Build an end-to-end reproducible Python ML workflow and evaluation package; do not treat a GUI, application, API, database, or deployment as required. | Assignment specification, proposal requirements, and explicit user decision | User approval in conversation | Higher-authority instruction or an explicitly logged scope change |

### Change log

| Version | Date | Change | Rationale | Status |
|---|---|---|---|---|
| 2.0.0 | 2026-07-28 | Created AI-first canonical context and synchronized Word companion after approval of Q-001 to Q-017 handling. | Provide a complete handoff package and fixed methodology before implementation. | SUPERSEDED |
| 3.0.0 | 2026-07-28 | Embedded complete formal-source transcripts; expanded machine-readable live state/data contract; documented preliminary audit exposure; fixed selection, threshold, bootstrap, calibration, fairness, and synchronization rules; regenerated and revalidated Word companion. | Make the two-file package self-contained, less ambiguous, and safer for future AI execution. | SUPERSEDED |
| 3.1.0 | 2026-07-29 | Adopted the assignment-aligned title "Design and Evaluation of Machine Learning Models for Employee Resignation Prediction"; added a machine-readable and narrative exact-build definition; explicitly separated the required Python ML workflow from optional/deployed software products; added SRC-007, Q-023, and AI-006; regenerated and revalidated the synchronized Word companion. | Remove ambiguity about what the assignment requires the group to build while preserving the approved target, dataset, methodology, and no-signal contingency. | ACTIVE |

## 8.37 Remaining questions requiring human or external input

| ID | Question | Recommended handling | Why not auto-decided | How to confirm | Blocks current work? |
| --- | --- | --- | --- | --- | --- |
| EXT-001 | Exact proposal date and lab time | Use official announced value; retain tentative week until then. | Only lecturer/LMS can set schedule. | Save LMS announcement or lecturer response. | No |
| EXT-002 | Must all six members speak? | Plan for all six until confirmed. | Formal document is silent. | Ask lecturer/module coordinator. | No |
| EXT-003 | Is Q&A within or after the 15 minutes? | Keep 1-2 minute delivery buffer; do not label it Q&A. | Formal document is silent. | Ask lecturer/module coordinator. | No |
| EXT-004 | Final report deadline and timezone | Use official portal value only. | Not supplied. | LMS/portal announcement. | No |
| EXT-005 | Viva date, format, and duration | Use official value only. | Not supplied. | LMS/lecturer. | No |
| EXT-006 | Submission platform and file naming | Use official instructions. | Not supplied. | Portal/LMS guidance. | No |
| EXT-007 | Citation style | Use official specified style; otherwise request confirmation. | Not supplied and institution convention is unsafe to assume. | Rubric/LMS/lecturer. | No |
| EXT-008 | Whether title, references, and appendices count in 15 pages | Use official clarification. | Only assessment authority can interpret limit. | Ask lecturer/module coordinator. | No |
| EXT-009 | Group number, lab batch, team-member names/student IDs, and role mapping | Fill identity fields after group agreement and official confirmation. | Personal/administrative identity cannot be inferred. | Group and LMS/timetable. | No |
| EXT-009A | Module coordinator or lecturer identity for formal records | Use official module information. | Title/name may change and is not in supplied PDFs. | LMS/module page. | No |
| EXT-010 | Repository path/URL | Create and record shared repository. | Not yet created or supplied. | Group setup. | No |
| EXT-011 | Official institutional AI/academic-integrity policy and any AI-text citation rule | Apply official policy in addition to the supplied disclosure requirement. | Not included in supplied evidence. | Institution/LMS/lecturer. | No for development; yes before final submission |
| EXT-012 | Required submission archive structure | Use official guidance. | Not supplied. | LMS/portal. | No |

## 8.38 References and source-file register

### Formal and project sources

1. **SRC-001:** Faculty of Computing. *IT2011 - Artificial Intelligence and Machine Learning: Group Assignment Specification*. Year 2 Semester 1, 2026. Local file: `Group Assignment Specification.pdf`. SHA-256: `7bcd43f095c4e76d89d941ee6c28da228260b6cebb70ae56c623e6d55867b481`.
2. **SRC-002:** Faculty of Computing. *IT2011 - Artificial Intelligence and Machine Learning: Proposal Presentation*. Year 2 Semester 1, 2026. Local file: `Proposal Presentation.pdf`. SHA-256: `7cbf695c122b194880beb62a672b15eb56eaa470d43bf286a3794e9c10abb4d3`.
3. **SRC-003:** Uploaded assigned archive, `archive.zip`. SHA-256: `abe4a9aa869fe553faec813a06ff93432900d72af2036058f8b84e3dd64db375`.
4. **SRC-004:** `Extended_Employee_Performance_and_Productivity_Data.csv`. SHA-256: `2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb`.
5. **SRC-005:** Kaggle. *Employee Performance and Productivity Data*. Dataset page: [Kaggle dataset page](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data). Listing observed as CC0: Public Domain.
6. **SRC-006:** User approval in project conversation, 2026-07-28: "Approve all recommendations."
7. **SRC-007:** User project-title and handoff update decision in project conversation, 2026-07-29: use "Design and Evaluation of Machine Learning Models for Employee Resignation Prediction" and update both context files for AI-only handoff.

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

