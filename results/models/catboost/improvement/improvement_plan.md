# CatBoost improvement plan (written before any Phase 3 run)

Evidence base: `audit_report.md`, `phase1/*.json|csv`, `baseline_snapshot.json`.
Primary metric: pooled out-of-fold **PR-AUC** (average precision); ties broken by Log Loss, then Brier.
Statistical support: 95% **paired bootstrap** (1,000 resamples, seed 42, shared resample indices) of the difference vs the *current best* model, plus the paired fold-wise difference (5 folds) as a secondary check.

## What the audit says the problem is

| Finding | Class | Consequence for the plan |
|---|---|---|
| Raw OOF Brier is high because CatBoost's Logloss starts at p = 0.5 (`boost_from_average` off) **and** folds picked different candidates (two Balanced winners), so the pooled OOF mixes probability scales | B (+D) | Fix the probability scale first (F1); pooled metrics are only meaningful once fold scales agree |
| Early stopping/selection on inner PR-AUC (≈960 positives) is noise-driven: trees per fold range 1–67 | B | Replace the criterion (F2) |
| Every probe variant gives PR-AUC ≈ prevalence and ROC-AUC ≈ 0.5; no predictor has single-feature AUC outside 0.49–0.51; no CI excludes the Dummy prior | C | Expect **no** discrimination gain; the plan is designed to detect one if it exists, not to manufacture one |
| Notebook parses holdout rows, hard-codes numbers, lacks figures, reports a single "winner" | A/D | Reporting fixes (F9), independent of model gains |

## Ordered fixes

| # | Fix (class) | Hypothesis | Exact change | Expected effect | Cost | Pre-declared success criterion |
|---|---|---|---|---|---|---|
| F1 | Probability-scale fix (A/B) | Starting from p = 0.5 makes small-tree models output ≈0.5 and makes Brier/Log loss depend on tree count and class weights rather than on skill | `configs/models/catboost.yaml` → `fixed_params.boost_from_average: true` (experiment **E1**) | Fold-mean probabilities ≈ prevalence; raw Brier ≈ Dummy prior's; ranking metrics unchanged | 1 nested CV (~4 min) | **Kept** iff every fold-mean probability is within 0.02 of prevalence **and** lower bound of ΔPR-AUC vs the baseline protocol > −0.005. This is a correctness fix, so it is kept even if PR-AUC does not rise |
| F2 | Early stopping & candidate selection on inner Log Loss, patience 200 (B) | Smooth, calibration-sensitive criterion on 12,000 inner rows is less noisy than PR-AUC | `eval_metric: Logloss`; `inner_validation.selection_metric: log_loss`; `early_stopping_rounds: 200` (**E2**) | More stable tree counts across folds; PR-AUC unchanged if there is no signal | 1 nested CV | Kept iff ΔPR-AUC 95% paired CI lower bound > 0 vs current best; otherwise kept only if Log Loss is significantly lower (CI upper < 0) at tied PR-AUC; else reverted |
| F3 | Class-weight strategy (B) | Weighting changes the ranking only through tree structure; calibration is then repaired post hoc | Four runs, each with the same 3 tree shapes: none / `Balanced` / `SqrtBalanced` / `scale_pos_weight=3` (**E3a–E3d**). Counts as **one** change for the stopping rule | None expected | 4 small nested CVs | Same as F2, per variant against current best |
| F4 | Wider random search inside the inner split (B) — *conditional* | Current 8-candidate grid is too narrow | 40 random trials: depth 2–8, l2 1–30, random_strength 0.2–3, bagging_temperature 0–2, border_count, one_hot_max_size, rsm; 3 repeated inner splits (**E4**). Random search chosen over Optuna to avoid a new dependency (AGENTS.md: ask before adding heavy dependencies) | Possibly better inner fit; outer gain unlikely | ~1 h | Same as F2 |
| F5 | Seed averaging, 5 final seeds (B) — *conditional* | Reduces variance of the refit model | `ensemble.final_seeds: [42..46]` (**E5**) | Tiny | 5× refit cost | Same as F2 |
| F6 | Shallow / strongly regularised fallback (B/C) — *conditional* | If signal is absent a depth-2–4, l2 10–30 model is the honest choice | Pool restricted to depth 2–4 | none | — | Same as F2; also reported as the simple-model recommendation |
| F7 | Calibration comparison (B) | Platt/isotonic cross-fitted improves Brier/Log loss without changing ranking | Post hoc on final raw OOF: none vs Platt vs isotonic; choose lowest cross-fitted pooled Log loss, then Brier | Brier/Log loss → Dummy prior level | seconds | Reported with CIs; no stop-rule effect (does not change ranking) |
| F8 | Capacity-based decision rule (D) | Operating point should come from an explicit capacity, not 0.5 | Report exact top-k tables (k = 1, 5, 10, 20, 30, 50 %); capacity chosen by HR budget, never tuned on holdout | n/a | — | n/a |
| F9 | Reporting fixes (A/D) | Notebook violates rule 1/5 and lacks required figures | New `scripts/run_catboost_audit.py`, `src/evaluation/audit_stats.py`, `audit_plots.py`; rebuild notebook (canonical loader, no holdout frame, computed narrative, 17 figures with captions); manifest rows appended, nothing deleted | — | — | All tests in `tests/test_catboost_audit.py` pass; notebook executes with zero errors; manifest artifacts exist and reconcile to 1e-5 |
| F10 | Feature engineering / `Hire_Date`-derived tenure (C) | May add signal | **Not performed.** Adding predictors is a prohibited silent change (`change_control`); recommendation only | — | — | Requires a documented shared decision |

Categorical options (`one_hot_max_size`, `max_ctr_complexity`) are only explored inside F4's random search; all four categoricals have ≤ 9 levels, so high-cardinality handling of `Job_Title` is not a real issue.

## Stopping rule (pre-declared)

Stop when **two consecutive justified changes** (F2, then F3 as one change, then F4, …) give no significant gain, or when the label-shuffle permutation test shows the best model is indistinguishable from shuffled labels. Experiments not run because the rule fired are listed as *not run* in the final report, not as failures.

## Verification plan

Label-shuffle test: ≥ 10 permutations of the development labels through the **complete** nested protocol of the final configuration; real score vs shuffled distribution with add-one permutation p-value. Determinism: final protocol twice with the same seed, diff of OOF. Holdout: never loaded (audit code uses `load_development_data`).

## Amendment 1 (recorded after E1, before E2 ran)

E1 (`boost_from_average=True` alone) was **reverted** by its own pre-declared criterion: the fold-mean probability check failed because candidate selection on inner PR-AUC still picks *Balanced* candidates in some folds, and class weighting pushes probabilities toward 0.5 independently of `boost_from_average`. The scale defect therefore has two causes, and the plan's F1 criterion cannot be met until the selection criterion (F2) stops preferring weighted candidates.

Change to the plan (no criterion is relaxed): after E2 is judged, **E1b** re-tests `boost_from_average=True` on top of the then-current best with the **identical** F1 criterion (fold-mean probability within 0.02 of prevalence and lower bound of ΔPR-AUC > −0.005). E1 itself stays in the log as reverted. As a fix (not a performance change) E1b does not count toward the two-consecutive-no-gain stopping rule.

## Amendment 2 (recorded after E1b/E3, before E4 ran)

State after E1, E2, E1b, E3a–d (see `experiment_log.csv`): kept E2 (Log-Loss tie-break) and E1b (scale fix); the E3 group (one change) was reverted, so the no-gain streak is 1. E4 is therefore run next, as planned.

Budget change: the planned 40 random trials × 3 repeated inner splits × 5 outer folds = 600 inner fits is projected at ~3 h of CPU (a single candidate fit took ≈17 s in E2). E4 is run with **20 trials × 3 repeated inner splits** (300 fits, ≈1.5 h). The spec allows any fixed budget; the reduced budget makes a negative result weaker evidence than 40 trials would, and that is stated in the final report. The decision criterion is unchanged.

## Amendment 3 (recorded after E4, before the permutation test and before E5/E6)

E4 (20-trial random search, 3 repeated inner splits, Log-Loss selection) was **kept** by the pre-declared rule: ΔPR-AUC vs E1b has a 95% paired-bootstrap lower bound above 0. Two cautions are recorded *before* the permutation test so they cannot be argued after the fact:

* The reference (E1b) had a pooled PR-AUC *below* the prevalence, so a positive delta against it is not evidence of skill. The deciding comparison is against the Dummy prior and against the label-shuffled null.
* Fold-wise paired p-value for E4 vs E1b is far from significant, and the selected models have 1–57 trees with near-constant outputs.

Stopping-rule trigger 2 (permutation test on the best model) is evaluated **now**, on the E4 protocol, before spending ~1 h each on E5 (seed averaging) and E6 (shallow fallback). If the real E4 score is inside the shuffled-label distribution (add-one permutation p ≥ 0.05 for PR-AUC and ROC-AUC) the run stops and E5/E6/categorical-option experiments are reported as *not run*. If the test is significant, E5 and E6 are run next, in order.

The final protocol is written to `configs/models/audited_catboost.yaml` (the older `configs/models/catboost.yaml` is left untouched because existing tests and group notebooks depend on it). Replacing the canonical config/OOF with the audited ones is a team decision and is listed as an open question in the final report.

The permutation test runs 10 permutations in parallel processes (1 CatBoost thread each; benchmark in the scratchpad showed 24 threads ≈ 8 threads ≈ 23–25 s per fit, 1 thread ≈ 67 s, so parallel single-thread processes give higher throughput). Thread count affects only speed, not the statistical protocol.
