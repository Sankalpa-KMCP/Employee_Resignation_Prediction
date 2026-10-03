# Group Notebooks

These notebooks represent integrated group work rather than one member's headline model.

1. `01_group_pipeline.ipynb` — consolidated demonstration of the six preprocessing contributions. It fits transformations on all development rows for explanation, exports no dataset or transformers, and is separate from the models' fold-specific training preprocessing.
2. `02_model_comparison.ipynb` — six-model OOF comparison and final development conclusions.
3. `03_calibration.ipynb` — raw OOF reliability, probability error, fold diagnostics and calibration decision.
4. `04_fairness_analysis.ipynb` — subgroup denominators/errors, fixed-capacity representation and descriptive gaps.
5. `05_retention_prioritization.ipynb` — exploratory ranking, fold-fitted proxy sensitivity and repeated-random allocation comparisons.
6. `06_final_holdout_evaluation.ipynb` — safe readiness report by default; one-time prediction corroboration after a verified development freeze and explicit activation.

The final holdout notebook is intentionally guarded and must not be unlocked simply to produce more results before model selection and comparison are complete.

Run the group notebooks in numerical order from a clean kernel. See [evaluation guidance](../../docs/EVALUATION_GUIDE.md) and the [final holdout protocol](../../docs/FINAL_HOLDOUT_PROTOCOL.md) for result interpretation and the supported freeze contract.
