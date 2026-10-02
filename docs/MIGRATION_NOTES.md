# Architecture Refactor Migration Notes

This repository was refactored from the review ZIP without overwriting the original package.

## Changed

- Preserved `notebooks/team_contributions/` as a first-class concept.
- Moved the six preprocessing notebooks to `notebooks/team_contributions/preprocessing/`.
- Moved the six member model notebooks to `notebooks/team_contributions/models/member_<n>_<model>/`.
- Moved integrated group notebooks to `notebooks/group/`.
- Updated moved notebooks to locate the repository root robustly instead of depending on fixed `../` depth where necessary.
- Moved the redundant root `outputs/monthly_salary_before_after_scaling.png` to `results/preprocessing/standardization/`.
- Moved external-review packaging files under `docs/internal_review/`.
- Moved the old evaluation guide under `docs/`.
- Updated notebook-architecture tests for the new paths.
- Repaired the LightGBM sklearn `fit()` call from unsupported `eval_X`/`eval_y` arguments to the supported `eval_set=[(X_val, y_val)]` form. This is a compatibility bug fix; stored historical result artifacts were not regenerated.

## Intentionally unchanged

- Raw and processed datasets.
- Stored notebook outputs.
- Model result artifacts and historical result-folder layouts.
- Model algorithms and saved experimental measurements.
- Historical project-context documents under `docs/context/`.

## Historical path warning

Archived/context documentation may mention the old notebook paths. Those documents describe earlier repository states and were preserved for traceability.
