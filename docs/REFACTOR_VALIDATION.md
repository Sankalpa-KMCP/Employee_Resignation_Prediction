# Upgrade Validation Record

The repository was reviewed after structural refactoring and then upgraded through a second review/fix cycle.

## Checks completed

- raw dataset integrity and schema;
- frozen split/outer-fold registry;
- six-model OOF Employee_ID and target alignment;
- no formal-holdout overlap in comparison inputs;
- model configs and active notebook paths;
- group comparison, calibration, fairness and prioritization notebook execution;
- XGBoost nested tuning evidence and rerunnable script;
- LightGBM API compatibility fix;
- active documentation consistency;
- Python compilation and automated tests;
- ZIP structure/integrity before delivery.

## Automated test state in review environment

`pytest -q` reaches the test suite successfully because `pytest.ini` now exposes the repository root. The environment passes all non-FT tests. FT-specific execution tests cannot run because `rtdl-revisiting-models` is unavailable in the review environment; network package installation was also unavailable. This is recorded as an environment limitation rather than silently treated as a passing test.

See `docs/FINAL_VALIDATION_STATUS.md` for unresolved evidence items.
