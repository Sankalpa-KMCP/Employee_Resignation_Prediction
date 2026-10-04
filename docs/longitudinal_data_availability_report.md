# Longitudinal 90-Day Resignation-Risk Data Availability Report

## Decision

**Training is blocked.** The current raw source is a static, one-row-per-
employee table with a retrospective `Resigned` label. It has no observation or
snapshot date and no resignation event date. Therefore the statement
“resigns after a snapshot and within the next 90 days” cannot be evaluated
without inventing event timing. No processed snapshot dataset, temporal split
registry, OOF prediction, calibration, threshold, or model result was created.

This report was produced from the header/schema only; it contains no employee
values.

## Current schema classification

| Fields | Classification | Condition / rationale |
|---|---|---|
| `Employee_ID` | Traceability only; prohibited model feature | Needed to join dated events and deterministic top-k ties. |
| `Resigned` | Historical retrospective target only | It has no resignation date and cannot construct a 90-day future target. |
| `Hire_Date` | Candidate pre-snapshot source | May derive tenure only relative to a supplied `snapshot_date`. |
| `Department`, `Gender`, `Job_Title`, `Education_Level` | Candidate pre-snapshot categorical features | Require effective/availability dates; do not assume current values were known at every prior snapshot. |
| `Age`, `Years_At_Company` | Candidate pre-snapshot derived/static fields | Must be calculated or recorded as of `snapshot_date`; do not use future values. |
| `Performance_Score`, `Monthly_Salary`, `Work_Hours_Per_Week`, `Projects_Handled`, `Overtime_Hours`, `Sick_Days`, `Remote_Work_Frequency`, `Team_Size`, `Training_Hours`, `Promotions`, `Employee_Satisfaction_Score` | Candidate pre-snapshot features only | The source does not state their measurement or availability dates, period boundaries, or histories. |
| Exit date, final work date, severance, termination reason, final settlement | Prohibited future/post-resignation features | Not present in this source; retain as permanently prohibited model inputs. |

## Missing data required before training

1. A repeated employee snapshot/observation date or dated event ledger.
2. Resignation effective date for every observed resignation, plus a data-as-of date so each 90-day outcome window can close.
3. Dated department, role, manager, team, and location history.
4. Dated salary, promotion, performance, satisfaction/engagement, workload/overtime, leave/absence, training, and internal-mobility histories.
5. A feature-availability timestamp for each source field or an auditable effective-date rule proving it is no later than the snapshot date.

## Scaffold delivered

- `configs/longitudinal_snapshot_contract.yaml` defines row grain, target semantics, prohibited fields, temporal split policy, and missing sources.
- `src/data/longitudinal.py` validates availability timestamps, the closed 90-day outcome window, post-resignation exclusion, feature restrictions, and temporal ordering.
- `data/processed/longitudinal/v1/` is a versioned empty output location that preserves the existing frozen registry and artifacts.

## Required decision

Provide the dated sources above, or explicitly select a different target horizon
that can be supported by available timestamps. Until then, the existing static
model remains a historical baseline only and must not be described as future
resignation-risk prediction.
