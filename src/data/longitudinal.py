"""Leakage-safe validation primitives for a future longitudinal snapshot dataset.

This module intentionally does not manufacture snapshots from the current static
source. It becomes usable only after dated employee records are supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging

import pandas as pd


LOGGER = logging.getLogger(__name__)
HORIZON_DAYS = 90
IDENTIFIER_COLUMN = "Employee_ID"
SNAPSHOT_COLUMN = "snapshot_date"
RESIGNATION_COLUMN = "resignation_date"
TARGET_COLUMN = "resigned_within_90_days"
REQUIRED_SNAPSHOT_COLUMNS = {
    IDENTIFIER_COLUMN,
    SNAPSHOT_COLUMN,
    RESIGNATION_COLUMN,
}
PROHIBITED_FEATURE_TOKENS = (
    "employee_id",
    "resigned",
    "resignation_date",
    "exit",
    "final_work",
    "last_work",
    "severance",
    "termination",
    "settlement",
    "notice_period",
)


def audit_longitudinal_schema(columns: Sequence[str]) -> dict[str, object]:
    """Return the missing structural fields without reading or exposing row values."""
    observed = set(columns)
    missing = sorted(REQUIRED_SNAPSHOT_COLUMNS - observed)
    return {
        "observed_columns": sorted(observed),
        "required_columns": sorted(REQUIRED_SNAPSHOT_COLUMNS),
        "missing_required_columns": missing,
        "supports_90_day_target": not missing,
    }


def _as_datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], errors="raise", utc=True)
    if values.isna().any():
        raise ValueError(f"{column} contains missing timestamps.")
    return values


def validate_snapshot_frame(
    snapshots: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    feature_availability_columns: Mapping[str, str],
    data_as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Validate dated snapshots and derive the strictly future 90-day target.

    The returned frame excludes snapshots whose outcome window has not closed or
    that occur on/after resignation. It logs shape changes for auditability.
    """
    missing = REQUIRED_SNAPSHOT_COLUMNS - set(snapshots.columns)
    if missing:
        raise ValueError(f"Longitudinal snapshots missing required columns: {sorted(missing)}")
    if snapshots[IDENTIFIER_COLUMN].isna().any():
        raise ValueError("Employee_ID must be complete in a snapshot dataset.")

    feature_columns = list(feature_columns)
    prohibited = [
        feature for feature in feature_columns
        if any(token in feature.lower() for token in PROHIBITED_FEATURE_TOKENS)
    ]
    if prohibited:
        raise ValueError(f"Prohibited model features: {sorted(prohibited)}")
    missing_features = set(feature_columns) - set(snapshots.columns)
    if missing_features:
        raise ValueError(f"Snapshot dataset missing features: {sorted(missing_features)}")
    missing_availability = set(feature_columns) - set(feature_availability_columns)
    if missing_availability:
        raise ValueError(
            "Every model feature requires an availability timestamp mapping: "
            f"{sorted(missing_availability)}"
        )

    checked = snapshots.copy()
    LOGGER.info("Snapshot shape before temporal filtering: %s", checked.shape)
    checked[SNAPSHOT_COLUMN] = _as_datetime(checked, SNAPSHOT_COLUMN)
    checked[RESIGNATION_COLUMN] = pd.to_datetime(
        checked[RESIGNATION_COLUMN], errors="raise", utc=True
    )
    as_of = pd.Timestamp(data_as_of_date, tz="UTC")
    if checked[SNAPSHOT_COLUMN].gt(as_of).any():
        raise ValueError("snapshot_date cannot be later than data_as_of_date.")

    for feature, available_at_column in feature_availability_columns.items():
        if available_at_column not in checked.columns:
            raise ValueError(
                f"Feature {feature} maps to missing availability column {available_at_column}."
            )
        available_at = _as_datetime(checked, available_at_column)
        if available_at.gt(checked[SNAPSHOT_COLUMN]).any():
            raise ValueError(
                f"Feature {feature} has values available after snapshot_date."
            )

    before_resignation = (
        checked[RESIGNATION_COLUMN].isna()
        | checked[SNAPSHOT_COLUMN].lt(checked[RESIGNATION_COLUMN])
    )
    outcome_closed = checked[SNAPSHOT_COLUMN].add(pd.Timedelta(days=HORIZON_DAYS)).le(as_of)
    eligible = checked.loc[before_resignation & outcome_closed].copy()
    LOGGER.info("Snapshot shape after resignation/outcome-window filtering: %s", eligible.shape)
    upper_bound = eligible[SNAPSHOT_COLUMN].add(pd.Timedelta(days=HORIZON_DAYS))
    eligible[TARGET_COLUMN] = (
        eligible[RESIGNATION_COLUMN].gt(eligible[SNAPSHOT_COLUMN])
        & eligible[RESIGNATION_COLUMN].le(upper_bound)
    ).astype(int)
    return eligible


def validate_temporal_split_order(registry: pd.DataFrame) -> None:
    """Require strictly ordered temporal partitions without inspecting model features."""
    required = {IDENTIFIER_COLUMN, SNAPSHOT_COLUMN, "partition"}
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"Temporal split registry missing: {sorted(missing)}")
    dates = _as_datetime(registry, SNAPSHOT_COLUMN)
    partitions = ("train", "validation", "holdout")
    ranges = {}
    for partition in partitions:
        partition_dates = dates.loc[registry["partition"] == partition]
        if partition_dates.empty:
            raise ValueError(f"Temporal split registry has no {partition} rows.")
        ranges[partition] = (partition_dates.min(), partition_dates.max())
    if not (ranges["train"][1] < ranges["validation"][0] <= ranges["validation"][1] < ranges["holdout"][0]):
        raise ValueError("Temporal partitions overlap or are not strictly ordered.")
