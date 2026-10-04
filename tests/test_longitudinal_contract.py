"""Tests for the future longitudinal snapshot safety contract."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.longitudinal import (
    TARGET_COLUMN,
    audit_longitudinal_schema,
    validate_snapshot_frame,
    validate_temporal_split_order,
)


def test_current_static_schema_cannot_support_a_90_day_target():
    audit = audit_longitudinal_schema(["Employee_ID", "Hire_Date", "Resigned"])
    assert audit["supports_90_day_target"] is False
    assert audit["missing_required_columns"] == ["resignation_date", "snapshot_date"]


def test_target_is_strictly_future_and_within_90_days():
    frame = pd.DataFrame({
        "Employee_ID": [1, 2, 3],
        "snapshot_date": ["2026-01-01", "2026-01-01", "2026-01-01"],
        "resignation_date": ["2026-04-01", "2026-04-02", None],
        "salary": [100.0, 200.0, 300.0],
        "salary_effective_date": ["2025-12-01", "2025-12-01", "2025-12-01"],
    })
    result = validate_snapshot_frame(
        frame,
        feature_columns=["salary"],
        feature_availability_columns={"salary": "salary_effective_date"},
        data_as_of_date="2026-07-01",
    )
    assert result[TARGET_COLUMN].tolist() == [1, 0, 0]


def test_future_feature_timestamp_and_identifier_feature_are_rejected():
    frame = pd.DataFrame({
        "Employee_ID": [1],
        "snapshot_date": ["2026-01-01"],
        "resignation_date": [None],
        "salary": [100.0],
        "salary_effective_date": ["2026-01-02"],
    })
    with pytest.raises(ValueError, match="after snapshot_date"):
        validate_snapshot_frame(
            frame,
            feature_columns=["salary"],
            feature_availability_columns={"salary": "salary_effective_date"},
            data_as_of_date="2026-07-01",
        )
    with pytest.raises(ValueError, match="Prohibited"):
        validate_snapshot_frame(
            frame,
            feature_columns=["Employee_ID"],
            feature_availability_columns={"Employee_ID": "salary_effective_date"},
            data_as_of_date="2026-07-01",
        )


def test_temporal_split_registry_requires_strict_ordering():
    valid = pd.DataFrame({
        "Employee_ID": [1, 2, 3],
        "snapshot_date": ["2026-01-01", "2026-02-01", "2026-03-01"],
        "partition": ["train", "validation", "holdout"],
    })
    validate_temporal_split_order(valid)
    invalid = valid.copy()
    invalid.loc[0, "snapshot_date"] = "2026-02-15"
    with pytest.raises(ValueError, match="not strictly ordered"):
        validate_temporal_split_order(invalid)
