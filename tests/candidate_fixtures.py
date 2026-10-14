"""Generated 16-row diagnostic protocol; no filesystem or real-data access.

Eight opaque building groups have two timestamped rows and both classes each.
Two whole-building outer folds retain four training groups, enough for 2/3 inner
folds with two-class support. Row indices are fixture identities, not predictors.
Every helper returns fresh objects; callers keep scoring labels separate.
"""

import numpy as np
import pandas as pd

from project.experiments.candidate_contracts import KEY_COLUMNS, FoldPartition

CORE_COLUMNS = (
    "meter_reading", "square_feet", "year_built", "floor_count", "hour",
    "weekday", "month", "year", "primary_use",
)


def core_frame() -> pd.DataFrame:
    """Return exactly 16 generated core rows plus separate group metadata attrs."""
    group_number = np.repeat(np.arange(8), 2)
    hour = np.tile([0, 1], 8)
    frame = pd.DataFrame({
        "meter_reading": 10. + group_number + 10 * hour,
        "square_feet": 1000. + 10 * group_number,
        "year_built": 1980 + group_number,
        "floor_count": 1 + group_number % 4,
        "hour": hour, "weekday": np.zeros(16, dtype=int),
        "month": np.ones(16, dtype=int), "year": np.full(16, 2024),
        "primary_use": ["office", "retail", "school", "office"] * 4,
    }, columns=CORE_COLUMNS, index=pd.Index(np.arange(16), name="row_index"))
    frame.attrs["groups"] = pd.Series([f"building_{g:02d}" for g in group_number],
                                      index=frame.index, name="building_id")
    return frame


def binary_target(frame: pd.DataFrame) -> pd.Series:
    """Generated labels follow fixture row identities, never predictor values."""
    if (not isinstance(frame, pd.DataFrame) or not frame.index.is_unique
            or frame.index.dtype.kind not in "iu" or (frame.index < 0).any()
            or (frame.index >= 16).any()):
        raise ValueError("binary_target requires a subset of the generated fixture row identities")
    return pd.Series((frame.index.to_numpy() % 2).astype(np.int64), index=frame.index, name="anomaly")


def synthetic_partitions() -> tuple[FoldPartition, ...]:
    """Return two explicit diagnostic outer folds, not alternative real splits."""
    frame = core_frame()
    groups, target = frame.attrs["groups"], binary_target(frame)
    timestamp = pd.Timestamp("2024-01-01") + pd.to_timedelta(frame.hour, unit="h")
    keys = pd.DataFrame({"row_index": frame.index, "building_id": groups,
                         "timestamp": timestamp, "repeat": 0, "fold": 0}, index=frame.index)
    partitions = []
    for fold in (0, 1):
        validation = frame.index[(frame.index // 2) % 2 == fold]
        training = frame.index[(frame.index // 2) % 2 != fold]
        validation_keys = keys.loc[validation, list(KEY_COLUMNS)].copy()
        validation_keys["fold"] = fold
        partitions.append(FoldPartition(0, fold, frame.loc[training], target.loc[training],
                                        groups.loc[training], frame.loc[validation], validation_keys))
    return tuple(partitions)
