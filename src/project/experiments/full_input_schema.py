"""Approved raw full-campaign names; ordering is not feature engineering."""

from collections.abc import Mapping

from project.features.provided import WEATHER_FEATURES

CORE_NUMERIC = (
    "meter_reading", "square_feet", "year_built", "floor_count",
    "hour", "weekday", "month", "year",
)
CORE_CATEGORICAL = ("primary_use",)
CORE = CORE_NUMERIC + CORE_CATEGORICAL
from project.features.registry import build_lead_baseline_registry
LEAD_CORE = build_lead_baseline_registry().names
PROVIDED_POOL = LEAD_CORE + WEATHER_FEATURES


def canonical_feature_order(names: tuple[str, ...]) -> tuple[str, ...]:
    """Reorder only the complete exact approved pool; never mutate a spec."""
    if (not isinstance(names, tuple)
            or any(not isinstance(name, str) for name in names)):
        raise ValueError("feature pool must be an immutable tuple of strings")
    if set(names) == set(CORE):
        return CORE
    if set(names) == set(PROVIDED_POOL):
        return PROVIDED_POOL
    raise ValueError("feature pool must contain exactly the approved core names or provided pool names")

def validate_input_dtypes(schema: Mapping) -> None:
    """Require the existing raw scalar/storage policy without altering values.

    In particular year_built remains numeric, including missing values; no new
    date interpretation, sentinel, imputation or rescaling is introduced.
    """
    import polars as pl

    if schema.get("building_id") not in (pl.String, pl.Int64):
        raise ValueError("unsupported opaque building dtype; string/int64 required")
    integers = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    for name in (*CORE_NUMERIC, *WEATHER_FEATURES):
        if name in schema and schema.get(name) not in (*integers, pl.Float32, pl.Float64):
            raise ValueError(f"unsupported numeric predictor dtype: {name}")
    if schema.get("primary_use") != pl.String:
        raise ValueError("unsupported categorical predictor dtype: primary_use")
