"""Explicit year-built input contracts; never infer encoding from value ranges."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

ABSOLUTE_CALENDAR_YEAR = "absolute_calendar_year"
LEAD_YEAR_OFFSET_1900_UINT8_V1 = "lead_year_offset_1900_uint8_v1"
YEAR_OFFSET = 1900
MISSING_YEAR_CODE = 255


def require_year_built_encoding(encoding: str) -> None:
    if encoding not in (ABSOLUTE_CALENDAR_YEAR, LEAD_YEAR_OFFSET_1900_UINT8_V1):
        raise ValueError(f"unsupported year_built encoding: {encoding!r}")


def year_built_values(column: pd.Series, encoding: str) -> np.ndarray:
    """Validate raw codes/calendar years without modifying the source column."""
    require_year_built_encoding(encoding)
    if (not is_numeric_dtype(column.dtype) or is_bool_dtype(column.dtype)
            or is_complex_dtype(column.dtype)):
        raise TypeError("year_built must have a real numeric, nonboolean dtype")
    values = column.to_numpy(dtype=float, na_value=np.nan)
    known = values[~np.isnan(values)]
    low, high = (1, 9999) if encoding == ABSOLUTE_CALENDAR_YEAR else (0, 255)
    if (not np.isfinite(known).all()
            or ((known < low) | (known > high) | (known != np.floor(known))).any()):
        raise ValueError(f"year_built must be integer-valued in [{low}, {high}] or missing")
    return values


def missing_values(column: pd.Series, encoding: str) -> np.ndarray:
    """Only year_built has a declared sentinel; zero floors remain known."""
    require_year_built_encoding(encoding)
    missing = column.isna().to_numpy(dtype=bool)
    if column.name == "year_built" and encoding == LEAD_YEAR_OFFSET_1900_UINT8_V1:
        missing = missing | column.eq(MISSING_YEAR_CODE).fillna(False).to_numpy(dtype=bool)
    return missing


def construction_years(column: pd.Series, encoding: str) -> np.ndarray:
    """Decode into float before arithmetic, masking sentinel and native nulls."""
    values = year_built_values(column, encoding).copy()
    if encoding == LEAD_YEAR_OFFSET_1900_UINT8_V1:
        values[values == MISSING_YEAR_CODE] = np.nan
        values += YEAR_OFFSET
    return values
