"""Validation utilities for data integrity, schemas, hashes, keys, and contracts."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

SUPPORTED_SCHEMA_TYPES: dict[str, Any] = {
    "Int64": lambda dt: "int64" in str(dt).lower(),
    "Float64": lambda dt: "float64" in str(dt).lower(),
    "Datetime": lambda dt: "datetime" in str(dt).lower(),
    "String": lambda dt: "str" in str(dt).lower() or "string" in str(dt).lower(),
}


def compute_file_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute sha256 hex digest of a file in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def count_file_rows(path: Path | str) -> int:
    """Count data rows in a CSV file excluding the header line, streaming row-by-row."""
    path_obj = Path(path)
    with open(path_obj, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # Skip header
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def validate_file_hash(path: Path | str, expected_sha256: str) -> bool:
    """Check if file matches expected sha256."""
    actual_sha256 = compute_file_sha256(path)
    return actual_sha256.lower() == expected_sha256.lower()


def validate_raw_file_metadata(
    path: Path | str,
    expected_sha256: str,
    expected_rows: int,
) -> dict[str, Any]:
    """Validate file hash and row count against contract; raise ValueError if mismatched."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Source file not found: {path_obj}")

    actual_sha256 = compute_file_sha256(path_obj)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Hash mismatch for {path_obj.name}: expected {expected_sha256}, got {actual_sha256}"
        )

    actual_rows = count_file_rows(path_obj)
    if actual_rows != expected_rows:
        raise ValueError(
            f"Row count mismatch for {path_obj.name}: expected {expected_rows}, got {actual_rows}"
        )

    return {
        "filename": path_obj.name,
        "path": str(path_obj),
        "sha256": actual_sha256,
        "rows": actual_rows,
        "bytes": path_obj.stat().st_size,
    }


def validate_schema(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    required_columns: Sequence[str],
    forbidden_columns: Sequence[str] | None = None,
    table_name: str = "table",
) -> dict[str, str]:
    """Validate schema columns and return snapshot mapping column name to string dtype."""
    schema = df_or_lf.collect_schema() if isinstance(df_or_lf, pl.LazyFrame) else df_or_lf.schema

    missing_cols = [c for c in required_columns if c not in schema]
    if missing_cols:
        raise ValueError(f"{table_name} missing required columns: {missing_cols}")

    if forbidden_columns:
        present_forbidden = [c for c in forbidden_columns if c in schema]
        if present_forbidden:
            raise ValueError(f"{table_name} contains forbidden columns: {present_forbidden}")

    return {col: str(dtype) for col, dtype in schema.items()}


def validate_schema_types(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    schema_types: Mapping[str, str],
    table_name: str = "table",
) -> None:
    """Validate that configured columns have expected data types.

    Fails closed if a configured column is missing or an expected type name is unsupported.
    """
    schema = df_or_lf.collect_schema() if isinstance(df_or_lf, pl.LazyFrame) else df_or_lf.schema
    for col, expected_type in schema_types.items():
        if expected_type not in SUPPORTED_SCHEMA_TYPES:
            raise ValueError(
                f"Unsupported expected type '{expected_type}' for column '{col}' in {table_name}"
            )
        if col not in schema:
            raise ValueError(
                f"Configured type column '{col}' missing from {table_name}"
            )
        actual_type = str(schema[col])
        if not SUPPORTED_SCHEMA_TYPES[expected_type](schema[col]):
            raise TypeError(
                f"{table_name} column '{col}' expected {expected_type} type, got {actual_type}"
            )


def validate_keys_uniqueness(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    key_cols: Sequence[str],
    table_name: str = "table",
) -> None:
    """Validate that key columns uniquely identify each row without full in-memory collection."""
    schema = df_or_lf.collect_schema() if isinstance(df_or_lf, pl.LazyFrame) else df_or_lf.schema
    for col in key_cols:
        if col not in schema:
            raise ValueError(f"Key column '{col}' missing from {table_name}")

    lf = df_or_lf.lazy() if isinstance(df_or_lf, pl.DataFrame) else df_or_lf
    dup_count = (
        lf.group_by(key_cols)
        .len()
        .filter(pl.col("len") > 1)
        .select(pl.col("len").sum())
        .collect(engine="streaming")
        .item()
    )
    dup_count = dup_count if dup_count is not None else 0
    if dup_count > 0:
        keys_str = ", ".join(key_cols)
        raise ValueError(
            f"{table_name} key ({keys_str}) is not unique; found {dup_count} duplicate key rows"
        )


def validate_target_presence(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    target_col: str = "anomaly",
    should_be_present: bool = True,
    table_name: str = "table",
) -> None:
    """Validate presence or absence of target column."""
    schema = df_or_lf.collect_schema() if isinstance(df_or_lf, pl.LazyFrame) else df_or_lf.schema
    if should_be_present and target_col not in schema:
        raise ValueError(f"Target column '{target_col}' missing from {table_name}")
    if not should_be_present and target_col in schema:
        raise ValueError(f"Target column '{target_col}' must not be present in {table_name}")


def validate_building_disjointness(
    train_df: pl.DataFrame | pl.LazyFrame,
    test_df: pl.DataFrame | pl.LazyFrame,
    building_col: str = "building_id",
) -> None:
    """Validate that train and test building IDs have no overlap without collecting full tables."""
    train_schema = train_df.collect_schema() if isinstance(train_df, pl.LazyFrame) else train_df.schema
    test_schema = test_df.collect_schema() if isinstance(test_df, pl.LazyFrame) else test_df.schema

    if building_col not in train_schema:
        raise ValueError(f"Building column '{building_col}' missing from train table")
    if building_col not in test_schema:
        raise ValueError(f"Building column '{building_col}' missing from test table")

    tr_lf = train_df.lazy() if isinstance(train_df, pl.DataFrame) else train_df
    ts_lf = test_df.lazy() if isinstance(test_df, pl.DataFrame) else test_df

    overlap_count = (
        tr_lf.select(building_col)
        .unique()
        .join(ts_lf.select(building_col).unique(), on=building_col, how="inner")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if overlap_count > 0:
        raise ValueError(
            f"Train and test share {overlap_count} {building_col} values (overlap detected)"
        )


def validate_duplicate_column_equality(
    df1: pl.DataFrame | pl.LazyFrame,
    df2: pl.DataFrame | pl.LazyFrame,
    col_name: str,
    join_keys: Sequence[str],
    name1: str = "df1",
    name2: str = "df2",
) -> None:
    """Validate that declared duplicate copy in df2 exactly matches df1 along join keys.

    Fails closed if the declared duplicate copy is missing from either table.
    """
    schema1 = df1.collect_schema() if isinstance(df1, pl.LazyFrame) else df1.schema
    schema2 = df2.collect_schema() if isinstance(df2, pl.LazyFrame) else df2.schema

    if col_name not in schema1:
        raise ValueError(f"Table {name1} missing declared duplicate copy column '{col_name}'")
    if col_name not in schema2:
        raise ValueError(f"Table {name2} missing declared duplicate copy column '{col_name}'")

    lf1 = df1.lazy() if isinstance(df1, pl.DataFrame) else df1
    lf2 = df2.lazy() if isinstance(df2, pl.DataFrame) else df2

    mismatches = (
        lf1.select([*join_keys, col_name])
        .join(lf2.select([*join_keys, col_name]), on=join_keys, how="inner", suffix="_dup")
        .filter(pl.col(col_name).ne_missing(pl.col(f"{col_name}_dup")))
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if mismatches > 0:
        raise ValueError(
            f"Duplicate column '{col_name}' values in {name2} do not match {name1} ({mismatches} mismatches)"
        )


def validate_sample_submission_alignment(
    test_df: pl.DataFrame | pl.LazyFrame,
    sample_sub_df: pl.DataFrame | pl.LazyFrame,
    row_id_col: str = "row_id",
) -> None:
    """Validate that test row_id order exactly matches sample submission without full collection."""
    ts_lf = test_df.lazy() if isinstance(test_df, pl.DataFrame) else test_df
    sub_lf = sample_sub_df.lazy() if isinstance(sample_sub_df, pl.DataFrame) else sample_sub_df

    ts_len = ts_lf.select(pl.len()).collect(engine="streaming").item()
    sub_len = sub_lf.select(pl.len()).collect(engine="streaming").item()
    if ts_len != sub_len:
        raise ValueError(f"test row count ({ts_len}) does not match sample_submission ({sub_len})")

    mismatches = (
        ts_lf.select(row_id_col)
        .with_row_index("idx")
        .join(sub_lf.select(row_id_col).with_row_index("idx"), on="idx", suffix="_sub")
        .filter(pl.col(row_id_col).ne_missing(pl.col(f"{row_id_col}_sub")))
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if mismatches > 0:
        raise ValueError(
            f"test row_id does not match sample_submission row_id order ({mismatches} mismatches)"
        )
