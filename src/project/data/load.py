"""Bounded-memory data ingestion using Polars lazy scans and explicit dtypes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml

from project.paths import ProjectPaths, find_project_root

DEFAULT_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "building_id": pl.Int64,
    "row_id": pl.Int64,
    "meter_reading": pl.Float64,
    "anomaly": pl.Int64,
}


def load_data_contract(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load data contract YAML configuration."""
    if config_path is None:
        root = find_project_root()
        path = root / "configs" / "data_contract.yaml"
    else:
        path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Data contract config not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        contract = yaml.safe_load(f)
    return contract


def scan_raw_table(
    path: Path | str,
    schema_overrides: dict[str, pl.DataType] | None = None,
) -> pl.LazyFrame:
    """Scan a CSV file lazily with automatic datetime parsing and explicit key/target dtypes."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"CSV file not found: {path_obj}")

    overrides = dict(DEFAULT_SCHEMA_OVERRIDES)
    if schema_overrides:
        overrides.update(schema_overrides)

    return pl.scan_csv(
        path_obj,
        try_parse_dates=True,
        schema_overrides=overrides,
    )


def scan_all_raw_tables(
    paths: ProjectPaths | None = None,
) -> dict[str, pl.LazyFrame]:
    """Scan all five competition CSV files lazily."""
    p = paths or ProjectPaths.resolve()
    return {
        "train": scan_raw_table(p.train_csv),
        "test": scan_raw_table(p.test_csv),
        "sample_submission": scan_raw_table(p.sample_submission_csv),
        "train_features": scan_raw_table(p.train_features_csv),
        "test_features": scan_raw_table(p.test_features_csv),
    }
