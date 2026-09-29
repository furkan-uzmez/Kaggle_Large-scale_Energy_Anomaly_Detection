"""Data loading, validation, and joining utilities."""

from __future__ import annotations

from project.data.join import build_feature_joins
from project.data.load import (
    load_data_contract,
    scan_all_raw_tables,
    scan_raw_table,
)
from project.data.validate import (
    compute_file_sha256,
    count_file_rows,
    validate_building_disjointness,
    validate_duplicate_column_equality,
    validate_file_hash,
    validate_keys_uniqueness,
    validate_raw_file_metadata,
    validate_sample_submission_alignment,
    validate_schema,
    validate_schema_types,
    validate_target_presence,
)

__all__ = [
    "build_feature_joins",
    "compute_file_sha256",
    "count_file_rows",
    "load_data_contract",
    "scan_all_raw_tables",
    "scan_raw_table",
    "validate_building_disjointness",
    "validate_duplicate_column_equality",
    "validate_file_hash",
    "validate_keys_uniqueness",
    "validate_raw_file_metadata",
    "validate_sample_submission_alignment",
    "validate_schema",
    "validate_schema_types",
    "validate_target_presence",
]
