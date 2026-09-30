"""Append-only experiment inventory writer and validation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REQUIRED_INVENTORY_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "hypothesis_id",
    "varied_factor_id",
    "baseline_id",
    "candidate_id",
    "recipe_id",
    "model_config_id",
    "data_sha256",
    "split_sha256",
    "config_sha256",
    "primary_metric",
    "oof_path",
    "wall_time_seconds",
    "outcome",
    "rejection_reason",
    "run_id",
    "timestamp",
)


def append_inventory_records(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Validate and append inventory records to an append-only JSONL file.

    Parameters
    ----------
    path : str | Path
        Path to the target JSONL file.
    records : Iterable[Mapping[str, Any]]
        A sequence or iterable of experiment records. Every record must be a
        mapping containing all required Task 6 fields and be JSON-serializable.

    Raises
    ------
    TypeError
        If a record is not a mapping.
    ValueError
        If any record is missing required fields or cannot be serialized to JSON.
        The file is left unchanged if validation fails on any record.
    """
    record_list = list(records)
    if not record_list:
        return

    serialized_lines: list[str] = []
    for idx, record in enumerate(record_list):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"Record at index {idx} must be a mapping, got {type(record).__name__}"
            )
        missing_fields = [f for f in REQUIRED_INVENTORY_FIELDS if f not in record]
        if missing_fields:
            raise ValueError(
                f"Record at index {idx} is missing required fields: {missing_fields}"
            )
        try:
            line = json.dumps(record)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Record at index {idx} is not JSON serializable: {err}"
            ) from err
        serialized_lines.append(line)

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    prefix = ""
    if target_path.exists() and target_path.stat().st_size > 0:
        with target_path.open("rb") as f_check:
            f_check.seek(-1, 2)
            if f_check.read(1) != b"\n":
                prefix = "\n"

    with target_path.open("a", encoding="utf-8") as f:
        if prefix:
            f.write(prefix)
        for line in serialized_lines:
            f.write(f"{line}\n")


__all__ = [
    "REQUIRED_INVENTORY_FIELDS",
    "append_inventory_records",
]
