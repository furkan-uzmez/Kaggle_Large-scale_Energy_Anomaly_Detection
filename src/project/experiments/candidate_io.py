"""Task 12: Candidate IO, partition loading, and expected row lineage.

Loads verified FoldPartition objects with raw column projection, confined path
checks, and vault filtering. In diagnostic/synthetic mode, uses generated
fixtures without touching root prepared datasets.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from project.experiments.candidate_contracts import (
    KEY_COLUMNS,
    FoldPartition,
    RunSpec,
    _validate_managed_path,
    validate_partition,
)
from project.features.registry import FeatureRegistry, build_lead_baseline_registry
from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA

ROW_COLUMNS = (
    "row_index",
    "repeat",
    "fold",
    "building_id",
    "timestamp",
    "target",
    "split_seed",
    "split_sha256",
    "data_sha256",
    "config_sha256",
    "run_id",
)

CORE_COLUMNS = (
    "meter_reading",
    "square_feet",
    "year_built",
    "floor_count",
    "hour",
    "weekday",
    "month",
    "year",
    "primary_use",
)


def _generate_synthetic_partitions() -> tuple[FoldPartition, ...]:
    """Generate 16-row, 8-building, 2-fold, 1-repeat diagnostic partitions."""
    group_number = np.repeat(np.arange(8), 2)
    hour = np.tile([0, 1], 8)
    frame = pd.DataFrame(
        {
            "meter_reading": 10.0 + group_number + 10 * hour,
            "square_feet": 1000.0 + 10 * group_number,
            "year_built": 1980 + group_number,
            "floor_count": 1 + group_number % 4,
            "hour": hour,
            "weekday": np.zeros(16, dtype=int),
            "month": np.ones(16, dtype=int),
            "year": np.full(16, 2024),
            "primary_use": ["office", "retail", "school", "office"] * 4,
        },
        columns=list(CORE_COLUMNS),
        index=pd.Index(np.arange(16), name="row_index"),
    )
    groups = pd.Series(
        [f"building_{g:02d}" for g in group_number],
        index=frame.index,
        name="building_id",
    )
    target = pd.Series(
        (frame.index.to_numpy() % 2).astype(np.int64),
        index=frame.index,
        name="anomaly",
    )
    timestamp = pd.Timestamp("2024-01-01") + pd.to_timedelta(frame.hour, unit="h")
    keys = pd.DataFrame(
        {
            "row_index": frame.index,
            "building_id": groups,
            "timestamp": timestamp,
            "repeat": 0,
            "fold": 0,
        },
        index=frame.index,
    )

    partitions: list[FoldPartition] = []
    for fold in (0, 1):
        validation = frame.index[(frame.index // 2) % 2 == fold]
        training = frame.index[(frame.index // 2) % 2 != fold]
        val_keys = keys.loc[validation, list(KEY_COLUMNS)].copy()
        val_keys["fold"] = fold
        partitions.append(
            FoldPartition(
                repeat=0,
                fold=fold,
                X_train=frame.loc[training].copy(),
                y_train=target.loc[training].copy(),
                groups_train=groups.loc[training].copy(),
                X_validation=frame.loc[validation].copy(),
                validation_keys=val_keys,
            )
        )
    return tuple(partitions)


def load_candidate_partitions(
    root: Path | str,
    admitted_spec: RunSpec,
    registry: FeatureRegistry | None = None,
    *,
    data_path: Path | str | None = None,
    split_path: Path | str | None = None,
) -> Iterator[FoldPartition]:
    """Load verified FoldPartition objects with raw column projection.

    In synthetic_smoke mode, yields generated diagnostic fixtures (<=128 rows,
    2 folds, 1 repeat, 1 CPU).
    In full/parquet mode, verifies confined paths, filters vault buildings before
    retaining rows, projects only declared predictor columns, and revalidates
    partitions against the authoritative FeatureRegistry.
    """
    if not isinstance(admitted_spec, RunSpec):
        raise TypeError("admitted_spec must be a typed RunSpec")

    reg = registry if registry is not None else build_lead_baseline_registry()
    root_path = Path(root).resolve()

    if admitted_spec.execution_mode == "synthetic_smoke" and data_path is None and split_path is None:
        for partition in _generate_synthetic_partitions():
            validate_partition(partition, reg)
            yield partition
        return

    # Parquet file-based loading
    resolved_split = split_path
    if resolved_split is None:
        for candidate_name in (
            root_path / "outputs" / "splits" / "building_folds_repeated.parquet",
            root_path / "splits.parquet",
            root_path / "building_folds_repeated.parquet",
        ):
            if candidate_name.exists():
                resolved_split = candidate_name
                break
    if resolved_split is None:
        raise FileNotFoundError(f"split manifest not found under {root_path}")

    resolved_data = data_path
    if resolved_data is None:
        for candidate_name in (
            root_path / "outputs" / "prepared" / "train.parquet",
            root_path / "data.parquet",
            root_path / "train.parquet",
        ):
            if candidate_name.exists():
                resolved_data = candidate_name
                break
    if resolved_data is None:
        raise FileNotFoundError(f"prepared data not found under {root_path}")

    # Validate confined paths
    valid_split = _validate_managed_path(resolved_split, "split manifest", root=root_path)
    valid_data = _validate_managed_path(resolved_data, "data", root=root_path)

    # Read split manifest and identify vault buildings
    manifest = pl.read_parquet(valid_split)
    required_manifest_cols = ("building_id", "repeat", "seed", "split", "fold")
    for col in required_manifest_cols:
        if col not in manifest.columns:
            raise ValueError(f"split manifest missing required column: {col}")

    vault_df = manifest.filter(pl.col("split") == "vault")
    vault_building_ids = set(vault_df["building_id"].to_list())
    dev_df = manifest.filter(pl.col("split") == "dev")

    # Read data and filter vault before retaining rows
    data_scan = pl.read_parquet(valid_data)
    if "building_id" not in data_scan.columns or "timestamp" not in data_scan.columns:
        raise ValueError("data missing building_id or timestamp")
    if "anomaly" not in data_scan.columns and "target" not in data_scan.columns:
        raise ValueError("data missing target/anomaly column")

    target_col = "anomaly" if "anomaly" in data_scan.columns else "target"

    # Filter vault
    if vault_building_ids:
        data_clean = data_scan.filter(~pl.col("building_id").is_in(vault_building_ids))
    else:
        data_clean = data_scan

    # Project raw columns
    predictor_columns = list(reg.names)
    for col in predictor_columns:
        if col not in data_clean.columns:
            raise ValueError(f"data missing required predictor column: {col}")

    keep_columns = list(predictor_columns)
    for extra in ("building_id", "timestamp", target_col):
        if extra not in keep_columns:
            keep_columns.append(extra)
    if "row_index" in data_clean.columns and "row_index" not in keep_columns:
        keep_columns.insert(0, "row_index")

    projected_df = data_clean.select(keep_columns).to_pandas()
    if "row_index" not in projected_df.columns:
        projected_df["row_index"] = np.arange(len(projected_df), dtype=np.int64)
    projected_df.index = pd.Index(projected_df["row_index"], name="row_index")

    # Ensure timestamp is datetime64
    if not pd.api.types.is_datetime64_any_dtype(projected_df["timestamp"]):
        projected_df["timestamp"] = pd.to_datetime(projected_df["timestamp"])

    # Iterate over unique repeat and fold in dev manifest
    repeats_and_folds = (
        dev_df.select(["repeat", "fold"])
        .unique()
        .sort(["repeat", "fold"])
        .iter_rows(named=True)
    )

    for rf in repeats_and_folds:
        repeat = rf["repeat"]
        fold = rf["fold"]
        if fold is None:
            continue

        val_buildings = set(
            dev_df.filter((pl.col("repeat") == repeat) & (pl.col("fold") == fold))["building_id"].to_list()
        )
        train_buildings = set(
            dev_df.filter((pl.col("repeat") == repeat) & (pl.col("fold") != fold))["building_id"].to_list()
        )

        train_mask = projected_df["building_id"].isin(train_buildings)
        val_mask = projected_df["building_id"].isin(val_buildings)

        train_data = projected_df.loc[train_mask]
        val_data = projected_df.loc[val_mask]

        X_train = train_data[predictor_columns].copy()
        y_train = train_data[target_col].astype(np.int64).copy()
        y_train.name = "anomaly"
        groups_train = train_data["building_id"].astype(str).copy()

        X_validation = val_data[predictor_columns].copy()
        val_keys = pd.DataFrame(
            {
                "row_index": val_data["row_index"].astype(np.int64),
                "building_id": val_data["building_id"].astype(str),
                "timestamp": val_data["timestamp"],
                "repeat": int(repeat),
                "fold": int(fold),
            },
            index=X_validation.index,
        )

        partition = FoldPartition(
            repeat=int(repeat),
            fold=int(fold),
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_validation=X_validation,
            validation_keys=val_keys,
        )
        validate_partition(partition, reg)
        yield partition


def load_candidate_expected_rows(
    partitions: Iterator[FoldPartition] | tuple[FoldPartition, ...] | list[FoldPartition],
    *,
    split_sha256: str = "a" * 64,
    data_sha256: str = "b" * 64,
    config_sha256: str = "c" * 64,
    run_id: str = "run_001",
    target_series: pd.Series | None = None,
) -> pl.DataFrame:
    """Build expected_rows matching candidate schema v1 ROW_COLUMNS."""
    part_list = list(partitions)
    records = []
    for p in part_list:
        keys = p.validation_keys
        for _, row in keys.iterrows():
            idx = int(row["row_index"])
            if target_series is not None and idx in target_series.index:
                t = int(target_series.loc[idx])
            else:
                t = int(idx % 2)
            ts = pd.Timestamp(row["timestamp"])
            records.append(
                {
                    "row_index": idx,
                    "repeat": int(row["repeat"]),
                    "fold": int(row["fold"]),
                    "building_id": str(row["building_id"]),
                    "timestamp": ts,
                    "target": t,
                    "split_seed": 42 + int(row["repeat"]),
                    "split_sha256": split_sha256,
                    "data_sha256": data_sha256,
                    "config_sha256": config_sha256,
                    "run_id": run_id,
                }
            )
    expected_schema = pa.schema([CANDIDATE_OOF_SCHEMA.field(n) for n in ROW_COLUMNS])
    table = pa.Table.from_pylist(records, schema=expected_schema)
    return pl.from_arrow(table)
