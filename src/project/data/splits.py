"""Building-isolated development and vault split construction."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

STRATIFICATION_METHOD = (
    "StratifiedKFold over contiguous, merged anomaly-count support strata"
)
SPLIT_RATIONALE = (
    "Buildings are the assignment unit. The vault is isolated before model work; "
    "the remaining building groups form development validation folds. Per-building "
    "anomaly-count strata preserve representation of rare support levels without "
    "splitting rows from a building across partitions."
)
MAX_RANDOM_SEED = 2**32 - 1


def _validate_columns(
    train: pl.DataFrame | pl.LazyFrame,
    building_col: str,
    target_col: str,
) -> pl.LazyFrame:
    schema = train.collect_schema() if isinstance(train, pl.LazyFrame) else train.schema
    missing_columns = [
        column for column in (building_col, target_col) if column not in schema
    ]
    if missing_columns:
        raise ValueError(f"Training data is missing split columns: {missing_columns}")
    if building_col == target_col:
        raise ValueError("building_col and target_col must name different columns")
    if schema[target_col] != pl.Null and not schema[target_col].is_integer():
        raise ValueError(
            f"Target column '{target_col}' must contain integer binary labels 0/1; "
            f"got {schema[target_col]}"
        )
    return (
        train.select(building_col, target_col).lazy()
        if isinstance(train, pl.DataFrame)
        else train.select(building_col, target_col)
    )


def _summarize_buildings(
    train: pl.DataFrame | pl.LazyFrame,
    building_col: str,
    target_col: str,
) -> pl.DataFrame:
    projected = _validate_columns(train, building_col, target_col)
    summary = (
        projected.group_by(building_col)
        .agg(
            pl.len().cast(pl.Int64).alias("row_count"),
            pl.col(target_col).null_count().cast(pl.Int64).alias("null_target_rows"),
            (pl.col(target_col).is_not_null() & ~pl.col(target_col).is_in([0, 1]))
            .sum()
            .cast(pl.Int64)
            .alias("invalid_target_rows"),
            pl.col(target_col).sum().cast(pl.Int64).alias("anomaly_count"),
            (pl.col(target_col) == 0).sum().cast(pl.Int64).alias("normal_count"),
        )
        .collect(engine="streaming")
        .sort(building_col)
    )

    null_buildings = summary.filter(pl.col(building_col).is_null()).height
    if null_buildings:
        raise ValueError(
            f"Building column '{building_col}' contains null values in {null_buildings} group(s)"
        )

    null_targets = int(summary.get_column("null_target_rows").sum() or 0)
    if null_targets:
        raise ValueError(
            f"Target column '{target_col}' contains {null_targets} null row(s)"
        )

    invalid_targets = int(summary.get_column("invalid_target_rows").sum() or 0)
    if invalid_targets:
        raise ValueError(
            f"Target column '{target_col}' must be binary with values 0/1; "
            f"found {invalid_targets} invalid row(s)"
        )

    if summary.is_empty():
        raise ValueError("Training data contains no building rows")

    return summary.drop("null_target_rows", "invalid_target_rows").with_columns(
        (pl.col("anomaly_count") / pl.col("row_count")).alias("anomaly_prevalence")
    )


def _merge_sparse_support_bins(
    support_counts: Sequence[int],
    minimum_bin_size: int,
) -> list[tuple[int, int, int]]:
    histogram = Counter(support_counts)
    bins: list[tuple[int, int, int]] = []
    lower_support: int | None = None
    upper_support: int | None = None
    bin_size = 0

    for support, count in sorted(histogram.items()):
        if bin_size >= minimum_bin_size:
            bins.append((int(lower_support), int(upper_support), bin_size))
            lower_support = None
            bin_size = 0
        if lower_support is None:
            lower_support = support
        upper_support = support
        bin_size += count

    if lower_support is not None and upper_support is not None:
        bins.append((lower_support, upper_support, bin_size))

    if len(bins) > 1 and bins[-1][2] < minimum_bin_size:
        previous_bin = bins[-2]
        final_bin = bins[-1]
        bins[-2] = (previous_bin[0], final_bin[1], previous_bin[2] + final_bin[2])
        bins.pop()

    return bins


def _assign_support_strata(
    summary: pl.DataFrame,
    partition_count: int,
) -> tuple[pl.DataFrame, list[tuple[int, int, int]]]:
    support_counts = [
        int(value) for value in summary.get_column("anomaly_count").to_list()
    ]
    strata = _merge_sparse_support_bins(support_counts, partition_count)
    if not strata or any(size < partition_count for _, _, size in strata):
        raise ValueError(
            "Building-level anomaly-support rarity strata cannot be formed with at least "
            f"{partition_count} buildings per stratum; found {len(summary)} buildings"
        )

    stratum_by_support: dict[int, int] = {}
    for stratum_id, (lower, upper, _size) in enumerate(strata):
        for support_count in set(support_counts):
            if lower <= support_count <= upper:
                stratum_by_support[support_count] = stratum_id

    labels = [stratum_by_support[count] for count in support_counts]
    return summary.with_columns(
        pl.Series("support_stratum", labels, dtype=pl.Int64)
    ), strata


def _validate_split_parameters(
    building_count: int,
    vault_fraction: float,
    n_dev_folds: int,
) -> int:
    if isinstance(vault_fraction, bool) or not isinstance(vault_fraction, Real):
        raise TypeError(
            f"vault_fraction must be a real number between 0 and 1, got {vault_fraction!r}"
        )
    if not 0 < vault_fraction < 1:
        raise ValueError(
            f"vault_fraction must be between 0 and 1, got {vault_fraction}"
        )
    if isinstance(n_dev_folds, bool) or not isinstance(n_dev_folds, Integral):
        raise TypeError(
            f"n_dev_folds must be an integer >= 1, got {n_dev_folds!r}"
        )
    if n_dev_folds < 1:
        raise ValueError(f"n_dev_folds must be at least 1, got {n_dev_folds}")

    partition_count = n_dev_folds + 1
    expected_vault_fraction = 1 / partition_count
    if not math.isclose(vault_fraction, expected_vault_fraction, abs_tol=1e-9):
        raise ValueError(
            "Equal-sized support-stratified groups require vault_fraction to equal "
            f"1 / (n_dev_folds + 1) = {expected_vault_fraction:.6g}; got {vault_fraction}"
        )
    if building_count < partition_count:
        raise ValueError(
            f"Building-level split requires at least {partition_count} buildings; "
            f"found {building_count}"
        )
    if building_count % partition_count:
        raise ValueError(
            f"Building-level split requires {building_count} buildings to divide evenly "
            f"across {partition_count} equal groups"
        )
    return partition_count


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    if not 0 <= seed <= MAX_RANDOM_SEED:
        raise ValueError(
            f"seed must be between 0 and {MAX_RANDOM_SEED}, got {seed}"
        )


def _validate_partition_support(manifest: pl.DataFrame, partition_count: int) -> None:
    for group_id in range(partition_count):
        group = manifest.filter(pl.col("group_id") == group_id)
        positive_rows = int(group.get_column("anomaly_count").sum() or 0)
        negative_rows = int(group.get_column("normal_count").sum() or 0)
        split_label = "vault" if group_id == 0 else f"dev fold {group_id - 1}"
        if positive_rows == 0 or negative_rows == 0:
            absent_classes = []
            if positive_rows == 0:
                absent_classes.append("positive")
            if negative_rows == 0:
                absent_classes.append("negative")
            raise ValueError(
                f"Building-level split infeasible: {split_label} has no "
                f"{' and '.join(absent_classes)} target rows; every partition requires "
                "both classes. No row-level fallback was attempted."
            )


def build_split_manifest(
    train: pl.DataFrame | pl.LazyFrame,
    building_col: str = "building_id",
    target_col: str = "anomaly",
    vault_fraction: float = 0.2,
    n_dev_folds: int = 4,
    seed: int = 42,
) -> pl.DataFrame:
    """Build one support-aware, deterministic split assignment per building.

    Raw rows are lazily aggregated to one compact per-building summary before any
    assignment. Exact anomaly-count bins are merged with adjacent support bins until
    every resulting rarity stratum can contribute to each equally-sized group.
    Group zero is the vault; subsequent groups are development folds.
    """
    _validate_seed(seed)
    summary = _summarize_buildings(train, building_col, target_col)
    partition_count = _validate_split_parameters(
        summary.height,
        vault_fraction,
        n_dev_folds,
    )

    positive_buildings = int((summary.get_column("anomaly_count") > 0).sum())
    negative_buildings = int((summary.get_column("normal_count") > 0).sum())
    if positive_buildings < partition_count:
        raise ValueError(
            f"Positive support has only {positive_buildings} buildings; {partition_count} "
            "disjoint partitions are required by the building-level split."
        )
    if negative_buildings < partition_count:
        raise ValueError(
            f"Negative support has only {negative_buildings} buildings; {partition_count} "
            "disjoint partitions are required by the building-level split."
        )

    stratified_summary, _strata = _assign_support_strata(summary, partition_count)
    stratum_labels = np.asarray(
        stratified_summary.get_column("support_stratum").to_list()
    )
    group_ids = np.full(stratified_summary.height, -1, dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=partition_count, shuffle=True, random_state=seed
    )
    for group_id, (_remaining_indices, partition_indices) in enumerate(
        splitter.split(np.zeros(stratified_summary.height), stratum_labels)
    ):
        group_ids[partition_indices] = group_id

    if (group_ids < 0).any():
        raise RuntimeError("Stratified assignment did not assign every building")

    manifest = stratified_summary.with_columns(
        pl.Series("group_id", group_ids, dtype=pl.Int64)
    ).with_columns(
        pl.when(pl.col("group_id") == 0)
        .then(pl.lit("vault"))
        .otherwise(pl.lit("dev"))
        .alias("split"),
        pl.when(pl.col("group_id") == 0)
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col("group_id") - 1)
        .alias("fold"),
    )

    _validate_partition_support(manifest, partition_count)
    return manifest.drop("group_id").sort(building_col)


def build_split_diagnostics(
    manifest: pl.DataFrame,
    seed: int,
    building_col: str = "building_id",
) -> dict[str, Any]:
    """Return JSON-serializable support and partition diagnostics for a manifest."""

    def summarize_partition(
        partition: pl.DataFrame, split: str, fold: int | None
    ) -> dict[str, Any]:
        row_count = int(partition.get_column("row_count").sum() or 0)
        positive_rows = int(partition.get_column("anomaly_count").sum() or 0)
        negative_rows = int(partition.get_column("normal_count").sum() or 0)
        return {
            "split": split,
            "fold": fold,
            "building_count": partition.height,
            "row_count": row_count,
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "anomaly_prevalence": positive_rows / row_count if row_count else None,
        }

    vault = manifest.filter(pl.col("split") == "vault")
    dev = manifest.filter(pl.col("split") == "dev")
    development_folds = [
        summarize_partition(dev.filter(pl.col("fold") == fold), "dev", fold)
        for fold in sorted(
            int(fold) for fold in dev.get_column("fold").drop_nulls().unique()
        )
    ]
    strata = (
        manifest.group_by("support_stratum")
        .agg(
            pl.len().alias("building_count"),
            pl.col("anomaly_count").min().alias("min_anomaly_rows"),
            pl.col("anomaly_count").max().alias("max_anomaly_rows"),
        )
        .sort("support_stratum")
        .to_dicts()
    )
    row_count = int(manifest.get_column("row_count").sum() or 0)
    positive_rows = int(manifest.get_column("anomaly_count").sum() or 0)

    return {
        "seed": seed,
        "stratification_method": STRATIFICATION_METHOD,
        "rationale": SPLIT_RATIONALE,
        "summary": {
            "building_count": manifest.height,
            "row_count": row_count,
            "positive_rows": positive_rows,
            "negative_rows": int(manifest.get_column("normal_count").sum() or 0),
            "anomaly_prevalence": positive_rows / row_count if row_count else None,
        },
        "class_support": {
            "vault": summarize_partition(vault, "vault", None),
            "development_folds": development_folds,
        },
        "support_strata": strata,
        "per_building_support": manifest.select(
            building_col,
            "split",
            "fold",
            "row_count",
            "anomaly_count",
            "normal_count",
            "anomaly_prevalence",
            "support_stratum",
        ).to_dicts(),
    }
