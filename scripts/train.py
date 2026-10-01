"""Run Task 6 development-only baseline OOF experiments.

Importing this module is side-effect free. All model fitting and artifact writes
are performed by :func:`main` after an explicit ``--smoke`` or ``--full`` mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import tracemalloc
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import yaml

from project.experiments.baseline_budget import load_baseline_budget, run_guarded
from project.experiments.inventory import append_inventory_records
from project.features.availability import select_predictor_columns
from project.features.registry import build_lead_baseline_registry
from project.models.artifacts import load_repeated_manifest, validate_oof_parquet
from project.models.baselines import build_baseline_estimators
from project.models.metrics import evaluate_binary_metrics, positive_class_scores

try:
    import resource as _resource
except ImportError:  # pragma: no cover - unavailable on Windows
    _resource = None


BUILDING_COLUMN = "building_id"
TIMESTAMP_COLUMN = "timestamp"
DEFAULT_TARGET_COLUMN = "anomaly"
EXPECTED_FOLD_IDS = (0, 1, 2, 3)
MAX_SMOKE_BUILDINGS = 8
APPROVED_SEED = 42
SPLIT_SEEDS = (42, 43, 44)
OOF_BATCH_SIZE = 64_000
EXPECTED_CANDIDATES = (
    "dummy_prior",
    "logistic_l2",
    "hist_gradient_boosting",
)
APPROVED_BASELINE_PARAMS: dict[str, dict[str, Any]] = {
    "dummy_prior": {"strategy": "prior"},
    "logistic_l2": {
        "C": 1.0,
        "l1_ratio": 0.0,
        "class_weight": None,
        "max_iter": 1000,
    },
    "hist_gradient_boosting": {"early_stopping": False},
}
METRIC_NAMES = ("roc_auc", "average_precision", "log_loss", "brier_score")

_CALENDAR_EXPRESSIONS: dict[str, pl.Expr] = {
    "hour": pl.col(TIMESTAMP_COLUMN).dt.hour(),
    "weekday": pl.col(TIMESTAMP_COLUMN).dt.weekday() - 1,
    "month": pl.col(TIMESTAMP_COLUMN).dt.month(),
    "year": pl.col(TIMESTAMP_COLUMN).dt.year(),
}


@dataclass(frozen=True)
class FoldData:
    """Training and validation rows for one persisted outer fold."""

    fold_id: int
    train: pl.DataFrame
    validation: pl.DataFrame
    repeat: int = 0
    split_seed: int = 42


@dataclass(frozen=True)
class PreparedRunData:
    """Development-only rows and fold assignments used by one run."""

    feature_names: tuple[str, ...]
    oof_population: pl.DataFrame
    folds: tuple[FoldData, ...]
    data_scope: str
    used_synthetic_fallback: bool
    development_assignments: pl.DataFrame | None = None


def validate_baseline_training_frame(
    frame: pl.DataFrame,
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the Task 6 predictor contract and return names in YAML order."""
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    feature_names = _configured_feature_names(config)
    if "target" not in config:
        raise ValueError("Config is missing required target key")
    target_name = config["target"]
    if target_name != DEFAULT_TARGET_COLUMN:
        raise ValueError(f"target must be {DEFAULT_TARGET_COLUMN!r}")

    required_columns = (
        BUILDING_COLUMN,
        TIMESTAMP_COLUMN,
        target_name,
        *feature_names,
    )
    missing_columns = [name for name in required_columns if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"Training frame is missing required columns: {missing_columns}")
    if frame.height == 0:
        raise ValueError("Training frame must contain at least one row")

    schema = frame.schema
    for key_column in (BUILDING_COLUMN, TIMESTAMP_COLUMN, target_name):
        if frame.get_column(key_column).null_count():
            raise ValueError(f"Required key/target column {key_column!r} contains nulls")

    timestamp_dtype = schema[TIMESTAMP_COLUMN]
    if not str(timestamp_dtype).startswith("Datetime"):
        raise TypeError(
            f"{TIMESTAMP_COLUMN} must have a Polars Datetime dtype, got {timestamp_dtype}"
        )

    target_dtype = schema[target_name]
    if not target_dtype.is_integer() or target_dtype == pl.Boolean:
        raise TypeError(f"{target_name} must use an integer binary dtype, got {target_dtype}")
    target_values = frame.get_column(target_name).unique().to_list()
    if not target_values or any(value not in (0, 1) for value in target_values):
        raise ValueError(f"{target_name} must contain only binary labels 0 and 1")

    configured = config["features"]
    numeric_features = tuple(configured["numeric"])
    categorical_features = tuple(configured["categorical"])
    for name in numeric_features:
        dtype = schema[name]
        if not dtype.is_numeric() or dtype == pl.Boolean:
            raise TypeError(f"Numeric feature {name!r} has incompatible dtype {dtype}")
    for name in categorical_features:
        dtype_name = str(schema[name])
        if dtype_name not in {"String", "Categorical"} and not dtype_name.startswith("Enum"):
            raise TypeError(
                f"Categorical feature {name!r} has incompatible dtype {schema[name]}"
            )

    registry = build_lead_baseline_registry()
    calendar_features = tuple(
        name
        for name in feature_names
        if (metadata := registry.get(name)) is not None
        and metadata.family == "calendar"
    )
    for name in calendar_features:
        if name not in _CALENDAR_EXPRESSIONS:
            raise ValueError(f"No timestamp validation rule is registered for {name!r}")
        if frame.get_column(name).null_count():
            raise ValueError(f"Calendar feature {name!r} contains null values")
        expression = _CALENDAR_EXPRESSIONS[name]
        has_mismatch = frame.select(
            (
                pl.col(name).cast(pl.Float64)
                != expression.cast(pl.Float64)
            )
            .fill_null(True)
            .any()
        ).item()
        if has_mismatch:
            raise ValueError(
                f"Calendar feature {name!r} does not match {TIMESTAMP_COLUMN}"
            )

    prediction_time = frame.get_column(TIMESTAMP_COLUMN).min()
    selected = select_predictor_columns(
        feature_names,
        registry,
        prediction_time=prediction_time,
    )
    if selected != feature_names:
        raise ValueError(
            "Configured baseline features are not all available approved predictors"
        )
    return feature_names


def _configured_feature_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    features = config.get("features")
    if not isinstance(features, Mapping) or set(features) != {"numeric", "categorical"}:
        raise ValueError("Config features must contain only numeric and categorical lists")

    numeric = _feature_sequence(features.get("numeric"), "features.numeric")
    categorical = _feature_sequence(
        features.get("categorical"), "features.categorical"
    )
    combined = (*numeric, *categorical)
    if len(combined) != len(set(combined)):
        raise ValueError("Configured feature lists must not contain duplicates")

    registry = build_lead_baseline_registry()
    expected_categorical = tuple(
        name
        for name in registry.names
        if (metadata := registry.get(name)) is not None
        and metadata.family == "categorical"
    )
    expected_numeric = tuple(
        name
        for name in registry.names
        if (metadata := registry.get(name)) is not None
        and metadata.family != "categorical"
    )
    if set(numeric) != set(expected_numeric):
        raise ValueError(
            "Config numeric feature set differs from the approved registry; "
            f"expected {sorted(expected_numeric)}, got {sorted(numeric)}"
        )
    if set(categorical) != set(expected_categorical):
        raise ValueError(
            "Config categorical feature set differs from the approved registry; "
            f"expected {sorted(expected_categorical)}, got {sorted(categorical)}"
        )
    if set(combined) != set(registry.names):
        raise ValueError("Config feature set differs from the complete approved registry")
    return tuple(combined)


def _feature_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of feature names")
    if any(not isinstance(feature, str) or not feature.strip() for feature in value):
        raise ValueError(f"{name} must contain non-empty strings")
    return tuple(value)


def _validate_execution_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    feature_names = _configured_feature_names(config)
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise TypeError("Config seed must be an integer")
    if config.get("seed") != APPROVED_SEED:
        raise ValueError(
            f"Task 6 requires frozen seed {APPROVED_SEED}, got {config.get('seed')}"
        )
    if config.get("folds") != len(EXPECTED_FOLD_IDS):
        raise ValueError(f"Config folds must be {len(EXPECTED_FOLD_IDS)}")
    if config.get("repeats") != 3 or config.get("split_seeds") != list(SPLIT_SEEDS):
        raise ValueError("Baseline requires three repeats with split seeds [42, 43, 44]")
    if config.get("target") != DEFAULT_TARGET_COLUMN:
        raise ValueError(f"Config target must be {DEFAULT_TARGET_COLUMN!r}")
    if config.get("positive_label") != 1:
        raise ValueError("Task 6 positive_label must be 1")
    if config.get("primary_metric") != "roc_auc":
        raise ValueError("Task 6 primary_metric must be 'roc_auc'")
    if tuple(config.get("guardrails", ())) != METRIC_NAMES[1:]:
        raise ValueError(f"Task 6 guardrails must be {list(METRIC_NAMES[1:])}")

    models = config.get("models")
    if not isinstance(models, Mapping) or tuple(models) != EXPECTED_CANDIDATES:
        raise ValueError(
            "Config models must list the three approved baseline candidates in order"
        )
    for candidate_id, candidate_config in models.items():
        if not isinstance(candidate_config, Mapping):
            raise TypeError(f"Model config {candidate_id!r} must be a mapping")
        for field in ("model_family", "recipe_id", "model_config_id"):
            value = candidate_config.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Model config {candidate_id!r} requires {field}")
        params = candidate_config.get("params")
        if not isinstance(params, Mapping):
            raise TypeError(f"Model config {candidate_id!r} params must be a mapping")
        expected_params = APPROVED_BASELINE_PARAMS[candidate_id]
        if dict(params) != expected_params:
            raise ValueError(
                f"Model config {candidate_id!r} params {dict(params)!r} do not match "
                f"approved frozen contract {expected_params!r}"
            )
    return feature_names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI; a run mode is mandatory to prevent accidental fitting."""
    parser = argparse.ArgumentParser(
        description="Run development-only Task 6 baseline OOF experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/baselines.yaml"),
        help="Baseline YAML path, relative to the project root unless absolute.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run one bounded dev fold.")
    mode.add_argument("--full", action="store_true", help="Run four dev folds across three repeats.")
    parser.add_argument(
        "--budget",
        type=Path,
        default=Path("configs/experiments/baseline_execution_budget.yaml"),
        help="Baseline-only approved execution budget required for --full.",
    )
    parser.add_argument(
        "--max-buildings",
        type=int,
        default=MAX_SMOKE_BUILDINGS,
        help="Maximum total whole buildings in smoke train and validation (2-8).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root; defaults to the repository containing this script.",
    )
    parsed = parser.parse_args(argv)
    if parsed.max_buildings < 2 or parsed.max_buildings > MAX_SMOKE_BUILDINGS:
        parser.error(
            f"--max-buildings must be between 2 and {MAX_SMOKE_BUILDINGS}"
        )
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    scan = pl.scan_parquet(path)
    columns = scan.collect_schema().names()
    if "repeat" in columns and "seed" in columns:
        manifest = load_repeated_manifest(path)
        return manifest, manifest.filter(pl.col("split") == "dev").select(
            BUILDING_COLUMN, "repeat", "seed", "fold"
        )
    required_columns = (BUILDING_COLUMN, "split", "fold")
    missing = [c for c in required_columns if c not in columns]
    if missing:
        raise ValueError(f"Building manifest is missing columns: {missing}")
    manifest = pl.read_parquet(path).select(required_columns)
    dev = manifest.filter(pl.col("split") == "dev").with_columns(
        pl.lit(0, dtype=pl.Int64).alias("repeat"),
        pl.lit(42, dtype=pl.Int64).alias("seed"),
    )
    return manifest, dev.select(BUILDING_COLUMN, "repeat", "seed", "fold")


def _validate_prepared_building_coverage(
    train_path: Path,
    manifest: pl.DataFrame,
) -> None:
    scan = pl.scan_parquet(train_path)
    if BUILDING_COLUMN not in scan.collect_schema():
        raise ValueError(f"Prepared training data is missing {BUILDING_COLUMN!r}")
    prepared_ids = (
        scan.select(BUILDING_COLUMN).unique().collect().get_column(BUILDING_COLUMN)
    )
    if prepared_ids.null_count():
        raise ValueError("Prepared training data contains null building IDs")
    if set(prepared_ids.to_list()) != set(manifest.get_column(BUILDING_COLUMN).to_list()):
        raise ValueError(
            "Prepared training building IDs do not exactly match the frozen manifest"
        )


def _collect_development_support(
    train_path: Path,
    development_assignments: pl.DataFrame,
    target_name: str,
) -> pl.DataFrame:
    scan = pl.scan_parquet(train_path)
    schema = scan.collect_schema()
    if target_name not in schema:
        raise ValueError(f"Prepared training data is missing target {target_name!r}")
    if not schema[target_name].is_integer() or schema[target_name] == pl.Boolean:
        raise TypeError(f"Prepared target {target_name!r} must have an integer dtype")

    development_ids = development_assignments.get_column(BUILDING_COLUMN).to_list()
    support = (
        scan.filter(pl.col(BUILDING_COLUMN).is_in(development_ids))
        .select(BUILDING_COLUMN, target_name)
        .group_by(BUILDING_COLUMN)
        .agg(
            pl.col(target_name).null_count().alias("__null_targets"),
            (
                pl.col(target_name).is_not_null()
                & ~pl.col(target_name).is_in([0, 1])
            )
            .sum()
            .alias("__invalid_targets"),
            (pl.col(target_name) == 0).sum().alias("__negative_count"),
            (pl.col(target_name) == 1).sum().alias("__positive_count"),
        )
        .collect(engine="streaming")
    )
    if support.height != len(development_ids):
        raise ValueError("Prepared development rows do not cover all dev manifest buildings")
    if support.get_column("__null_targets").sum() or support.get_column(
        "__invalid_targets"
    ).sum():
        raise ValueError("Prepared development target contains null or non-binary labels")

    support = support.join(development_assignments, on=BUILDING_COLUMN, how="left")
    if support.get_column("fold").null_count():
        raise ValueError("Development target support does not align with fold assignments")
    return support


def _choose_groups_with_both_classes(
    support: pl.DataFrame,
    candidate_folds: Sequence[int],
    maximum_groups: int,
) -> tuple[Any, ...] | None:
    candidates = (
        support.filter(pl.col("fold").is_in(list(candidate_folds)))
        .sort(BUILDING_COLUMN)
        .to_dicts()
    )
    if maximum_groups < 1 or len(candidates) < 1:
        return None

    selected: list[dict[str, Any]] = []
    covered: set[int] = set()
    while len(selected) < maximum_groups and covered != {0, 1}:
        best: dict[str, Any] | None = None
        best_gain = 0
        for candidate in candidates:
            if candidate in selected:
                continue
            classes = set()
            if candidate["__negative_count"]:
                classes.add(0)
            if candidate["__positive_count"]:
                classes.add(1)
            gain = len(classes - covered)
            if gain > best_gain:
                best = candidate
                best_gain = gain
        if best is None:
            break
        selected.append(best)
        if best["__negative_count"]:
            covered.add(0)
        if best["__positive_count"]:
            covered.add(1)

    if covered != {0, 1}:
        return None
    for candidate in candidates:
        if len(selected) >= maximum_groups:
            break
        if candidate not in selected:
            selected.append(candidate)
    return tuple(row[BUILDING_COLUMN] for row in selected)


def _select_smoke_groups(
    support: pl.DataFrame,
    max_buildings: int,
) -> tuple[int, tuple[Any, ...], tuple[Any, ...]] | None:
    validation_limit = max_buildings // 2
    for fold_id in EXPECTED_FOLD_IDS:
        validation_groups = _choose_groups_with_both_classes(
            support, (fold_id,), validation_limit
        )
        if validation_groups is None:
            continue
        training_folds = tuple(value for value in EXPECTED_FOLD_IDS if value != fold_id)
        training_limit = max_buildings - len(validation_groups)
        training_groups = _choose_groups_with_both_classes(
            support, training_folds, training_limit
        )
        if training_groups is None:
            continue
        if set(training_groups).intersection(validation_groups):
            raise ValueError("Smoke train and validation buildings overlap")
        return fold_id, training_groups, validation_groups
    return None


def _load_rows_for_buildings(
    train_path: Path,
    building_ids: Sequence[Any],
    development_assignments: pl.DataFrame,
    feature_names: Sequence[str],
    target_name: str,
) -> pl.DataFrame:
    scan = pl.scan_parquet(train_path)
    columns = (BUILDING_COLUMN, TIMESTAMP_COLUMN, target_name, *feature_names)
    schema = scan.collect_schema()
    missing = [column for column in columns if column not in schema]
    if missing:
        raise ValueError(f"Prepared training data is missing required columns: {missing}")
    if not building_ids:
        raise ValueError("At least one development building must be selected")

    rows = (
        scan.filter(pl.col(BUILDING_COLUMN).is_in(list(building_ids)))
        .select(columns)
        .collect(engine="streaming")
    )
    if rows.height == 0:
        raise ValueError("No prepared development rows matched selected buildings")
    if set(rows.get_column(BUILDING_COLUMN).unique().to_list()) != set(building_ids):
        raise ValueError("Prepared rows do not cover every selected development building")
    rows = rows.join(development_assignments, on=BUILDING_COLUMN, how="left")
    if rows.get_column("fold").null_count():
        raise ValueError("Prepared rows include a building without a dev fold assignment")
    _validate_natural_keys(rows)
    return rows.sort([BUILDING_COLUMN, TIMESTAMP_COLUMN])


def _validate_natural_keys(frame: pl.DataFrame) -> None:
    if frame.get_column(TIMESTAMP_COLUMN).null_count():
        raise ValueError("Prepared development rows contain null timestamps")
    if frame.select(BUILDING_COLUMN, TIMESTAMP_COLUMN).is_duplicated().any():
        raise ValueError("Prepared development natural keys are not unique")


def _with_canonical_row_index(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.sort([BUILDING_COLUMN, TIMESTAMP_COLUMN])
        .with_row_index("row_index")
        .with_columns(pl.col("row_index").cast(pl.Int64))
    )


def _require_both_classes(frame: pl.DataFrame, side: str, target_name: str) -> None:
    labels = set(frame.get_column(target_name).unique().to_list())
    if labels != {0, 1}:
        raise ValueError(f"{side} must contain both target classes 0 and 1; found {labels}")


def _prepare_run_data(
    mode: str,
    max_buildings: int,
    train_path: Path,
    split_path: Path,
    config: Mapping[str, Any],
    feature_names: tuple[str, ...],
) -> PreparedRunData:
    manifest, development_assignments = _load_manifest(split_path)
    _validate_prepared_building_coverage(train_path, manifest)
    target_name = str(config["target"])
    first_assignments = development_assignments.filter(pl.col("repeat") == 0).select(
        BUILDING_COLUMN, "fold"
    )

    if mode == "full":
        rows = _load_rows_for_buildings(
            train_path,
            first_assignments.get_column(BUILDING_COLUMN).to_list(),
            first_assignments,
            feature_names,
            target_name,
        )
        validate_baseline_training_frame(rows, config)
        rows = _with_canonical_row_index(rows)
        return PreparedRunData(
            feature_names,
            rows,
            (),
            "all development buildings across four persisted folds and three repeats",
            False,
            development_assignments,
        )

    support = _collect_development_support(
        train_path, first_assignments, target_name
    )
    selection = _select_smoke_groups(support, max_buildings)
    if selection is None:
        synthetic_train, synthetic_validation, fold_id = _synthetic_smoke_frames(
            feature_names, target_name, EXPECTED_FOLD_IDS[0]
        )
        validate_baseline_training_frame(synthetic_train, config)
        validate_baseline_training_frame(synthetic_validation, config)
        _validate_outer_fold(
            synthetic_train, synthetic_validation, fold_id, target_name
        )
        population = _with_canonical_row_index(synthetic_validation)
        return PreparedRunData(
            feature_names,
            population,
            (FoldData(fold_id, synthetic_train, population),),
            "synthetic fallback; no real-data performance claim",
            True,
        )

    fold_id, training_groups, validation_groups = selection
    selected_groups = (*training_groups, *validation_groups)
    rows = _load_rows_for_buildings(
        train_path,
        selected_groups,
        first_assignments,
        feature_names,
        target_name,
    )
    validate_baseline_training_frame(rows, config)
    train = rows.filter(pl.col(BUILDING_COLUMN).is_in(training_groups))
    validation = rows.filter(pl.col(BUILDING_COLUMN).is_in(validation_groups))
    _validate_outer_fold(train, validation, fold_id, target_name)
    population = _with_canonical_row_index(validation)
    return PreparedRunData(
        feature_names,
        population,
        (FoldData(fold_id, train, population),),
        f"smoke subset of at most {max_buildings} dev buildings; fold {fold_id}",
        False,
    )


def _validate_outer_fold(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    fold_id: int,
    target_name: str,
) -> None:
    if train.height == 0 or validation.height == 0:
        raise ValueError(f"Fold {fold_id} must have non-empty train and validation rows")
    train_groups = set(train.get_column(BUILDING_COLUMN).unique().to_list())
    validation_groups = set(
        validation.get_column(BUILDING_COLUMN).unique().to_list()
    )
    if train_groups.intersection(validation_groups):
        raise ValueError(f"Fold {fold_id} splits a building across train and validation")
    if validation.get_column("fold").unique().to_list() != [fold_id]:
        raise ValueError(f"Fold {fold_id} validation rows do not match the frozen manifest")
    if any(value == fold_id for value in train.get_column("fold").unique().to_list()):
        raise ValueError(f"Fold {fold_id} validation building appears in outer training")
    _require_both_classes(train, f"Fold {fold_id} outer train", target_name)
    _require_both_classes(validation, f"Fold {fold_id} validation", target_name)


def _synthetic_smoke_frames(
    feature_names: Sequence[str],
    target_name: str,
    fold_id: int,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    del feature_names  # The fixture uses the fixed approved schema below.
    # Project timestamps are timezone-naive to match prepared data.
    base_time = datetime(2022, 1, 3, 8)  # noqa: DTZ001
    rows: list[dict[str, Any]] = []
    for split_name, building_prefix, fold_value, base_meter in (
        ("train", "synthetic_train", (fold_id + 1) % len(EXPECTED_FOLD_IDS), 10.0),
        ("validation", "synthetic_validation", fold_id, 20.0),
    ):
        for building_number in range(1):
            building_id = f"{building_prefix}_{building_number}"
            for label in (0, 1):
                timestamp = base_time.replace(hour=8 + label)
                rows.append(
                    {
                        BUILDING_COLUMN: building_id,
                        TIMESTAMP_COLUMN: timestamp,
                        target_name: label,
                        "meter_reading": base_meter + label,
                        "primary_use": "Education" if label == 0 else "Office",
                        "square_feet": 50000.0 + label,
                        "year_built": 1980 + label,
                        "floor_count": 4 + label,
                        "hour": timestamp.hour,
                        "weekday": timestamp.weekday(),
                        "month": timestamp.month,
                        "year": timestamp.year,
                        "fold": fold_value,
                        "__synthetic_split": split_name,
                    }
                )
    fixture = pl.DataFrame(rows)
    train = fixture.filter(pl.col("__synthetic_split") == "train").drop(
        "__synthetic_split"
    )
    validation = fixture.filter(pl.col("__synthetic_split") == "validation").drop(
        "__synthetic_split"
    )
    return train, validation, fold_id


def _expected_rows(
    population: pl.DataFrame,
    target_name: str,
    run_id: str,
    hashes: Mapping[str, str],
) -> pl.DataFrame:
    return population.select(
        "row_index",
        BUILDING_COLUMN,
        TIMESTAMP_COLUMN,
        "fold",
        pl.col(target_name).alias("target"),
    ).with_columns(
        pl.lit(hashes["split_sha256"]).alias("split_sha256"),
        pl.lit(hashes["data_sha256"]).alias("data_sha256"),
        pl.lit(hashes["config_sha256"]).alias("config_sha256"),
        pl.lit(run_id).alias("run_id"),
    )


def _iter_folds(run_data: PreparedRunData, target_name: str) -> Iterator[FoldData]:
    """Materialize only the current fold pair, never twelve pairs."""
    if run_data.development_assignments is None:
        yield from run_data.folds
        return
    for repeat, seed in enumerate(SPLIT_SEEDS):
        assignments = run_data.development_assignments.filter(pl.col("repeat") == repeat)
        for fold_id in EXPECTED_FOLD_IDS:
            validation_ids = assignments.filter(pl.col("fold") == fold_id)[BUILDING_COLUMN]
            validation = run_data.oof_population.filter(pl.col(BUILDING_COLUMN).is_in(validation_ids.implode())).with_columns(pl.lit(fold_id).cast(pl.Int64).alias("fold"))
            training = run_data.oof_population.filter(~pl.col(BUILDING_COLUMN).is_in(validation_ids.implode())).drop("fold").join(assignments.select(BUILDING_COLUMN, "fold"), on=BUILDING_COLUMN)
            _validate_outer_fold(training, validation, fold_id, target_name)
            yield FoldData(fold_id, training, validation, repeat, seed)
            del training, validation


def _expected_rows_by_repeat(run_data, target_name, run_id, hashes):
    if run_data.development_assignments is None:
        yield 0, _expected_rows(run_data.oof_population, target_name, run_id, hashes).with_columns(
            pl.lit(42).alias("split_seed"))
        return
    for repeat, seed in enumerate(SPLIT_SEEDS):
        assignments = run_data.development_assignments.filter(pl.col("repeat") == repeat)
        population = run_data.oof_population.select("row_index", BUILDING_COLUMN, TIMESTAMP_COLUMN,
            target_name).join(assignments.select(BUILDING_COLUMN, "fold"), on=BUILDING_COLUMN)
        yield repeat, _expected_rows(population, target_name, run_id, hashes).with_columns(
            pl.lit(seed).alias("split_seed"))
        del population


def _write_prediction_blocks(
    path: Path,
    blocks: Iterable[tuple[pl.DataFrame, np.ndarray, Mapping[str, Any]]],
    *,
    target_name: str,
    batch_size: int = OOF_BATCH_SIZE,
) -> None:
    """Write lossless column-wise blocks through one open canonical writer."""
    partial = path.with_suffix(path.suffix + ".partial")
    writer = None
    try:
        for keys, predictions, metadata in blocks:
            if keys.height != len(predictions):
                raise ValueError("Prediction/key alignment mismatch")
            for offset in range(0, keys.height, batch_size):
                frame = keys.slice(offset, batch_size).rename({target_name: "target"}).with_columns(
                    pl.Series("prediction", predictions[offset:offset + batch_size], dtype=pl.Float64),
                    *(pl.lit(value).alias(name) for name, value in metadata.items()),
                ).with_columns(pl.col("repeat", "fold").cast(pl.Int64), pl.col("target").cast(pl.Int64))
                table = frame.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=batch_size)
                del table, frame
        if writer is None:
            raise ValueError("No development OOF predictions were generated")
    finally:
        if writer is not None:
            writer.close()
    partial.replace(path)


def _run_candidate_fits(
    config: Mapping[str, Any],
    run_data: PreparedRunData,
    hashes: Mapping[str, str],
    run_id: str,
    mode: str,
    oof_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    target_name = str(config["target"])
    candidate_configs = config["models"]
    fold_metrics: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in candidate_configs
    }
    candidate_wall_times = {candidate_id: 0.0 for candidate_id in candidate_configs}

    def prediction_blocks():
        for fold in _iter_folds(run_data, target_name):
            x_train = fold.train.select(run_data.feature_names).to_pandas()
            y_train = fold.train.get_column(target_name).to_numpy()
            x_validation = fold.validation.select(run_data.feature_names).to_pandas()
            y_validation = fold.validation.get_column(target_name).to_numpy()
            for candidate_id, candidate_config in candidate_configs.items():
                estimator = build_baseline_estimators(config)[candidate_id]
                fit_started = time.perf_counter()
                estimator.fit(x_train, y_train)
                predictions = positive_class_scores(estimator, x_validation, positive_label=1)
                metrics = evaluate_binary_metrics(y_validation, predictions)
                candidate_wall_times[candidate_id] += time.perf_counter() - fit_started
                fold_metrics[candidate_id].append({
                    "repeat": fold.repeat, "split_seed": fold.split_seed, "fold": fold.fold_id,
                    "train_building_count": fold.train[BUILDING_COLUMN].n_unique(),
                    "validation_building_count": fold.validation[BUILDING_COLUMN].n_unique(),
                    "train_row_count": fold.train.height, "validation_row_count": fold.validation.height,
                    **metrics,
                })
                keys = fold.validation.select("row_index", BUILDING_COLUMN, TIMESTAMP_COLUMN,
                                              "fold", target_name)
                yield keys, predictions, {
                    "candidate_id": candidate_id,
                    **{name: candidate_config[name] for name in ("recipe_id", "model_family", "model_config_id")},
                    "repeat": fold.repeat, "split_seed": fold.split_seed,
                    **hashes, "run_id": run_id, "execution_mode": mode, "non_comparative": mode == "smoke",
                }
                del estimator, predictions, keys
            del x_train, y_train, x_validation, y_validation, fold
    _write_prediction_blocks(oof_path, prediction_blocks(), target_name=target_name)
    return fold_metrics, candidate_wall_times


def _build_metrics_payload(
    validated: dict,
    fold_metrics: Mapping[str, list[dict[str, Any]]],
    candidate_wall_times: Mapping[str, float],
    run_data: PreparedRunData,
    config: Mapping[str, Any],
    mode: str,
    run_id: str,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for candidate_id in config["models"]:
        candidates[candidate_id] = {
            **validated["candidate_metrics"][candidate_id],
            "folds": fold_metrics[candidate_id],
            "wall_time_seconds": candidate_wall_times[candidate_id],
        }

    non_comparative = mode == "smoke"
    return {
        "run_id": run_id,
        "mode": mode,
        "non_comparative": non_comparative,
        "data_scope": run_data.data_scope,
        "used_synthetic_fallback": run_data.used_synthetic_fallback,
        "target": config["target"],
        "positive_label": config["positive_label"],
        "primary_metric": config["primary_metric"],
        "guardrails": list(config["guardrails"]),
        "fold_ids": sorted({row["fold"] for row in validated["fold_metrics"]}),
        **{name: validated[name] for name in ("repeat_count", "repeat_ids", "split_seeds",
           "development_oof_row_count", "oof_record_count", "repeat_metrics", "fold_metrics")},
        "metric_aggregation": "equal_mean_of_repeat_pooled_metrics",
        "candidate_metrics": candidates,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _model_card(
    run_id: str,
    mode: str,
    run_data: PreparedRunData,
    feature_names: Sequence[str],
    hashes: Mapping[str, str],
    config: Mapping[str, Any],
) -> str:
    status = "non-comparative smoke evidence" if mode == "smoke" else "full dev OOF"
    candidate_lines = [
        f"- `{candidate_id}` (`{candidate_config['model_family']}`): "
        f"`{candidate_config['recipe_id']}`, `{candidate_config['model_config_id']}`"
        for candidate_id, candidate_config in config["models"].items()
    ]
    synthetic_note = (
        "\nThe smoke fit used a deterministic synthetic fixture because no eligible "
        "real subset with both classes on both sides was available. No real-data "
        "performance claim is made.\n"
        if run_data.used_synthetic_fallback
        else "\n"
    )
    return (
        f"# Baseline OOF Model Card — {run_id}\n\n"
        f"- **Run status:** {status}\n"
        f"- **Mode:** `{mode}`\n"
        f"- **Prediction task:** point anomaly classification at observed hourly rows\n"
        f"- **Target:** `{config['target']}`; positive class `{config['positive_label']}`\n"
        f"- **Primary metric:** `{config['primary_metric']}`\n"
        f"- **Guardrails:** {', '.join(f'`{item}`' for item in config['guardrails'])}\n"
        f"- **Data scope:** {run_data.data_scope}\n"
        "- **Fit/score boundary:** `split=dev` only; persisted building folds; "
        "vault labels and test data are excluded.\n"
        f"- **Predictors (config order):** {', '.join(f'`{name}`' for name in feature_names)}\n"
        "- **Preprocessing:** fold-local pipelines from the baseline estimator factory.\n"
        f"- **Candidates:**\n{chr(10).join(candidate_lines)}\n"
        f"- **Data SHA-256:** `{hashes['data_sha256']}`\n"
        f"- **Split SHA-256:** `{hashes['split_sha256']}`\n"
        f"- **Config SHA-256:** `{hashes['config_sha256']}`\n"
        f"{synthetic_note}"
        "## Interpretation\n\n"
        "Metrics are recorded in `metrics.json`. Smoke results are workflow-continuity "
        "evidence only and must not be used for candidate ranking or performance claims. "
        "Readers must require `run_status.json` status `completed` before consuming a run.\n"
    )


def _safe_output_root(project_root: Path) -> Path:
    output_root = project_root / "outputs"
    resolved_root = output_root.resolve()
    if not resolved_root.is_relative_to(project_root.resolve()):
        raise ValueError("outputs/ resolves outside the project root")
    return resolved_root


def _ensure_within_outputs(path: Path, output_root: Path) -> Path:
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(output_root.resolve()):
        raise ValueError(f"Output path {path} resolves outside outputs/")
    return resolved_path


def _create_run_directory(output_root: Path) -> tuple[str, Path]:
    runs_dir = output_root / "runs"
    _ensure_within_outputs(runs_dir, output_root)
    runs_dir.mkdir(parents=True, exist_ok=True)
    _ensure_within_outputs(runs_dir, output_root)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    while True:
        run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        run_dir = runs_dir / run_id
        _ensure_within_outputs(run_dir, output_root)
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        _ensure_within_outputs(run_dir, output_root)
        return run_id, run_dir


def _process_peak_rss_mb() -> float | None:
    if _resource is None:
        return None
    peak = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak_bytes = peak
    else:
        peak_bytes = peak * 1024
    return round(peak_bytes / (1024 * 1024), 3)


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _run(args: argparse.Namespace) -> int:
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    config_path = _resolve_path(args.config, project_root)
    if not config_path.is_file():
        raise FileNotFoundError(f"Baseline config not found: {config_path}")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, Mapping):
        raise TypeError("Baseline YAML must contain a mapping")
    feature_names = _validate_execution_config(config)

    output_root = _safe_output_root(project_root)
    train_path = output_root / "prepared" / "train_joined.parquet"
    split_path = _resolve_path(config["split_manifest"], project_root)
    _ensure_within_outputs(train_path, output_root)
    _ensure_within_outputs(split_path, output_root)
    if not train_path.is_file():
        raise FileNotFoundError(f"Prepared training Parquet not found: {train_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Frozen building manifest not found: {split_path}")

    hashes = {
        "data_sha256": _sha256_file(train_path),
        "split_sha256": _sha256_file(split_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    mode = "smoke" if args.smoke else "full"
    max_buildings = args.max_buildings if args.smoke else MAX_SMOKE_BUILDINGS

    started = time.perf_counter()
    tracemalloc.start()
    try:
        run_data = _prepare_run_data(
            mode,
            max_buildings,
            train_path,
            split_path,
            config,
            feature_names,
        )
        guarded_run_dir = getattr(args, "guarded_run_dir", None)
        if guarded_run_dir is None:
            run_id, run_dir = _create_run_directory(output_root)
        else:
            run_dir = _ensure_within_outputs(Path(guarded_run_dir), output_root)
            run_id = run_dir.name
        _write_json(
            run_dir / "run_status.json",
            {"run_id": run_id, "status": "in_progress"},
        )
        (run_dir / "baselines.yaml").write_bytes(config_bytes)

        oof_path = run_dir / "oof_predictions.parquet"
        fold_metrics, candidate_wall_times = _run_candidate_fits(
            config, run_data, hashes, run_id, mode, oof_path
        )
        validated = validate_oof_parquet(
            oof_path, _expected_rows_by_repeat(run_data, str(config["target"]), run_id, hashes),
            expected_candidate_ids=tuple(config["models"]), execution_mode=mode,
            candidate_metadata=config["models"],
        )
        if args.full and (validated["repeat_ids"] != [0, 1, 2] or len(validated["fold_metrics"]) != 36):
            raise ValueError("Full OOF must cover all twelve repeat/fold arms for three candidates")

        metrics_payload = _build_metrics_payload(
            validated,
            fold_metrics,
            candidate_wall_times,
            run_data,
            config,
            mode,
            run_id,
        )
        metrics_path = run_dir / "metrics.json"
        _write_json(metrics_path, metrics_payload)
    finally:
        _current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    elapsed = time.perf_counter() - started
    resource_profile = {
        "run_id": run_id,
        "mode": mode,
        "non_comparative": mode == "smoke",
        "elapsed_seconds": round(elapsed, 3),
        "python_traced_peak_memory_mb": round(peak_memory / (1024 * 1024), 3),
        "process_peak_rss_mb": _process_peak_rss_mb(),
        "candidate_fit_wall_time_seconds": candidate_wall_times,
        "model_fit_count": sum(len(rows) for rows in fold_metrics.values()),
        "fold_ids": metrics_payload["fold_ids"],
        "repeat_ids": validated["repeat_ids"],
        "repeat_count": validated["repeat_count"],
        "split_seeds": validated["split_seeds"],
        "memory_note": (
            "Python-traced memory excludes some native-library allocations; process peak RSS "
            "is included when the platform exposes resource.getrusage."
        ),
    }
    _write_json(run_dir / "resource_profile.json", resource_profile)
    (run_dir / "model_card.md").write_text(
        _model_card(
            run_id, mode, run_data, feature_names, hashes, config
        ),
        encoding="utf-8",
    )
    for candidate_id, candidate_config in config["models"].items():
        candidate_card_config = {**config, "models": {candidate_id: candidate_config}}
        (run_dir / f"model_card_{candidate_id}.md").write_text(
            _model_card(run_id, mode, run_data, feature_names, hashes, candidate_card_config),
            encoding="utf-8",
        )

    oof_relative_path = oof_path.relative_to(project_root).as_posix()
    inventory_records = []
    for candidate_id, candidate_config in config["models"].items():
        inventory_records.append(
            {
                "experiment_id": "task6_baseline_oof",
                "hypothesis_id": "approved_baseline_portfolio",
                "varied_factor_id": "baseline_model_family",
                "baseline_id": "dummy_prior",
                "candidate_id": candidate_id,
                "recipe_id": candidate_config["recipe_id"],
                "model_config_id": candidate_config["model_config_id"],
                "data_sha256": hashes["data_sha256"],
                "split_sha256": hashes["split_sha256"],
                "config_sha256": hashes["config_sha256"],
                "primary_metric": config["primary_metric"],
                "oof_path": oof_relative_path,
                "wall_time_seconds": candidate_wall_times[candidate_id],
                "outcome": "non_comparative" if mode == "smoke" else "completed",
                "rejection_reason": (
                    "Smoke metrics are workflow-continuity evidence only."
                    if mode == "smoke"
                    else None
                ),
                "run_id": run_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "execution_mode": mode,
                "model_family": candidate_config["model_family"],
                "non_comparative": mode == "smoke",
            }
        )

    inventory_path = output_root / "runs" / "experiment_inventory.jsonl"
    _ensure_within_outputs(inventory_path, output_root)
    append_inventory_records(inventory_path, inventory_records)
    if guarded_run_dir is not None:
        _write_json(run_dir / "run_status.json", {"run_id": run_id, "status": "awaiting_guard"})
        return 0
    _write_json(run_dir / "run_status.json", {"run_id": run_id, "status": "completed"})
    print(
        f"Completed {mode} run {run_id}: {run_dir}; "
        f"OOF rows={metrics_payload['development_oof_row_count']}, "
        f"non_comparative={metrics_payload['non_comparative']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a config-driven baseline OOF experiment after explicit mode selection."""
    args = parse_args(argv)
    try:
        root = args.project_root.resolve() if args.project_root else Path(__file__).resolve().parents[1]
        config = yaml.safe_load(_resolve_path(args.config, root).read_bytes())
        _validate_execution_config(config)
        budget = load_baseline_budget(_resolve_path(args.budget, root), config)
        run_id, run_dir = _create_run_directory(_safe_output_root(root))
        args.guarded_run_dir = run_dir
        _write_json(run_dir / "run_status.json", {"run_id": run_id, "status": "in_progress"})
        _write_json(run_dir / "baseline_execution_budget.json", budget)
        exit_code = run_guarded(_run, args, run_dir, budget)
        if exit_code:
            return exit_code
        if json.loads((run_dir / "run_status.json").read_text())["status"] != "awaiting_guard":
            raise ValueError("Worker did not finish the baseline artifact set")
        _write_json(run_dir / "run_status.json", {"run_id": run_id, "status": "completed"})
        print(f"Completed guarded {'smoke' if args.smoke else 'full'} run {run_id}: {run_dir}")
        return 0
    except Exception as error:  # noqa: BLE001
        print(f"Training run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
