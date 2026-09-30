"""Validation for pooled out-of-fold prediction artifacts."""

from __future__ import annotations

import numpy as np
import polars as pl

from project.models.metrics import evaluate_binary_metrics

_OOF_REQUIRED_COLUMNS = (
    "candidate_id",
    "recipe_id",
    "model_family",
    "model_config_id",
    "row_index",
    "repeat",
    "fold",
    "building_id",
    "timestamp",
    "prediction",
    "target",
    "split_sha256",
    "data_sha256",
    "config_sha256",
    "run_id",
)

_EXPECTED_REQUIRED_COLUMNS = (
    "row_index",
    "building_id",
    "timestamp",
    "fold",
    "target",
    "split_sha256",
    "data_sha256",
    "config_sha256",
    "run_id",
)

_TASK_KEY_COLUMNS = (
    "candidate_id",
    "recipe_id",
    "model_config_id",
    "row_index",
    "repeat",
    "fold",
)

_ROW_ALIGNMENT_COLUMNS = (
    "building_id",
    "timestamp",
    "fold",
    "target",
    "split_sha256",
    "data_sha256",
    "config_sha256",
    "run_id",
)

_RUN_LEVEL_COLUMNS = ("split_sha256", "data_sha256", "config_sha256", "run_id")
_CANDIDATE_METADATA_COLUMNS = ("recipe_id", "model_family", "model_config_id")
_METRIC_NAMES = ("roc_auc", "average_precision", "log_loss", "brier_score")


def validate_oof_artifact(
    oof: pl.DataFrame,
    expected_rows: pl.DataFrame,
) -> dict[str, dict[str, float]]:
    """Validate development-only OOF predictions and return pooled metrics.

    Each candidate must have one repeat-0 prediction for every row in the
    expected development manifest. Natural keys and run metadata must agree
    with that manifest before pooled binary metrics are calculated.
    """
    if not isinstance(oof, pl.DataFrame):
        raise TypeError("oof must be a Polars DataFrame")
    if not isinstance(expected_rows, pl.DataFrame):
        raise TypeError("expected_rows must be a Polars DataFrame")

    _require_columns(oof, _OOF_REQUIRED_COLUMNS, "OOF artifact")
    _require_columns(expected_rows, _EXPECTED_REQUIRED_COLUMNS, "expected rows")
    _require_non_null_columns(oof, _OOF_REQUIRED_COLUMNS, "OOF artifact")
    _require_non_null_columns(
        expected_rows, _EXPECTED_REQUIRED_COLUMNS, "expected rows"
    )

    if oof.height == 0:
        raise ValueError("OOF artifact must contain at least one prediction")
    if expected_rows.height == 0:
        raise ValueError("expected development rows must not be empty")

    if _has_duplicate_rows(expected_rows, ("row_index",)):
        raise ValueError("expected rows contain duplicate row_index values")
    if _has_duplicate_rows(expected_rows, ("building_id", "timestamp")):
        raise ValueError("expected rows contain duplicate natural keys")
    if _has_duplicate_rows(oof, _TASK_KEY_COLUMNS):
        raise ValueError("OOF artifact contains a duplicate task key")
    if _has_duplicate_rows(oof, ("candidate_id", "row_index")):
        raise ValueError("OOF artifact contains duplicate candidate row coverage")

    _validate_canonical_row_indices(expected_rows)
    _validate_repeat_zero(oof)
    _validate_run_metadata(oof, expected_rows)

    prediction_series = oof.get_column("prediction")
    try:
        predictions = np.asarray(prediction_series.to_numpy(), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("prediction scores must be numeric and finite") from error
    if not np.isfinite(predictions).all():
        raise ValueError("prediction scores must be finite")
    if ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("prediction scores must be within [0, 1]")

    candidate_ids = oof.get_column("candidate_id").unique().to_list()
    if any(
        not isinstance(candidate_id, str) or not candidate_id.strip()
        for candidate_id in candidate_ids
    ):
        raise ValueError("candidate_id values must be non-empty strings")

    _validate_row_alignment(oof, expected_rows)

    results: dict[str, dict[str, float]] = {}
    for candidate_id in candidate_ids:
        candidate_oof = oof.filter(pl.col("candidate_id") == candidate_id)
        if candidate_oof.height != expected_rows.height:
            raise ValueError(
                f"candidate {candidate_id!r} has missing or incomplete development-row coverage"
            )

        for column in _CANDIDATE_METADATA_COLUMNS:
            _require_single_text_value(candidate_oof, column, f"candidate {candidate_id!r}")

        try:
            metrics = evaluate_binary_metrics(
                candidate_oof.get_column("target").to_numpy(),
                candidate_oof.get_column("prediction").cast(pl.Float64).to_numpy(),
            )
        except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise ValueError(
                f"candidate {candidate_id!r} has invalid OOF labels or predictions: {error}"
            ) from error

        results[candidate_id] = {
            name: float(metrics[name]) for name in _METRIC_NAMES
        }

    return results


def _require_columns(
    frame: pl.DataFrame,
    required_columns: tuple[str, ...],
    frame_name: str,
) -> None:
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{frame_name} is missing required columns: {missing_columns}")


def _require_non_null_columns(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    frame_name: str,
) -> None:
    null_columns = [column for column in columns if frame.get_column(column).null_count()]
    if null_columns:
        if "prediction" in null_columns:
            raise ValueError("OOF prediction scores must be finite and non-null")
        raise ValueError(f"{frame_name} contains null values in columns: {null_columns}")


def _has_duplicate_rows(frame: pl.DataFrame, columns: tuple[str, ...]) -> bool:
    try:
        return bool(frame.select(columns).is_duplicated().any())
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise ValueError(f"columns {columns} cannot be checked for uniqueness") from error


def _validate_canonical_row_indices(expected_rows: pl.DataFrame) -> None:
    try:
        ordered_indices = (
            expected_rows.sort(["building_id", "timestamp"])
            .get_column("row_index")
            .to_numpy()
        )
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise ValueError(
            "expected natural keys cannot be sorted for row_index validation"
        ) from error

    if not np.array_equal(ordered_indices, np.arange(expected_rows.height)):
        raise ValueError(
            "expected row_index must be zero-based and follow (building_id, timestamp) order"
        )


def _validate_repeat_zero(oof: pl.DataFrame) -> None:
    repeats = oof.get_column("repeat").unique().to_list()
    if repeats != [0]:
        raise ValueError("OOF artifact must contain exactly one repeat with repeat=0")


def _require_single_text_value(
    frame: pl.DataFrame,
    column: str,
    frame_name: str,
) -> str:
    values = frame.get_column(column).unique().to_list()
    if len(values) != 1 or not isinstance(values[0], str) or not values[0].strip():
        raise ValueError(f"{frame_name} must have one non-empty string value for {column}")
    return values[0]


def _validate_run_metadata(oof: pl.DataFrame, expected_rows: pl.DataFrame) -> None:
    for column in _RUN_LEVEL_COLUMNS:
        expected_value = _require_single_text_value(expected_rows, column, "expected rows")
        oof_value = _require_single_text_value(oof, column, "OOF artifact")
        if oof_value != expected_value:
            raise ValueError(f"OOF {column} mismatch with expected rows")


def _validate_row_alignment(oof: pl.DataFrame, expected_rows: pl.DataFrame) -> None:
    expected_lookup = expected_rows.select(
        "row_index", *_ROW_ALIGNMENT_COLUMNS
    ).rename({column: f"__expected_{column}" for column in _ROW_ALIGNMENT_COLUMNS})
    oof_keys = oof.select("row_index", *_ROW_ALIGNMENT_COLUMNS)

    try:
        aligned = oof_keys.join(expected_lookup, on="row_index", how="left")
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise ValueError("OOF row_index values cannot be aligned with expected rows") from error

    try:
        if aligned.get_column("__expected_building_id").null_count():
            raise ValueError("OOF artifact contains unexpected, extra, or vault rows")

        mismatch_expressions = [
            pl.col(column) != pl.col(f"__expected_{column}")
            for column in _ROW_ALIGNMENT_COLUMNS
        ]
        has_mismatch = aligned.select(
            pl.any_horizontal(*mismatch_expressions).any()
        ).item()
    except (TypeError, pl.exceptions.PolarsError) as error:
        raise ValueError("OOF row metadata cannot be compared with expected rows") from error

    if has_mismatch:
        raise ValueError(
            "OOF natural key, fold, target, hash, or run_id mismatch with expected rows"
        )
