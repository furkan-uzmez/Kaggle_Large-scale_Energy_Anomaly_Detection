"""Validation for pooled out-of-fold prediction artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.dataset as ds
import pyarrow.parquet as pq

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
    *,
    expected_repeat: int = 0,
    expected_candidate_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Validate development-only OOF predictions and return pooled metrics.

    Each candidate must have one prediction in the specified repeat for every row in the
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
    if type(expected_repeat) is not int or oof["repeat"].unique().to_list() != [expected_repeat]:
        raise ValueError(f"OOF artifact must contain exactly one repeat with repeat={expected_repeat}")
    if "split_seed" in expected_rows:
        _require_columns(oof, ("split_seed",), "OOF artifact")
        _require_non_null_columns(oof, ("split_seed",), "OOF artifact")
        if oof["split_seed"].unique().to_list() != expected_rows["split_seed"].unique().to_list():
            raise ValueError("OOF split seed mismatch")
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
    if expected_candidate_ids is not None and set(candidate_ids) != set(expected_candidate_ids):
        raise ValueError("Missing or unexpected candidate coverage")
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


def validate_oof_parquet(
    path: Path,
    expected_rows_by_repeat: Iterable[tuple[int, pl.DataFrame]],
    *,
    expected_candidate_ids: Sequence[str],
    execution_mode: str = "full",
    candidate_metadata: dict | None = None,
) -> dict:
    """Validate actual records, materializing at most one candidate/repeat.

    Summary scores are equal means of per-repeat pooled scores. Fold and repeat
    spread are descriptive, not independent-sample confidence intervals.
    """
    parquet = pq.ParquetFile(path)
    required = (*_OOF_REQUIRED_COLUMNS, "split_seed", "execution_mode", "non_comparative")
    if set(required) - set(parquet.schema_arrow.names):
        raise ValueError("OOF Parquet is missing required columns")
    observed = set()
    record_count = 0
    # A bounded scan prevents unknown arms from disappearing behind filters.
    for batch in parquet.iter_batches(batch_size=64_000, columns=[
        "candidate_id", "repeat", "execution_mode", "non_comparative",
    ]):
        frame = pl.from_arrow(batch)
        _require_non_null_columns(frame, tuple(frame.columns), "OOF artifact")
        if frame["execution_mode"].unique().to_list() != [execution_mode] or frame["non_comparative"].unique().to_list() != [execution_mode == "smoke"]:
            raise ValueError("OOF execution mode is incompatible")
        observed.update(frame.select("candidate_id", "repeat").unique().iter_rows())
        record_count += frame.height
    dataset = ds.dataset(path, format="parquet")
    repeat_rows, fold_rows = [], []
    expected_arms = set()
    population_size = None
    repeat_ids = []
    for repeat, expected in expected_rows_by_repeat:
        if type(repeat) is not int or repeat in repeat_ids:
            raise ValueError("Invalid or duplicate expected repeat")
        repeat_ids.append(repeat)
        if population_size is None:
            population_size = expected.height
        elif population_size != expected.height:
            raise ValueError("Repeat populations differ")
        for candidate in expected_candidate_ids:
            expected_arms.add((candidate, repeat))
            table = dataset.scanner(filter=(ds.field("repeat") == repeat) &
                (ds.field("candidate_id") == candidate), batch_size=64_000).to_table()
            part = pl.from_arrow(table)
            del table
            metrics = validate_oof_artifact(part, expected, expected_repeat=repeat,
                                            expected_candidate_ids=(candidate,))[candidate]
            if candidate_metadata is not None:
                for name in _CANDIDATE_METADATA_COLUMNS:
                    if part[name].unique().to_list() != [candidate_metadata[candidate][name]]:
                        raise ValueError(f"OOF {candidate} {name} differs from frozen config")
            split_seed = expected["split_seed"][0]
            repeat_rows.append({"candidate_id": candidate, "repeat": repeat,
                                "split_seed": split_seed, **metrics})
            for fold in sorted(expected["fold"].unique().to_list()):
                fold_part = part.filter(pl.col("fold") == fold)
                values = evaluate_binary_metrics(fold_part["target"].to_numpy(),
                                                  fold_part["prediction"].to_numpy())
                fold_rows.append({"candidate_id": candidate, "repeat": repeat,
                                  "split_seed": split_seed, "fold": fold,
                                  **{name: float(values[name]) for name in _METRIC_NAMES}})
                del fold_part
            del part
    if not repeat_ids or observed != expected_arms:
        raise ValueError("OOF repeat/candidate coverage differs from expected manifest")
    if record_count != population_size * len(expected_arms):
        raise ValueError("OOF record coverage is incomplete")
    candidates = {}
    for candidate in expected_candidate_ids:
        repeats = [row for row in repeat_rows if row["candidate_id"] == candidate]
        folds = [row for row in fold_rows if row["candidate_id"] == candidate]
        candidates[candidate] = {
            "pooled": {name: float(np.mean([row[name] for row in repeats])) for name in _METRIC_NAMES},
            "repeat_std": {name: float(np.std([row[name] for row in repeats], ddof=1))
                           if len(repeats) > 1 else None for name in _METRIC_NAMES},
            "repeats": repeats, "folds": folds,
        }
    return {"repeat_count": len(repeat_ids), "repeat_ids": repeat_ids,
            "split_seeds": [next(row["split_seed"] for row in repeat_rows if row["repeat"] == r)
                            for r in repeat_ids],
            "development_oof_row_count": population_size, "oof_record_count": record_count,
            "repeat_metrics": repeat_rows, "fold_metrics": fold_rows, "candidate_metrics": candidates}


def load_repeated_manifest(path: Path) -> pl.DataFrame:
    """Consume the gen7 artifact, never generate or repair split assignments."""
    manifest = pl.read_parquet(path)
    required = ("building_id", "repeat", "seed", "split", "fold")
    _require_columns(manifest, required, "repeated manifest")
    _require_non_null_columns(manifest, required[:-1], "repeated manifest")
    if _has_duplicate_rows(manifest, ("repeat", "building_id")):
        raise ValueError("Repeated manifest contains duplicate building/repeat keys")
    if sorted(manifest["repeat"].unique().to_list()) != [0, 1, 2]:
        raise ValueError("Expected repeats 0, 1, 2")
    vault_ids, dev_ids, partitions = None, None, set()
    for repeat, seed in enumerate((42, 43, 44)):
        part = manifest.filter(pl.col("repeat") == repeat)
        if part["seed"].unique().to_list() != [seed] or set(part["split"].unique()) != {"dev", "vault"}:
            raise ValueError("Repeated split seed/scope mismatch")
        dev, vault = part.filter(pl.col("split") == "dev"), part.filter(pl.col("split") == "vault")
        if dev.height == 0 or vault.height == 0 or vault["fold"].null_count() != vault.height or dev["fold"].null_count():
            raise ValueError("Expected development buildings with valid folds and vault buildings with null folds")
        if dev.height == 160 and vault.height != 40:
            raise ValueError("Expected frozen 160 development / 40 vault buildings")
        current_vault, current_dev = set(vault["building_id"]), set(dev["building_id"])
        if vault_ids is None:
            vault_ids, dev_ids = current_vault, current_dev
        elif current_vault != vault_ids or current_dev != dev_ids:
            raise ValueError("Development/vault membership changed across repeats")
        if sorted(dev["fold"].unique()) != [0, 1, 2, 3]:
            raise ValueError("Expected four development folds")
        partition = tuple(sorted(tuple(sorted(dev.filter(pl.col("fold") == f)["building_id"]))
                                 for f in range(4)))
        if dev.height > 4 and partition in partitions:
            raise ValueError("Split repetitions are copied partitions or fold-label permutations")
        partitions.add(partition)
    metadata_path = path.with_name("split_metadata_repeated.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        digest = sha256_file(path)
        if metadata.get("split_seeds") != [42, 43, 44] or metadata.get("repeats") != 3 or metadata.get("n_dev_folds") != 4 or metadata.get("artifacts", {}).get("building_folds_sha256") != digest:
            raise ValueError("Repeated split metadata/hash mismatch")
        vault_keys = metadata.get("vault_building_ids") or metadata.get("vault_ids") or ()
        if set(vault_keys) != vault_ids:
            raise ValueError("Repeated split metadata vault mismatch")
        source = metadata.get("source_manifest")
        if source:
            root = path.parent.parent.parent.resolve()
            source_path = (root / source["path"]).resolve()
            if source_path.exists() and source_path.is_relative_to(root / "outputs"):
                if sha256_file(source_path) != source["sha256"]:
                    raise ValueError("Frozen source manifest hash mismatch")
                original = pl.read_parquet(source_path)
                if set(original.filter(pl.col("split") == "vault")["building_id"]) != vault_ids or set(original.filter(pl.col("split") == "dev")["building_id"]) != dev_ids:
                    raise ValueError("Repeated manifest differs from frozen original vault/development")
    return manifest


def sha256_file(path: Path) -> str:
    """Hash a persisted artifact without loading its bytes into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
