"""Measure train/test domain shift using building-level out-of-fold predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from project.config import BUILDING_ID_COLUMN, DEFAULT_SEED, TARGET_COLUMN
from project.models.metrics import evaluate_binary_metrics, positive_class_scores
from project.paths import ProjectPaths

DEFAULT_FOLDS = 5
IMPORTANCE_REPEATS = 5
FOREST_TREES = 80
METRIC_NAMES = ("roc_auc", "average_precision", "log_loss")
NUMERIC_SUMMARY_STATISTICS = ("mean", "median", "missing_rate")
TARGET_NAMES = {TARGET_COLUMN.casefold(), "target"}
ID_NAMES = {"id", "row_id", BUILDING_ID_COLUMN.casefold()}
TIME_NAMES = {"time", "date", "datetime", "timestamp"}
DOMAIN_SPLIT_NAMES = {"fold", "split", "domain", "domain_label", "fold_label", "split_label"}


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for one reusable domain-validation run."""

    prepared_dir: Path
    output_dir: Path
    folds: int = DEFAULT_FOLDS
    seed: int = DEFAULT_SEED
    project_root: Path | None = None


@dataclass(frozen=True)
class PredictorSelection:
    """Source columns eligible for shift-only per-building summaries."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    excluded_fields: dict[str, str]
    categorical_mode_tie_break: str = "lexical_ascending"

    @property
    def numeric_summary_columns(self) -> tuple[str, ...]:
        return tuple(
            f"{column}__{statistic}"
            for column in self.numeric_columns
            for statistic in NUMERIC_SUMMARY_STATISTICS
        )

    @property
    def categorical_summary_columns(self) -> tuple[str, ...]:
        return tuple(f"{column}__mode" for column in self.categorical_columns)

    @property
    def predictor_columns(self) -> tuple[str, ...]:
        return self.numeric_summary_columns + self.categorical_summary_columns

    @property
    def gte_shift_only_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                column
                for column in (*self.numeric_columns, *self.categorical_columns)
                if column.casefold().startswith("gte_")
            )
        )


@dataclass(frozen=True)
class DomainDataset:
    """One classifier row per building, with IDs and counts kept as metadata."""

    features: pd.DataFrame
    labels: np.ndarray
    metadata: pl.DataFrame
    selection: PredictorSelection


def select_predictor_columns(
    train_schema: pl.Schema | dict[str, pl.DataType],
    test_schema: pl.Schema | dict[str, pl.DataType],
) -> PredictorSelection:
    """Select common numeric/categorical fields and record every excluded field."""

    train_fields = dict(train_schema)
    test_fields = dict(test_schema)
    all_fields = sorted(set(train_fields) | set(test_fields))
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    excluded_fields: dict[str, str] = {}

    for column in all_fields:
        train_dtype = train_fields.get(column)
        test_dtype = test_fields.get(column)
        exclusion = _excluded_field_reason(column, train_dtype, test_dtype)
        if exclusion is not None:
            excluded_fields[column] = exclusion
            continue
        if train_dtype is None or test_dtype is None:
            excluded_fields[column] = "not_present_in_both_domains"
            continue

        train_family = _dtype_family(train_dtype)
        test_family = _dtype_family(test_dtype)
        if train_family is None or test_family is None:
            excluded_fields[column] = "unsupported_dtype"
        elif train_family != test_family:
            excluded_fields[column] = "incompatible_train_test_dtypes"
        elif train_family == "numeric":
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    selection = PredictorSelection(
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        excluded_fields=excluded_fields,
    )
    _validate_summary_column_names(selection)
    _assert_no_forbidden_predictors(selection)
    return selection


def summarize_buildings(
    frame: pl.DataFrame | pl.LazyFrame,
    selection: PredictorSelection,
) -> pl.DataFrame:
    """Aggregate hourly rows to one deterministic summary row per building."""

    lazy_frame = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    schema = lazy_frame.collect_schema()
    if BUILDING_ID_COLUMN not in schema:
        raise ValueError(f"Required building identifier '{BUILDING_ID_COLUMN}' is missing")

    null_building_ids = (
        lazy_frame.select(pl.col(BUILDING_ID_COLUMN).null_count())
        .collect(engine="streaming")
        .item()
    )
    if null_building_ids:
        raise ValueError(f"{BUILDING_ID_COLUMN} contains null building IDs")

    aggregations: list[pl.Expr] = [pl.len().alias("hourly_row_count")]
    for column in selection.numeric_columns:
        value = pl.col(column)
        is_float = str(schema[column]).casefold().startswith("float")
        clean_value = value.fill_nan(None) if is_float else value
        missing = value.is_null() | value.is_nan() if is_float else value.is_null()
        aggregations.extend(
            [
                clean_value.mean().alias(f"{column}__mean"),
                clean_value.median().alias(f"{column}__median"),
                missing.cast(pl.Float64).mean().alias(f"{column}__missing_rate"),
            ]
        )

    summary = lazy_frame.group_by(BUILDING_ID_COLUMN).agg(aggregations)
    for column in selection.categorical_columns:
        mode_column = f"{column}__mode"
        modes = (
            lazy_frame.filter(pl.col(column).is_not_null())
            .group_by([BUILDING_ID_COLUMN, column])
            .agg(
                pl.len().alias("__mode_count"),
                pl.col(column).cast(pl.String).first().alias("__mode_sort_key"),
            )
            .sort(
                [BUILDING_ID_COLUMN, "__mode_count", "__mode_sort_key"],
                descending=[False, True, False],
            )
            .unique(
                subset=[BUILDING_ID_COLUMN], keep="first", maintain_order=True
            )
            .select(
                BUILDING_ID_COLUMN,
                pl.col(column).alias(mode_column),
            )
        )
        summary = summary.join(modes, on=BUILDING_ID_COLUMN, how="left")

    materialized = summary.sort(BUILDING_ID_COLUMN).collect(engine="streaming")
    if materialized.height != materialized.get_column(BUILDING_ID_COLUMN).n_unique():
        raise AssertionError("Building summary must contain exactly one row per building")
    if materialized.height == 0:
        raise ValueError("Prepared table contains no buildings")
    return materialized


def build_domain_dataset(
    train: pl.DataFrame | pl.LazyFrame,
    test: pl.DataFrame | pl.LazyFrame,
) -> DomainDataset:
    """Create X/y and metadata at the building—not hourly-row—level."""

    train_schema = _collect_schema(train)
    test_schema = _collect_schema(test)
    if BUILDING_ID_COLUMN not in train_schema or BUILDING_ID_COLUMN not in test_schema:
        raise ValueError(f"Both prepared tables must contain '{BUILDING_ID_COLUMN}'")

    selection = select_predictor_columns(train_schema, test_schema)
    train_summary = summarize_buildings(train, selection)
    test_summary = summarize_buildings(test, selection)
    _assert_disjoint_buildings(train_summary, test_summary)

    predictor_columns = selection.predictor_columns
    _assert_no_forbidden_predictors(selection)
    if not predictor_columns:
        raise ValueError("No common numeric or categorical predictors remain after exclusions")

    train_domain = train_summary.with_columns(pl.lit(0, dtype=pl.Int8).alias("domain_label"))
    test_domain = test_summary.with_columns(pl.lit(1, dtype=pl.Int8).alias("domain_label"))
    all_buildings = pl.concat([train_domain, test_domain], how="vertical_relaxed")
    features = all_buildings.select(predictor_columns).to_pandas()
    metadata = all_buildings.select(
        BUILDING_ID_COLUMN, "domain_label", "hourly_row_count"
    )
    labels = metadata.get_column("domain_label").to_numpy().astype(np.int8, copy=False)

    if BUILDING_ID_COLUMN in features.columns or "hourly_row_count" in features.columns:
        raise AssertionError("Identifiers and row counts must remain outside classifier X")
    return DomainDataset(features, labels, metadata, selection)


def run_adversarial_validation(config: ValidationConfig) -> dict[str, Any]:
    """Run five-fold domain OOF evaluation and write the three requested artifacts."""

    _validate_config(config)
    project_root = _resolve_project_root(config.project_root)
    prepared_dir = _resolve_input_directory(config.prepared_dir, project_root)
    output_dir = _validate_output_directory(config.output_dir, project_root)
    train_path = prepared_dir / "train_joined.parquet"
    test_path = prepared_dir / "test_joined.parquet"
    _require_input_artifacts(train_path, test_path)

    train = pl.scan_parquet(train_path)
    test = pl.scan_parquet(test_path)
    dataset = build_domain_dataset(train, test)
    _validate_domain_support(dataset.labels, config.folds)

    oof_predictions, fold_ids, fold_metrics, fold_importances = _fit_oof_predictions(
        dataset, config
    )
    pooled_metrics = _evaluate_scores(dataset.labels, oof_predictions)
    prevalence = float(np.mean(dataset.labels == 1))
    null_log_loss = _binary_entropy(prevalence)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = _validate_output_directory(output_dir, project_root)
    oof_path = output_dir / "domain_oof.parquet"
    importance_path = output_dir / "fold_averaged_importance.csv"
    report_path = output_dir / "adversarial_validation.json"

    oof = dataset.metadata.with_columns(
        pl.Series("fold", fold_ids, dtype=pl.Int8),
        pl.Series("domain_test_probability", oof_predictions, dtype=pl.Float64),
    ).select(
        BUILDING_ID_COLUMN,
        "domain_label",
        "fold",
        "domain_test_probability",
        "hourly_row_count",
    )
    _validate_oof_artifact(oof, dataset.metadata.height, config.folds)
    oof.write_parquet(oof_path)

    importance = _fold_averaged_importance(fold_importances)
    importance.write_csv(importance_path)

    report = _build_report(
        config=config,
        dataset=dataset,
        train_path=train_path,
        test_path=test_path,
        oof_path=oof_path,
        importance_path=importance_path,
        report_path=report_path,
        pooled_metrics=pooled_metrics,
        fold_metrics=fold_metrics,
        prevalence=prevalence,
        null_log_loss=null_log_loss,
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse the standalone script's command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Measure train/test domain shift with building-level OOF predictions."
    )
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--folds",
        type=int,
        choices=[DEFAULT_FOLDS],
        default=DEFAULT_FOLDS,
        help="Stratified OOF fold count (this diagnostic contract fixes it at 5).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Run the CLI and return a process-style exit code."""

    parsed = parse_args(args)
    try:
        project_paths = ProjectPaths.resolve()
        config = ValidationConfig(
            prepared_dir=parsed.prepared_dir or project_paths.prepared_dir,
            output_dir=parsed.output_dir or project_paths.shift_dir,
            folds=parsed.folds,
            seed=parsed.seed,
            project_root=project_paths.root,
        )
        report = run_adversarial_validation(config)
    except (FileNotFoundError, OSError, ValueError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report["metrics"]["pooled_oof"], sort_keys=True))
    return 0


def _fit_oof_predictions(
    dataset: DomainDataset,
    config: ValidationConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, list[float]]]:
    fold_count = config.folds
    splitter = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=config.seed,
    )
    probabilities = np.full(dataset.labels.shape[0], np.nan, dtype=np.float64)
    fold_ids = np.full(dataset.labels.shape[0], -1, dtype=np.int8)
    fold_metrics: list[dict[str, Any]] = []
    importances = {column: [] for column in dataset.features.columns}

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(dataset.features, dataset.labels)
    ):
        estimator = _make_estimator(dataset.selection, config.seed + fold)
        estimator.fit(
            dataset.features.iloc[train_indices], dataset.labels[train_indices]
        )
        validation_features = dataset.features.iloc[validation_indices]
        validation_labels = dataset.labels[validation_indices]
        fold_probabilities = positive_class_scores(estimator, validation_features)
        fold_report = _evaluate_scores(validation_labels, fold_probabilities)
        probabilities[validation_indices] = fold_probabilities
        fold_ids[validation_indices] = fold

        fold_metrics.append(
            {
                "fold": fold,
                "training_buildings": int(train_indices.size),
                "validation_buildings": int(validation_indices.size),
                **fold_report,
            }
        )
        permutation = permutation_importance(
            estimator,
            validation_features,
            validation_labels,
            scoring="roc_auc",
            n_repeats=IMPORTANCE_REPEATS,
            random_state=config.seed + fold,
            n_jobs=1,
        )
        for column, mean_importance in zip(
            dataset.features.columns, permutation.importances_mean, strict=True
        ):
            importances[column].append(float(mean_importance))

    if (fold_ids < 0).any() or not np.isfinite(probabilities).all():
        raise ValueError("OOF validation did not produce one finite prediction per building")
    return probabilities, fold_ids, fold_metrics, importances


def _make_estimator(selection: PredictorSelection, seed: int) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if selection.numeric_summary_columns:
        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                )
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, list(selection.numeric_summary_columns))
        )
    if selection.categorical_summary_columns:
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True,
                    ),
                ),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(selection.categorical_summary_columns),
            )
        )

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    classifier = RandomForestClassifier(
        n_estimators=FOREST_TREES,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _evaluate_scores(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = evaluate_binary_metrics(labels, probabilities)
    return {name: float(metrics[name]) for name in METRIC_NAMES}


def _fold_metric_summary(
    fold_metrics: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([fold[metric] for fold in fold_metrics])),
            "std": float(np.std([fold[metric] for fold in fold_metrics], ddof=0)),
        }
        for metric in METRIC_NAMES
    }


def _fold_averaged_importance(
    fold_importances: dict[str, list[float]],
) -> pl.DataFrame:
    rows = [
        {
            "feature": feature,
            "importance_mean": float(np.mean(values)),
            "importance_std": float(np.std(values, ddof=0)),
            "fold_count": len(values),
        }
        for feature, values in sorted(fold_importances.items())
    ]
    return pl.DataFrame(
        rows,
        schema={
            "feature": pl.String,
            "importance_mean": pl.Float64,
            "importance_std": pl.Float64,
            "fold_count": pl.UInt8,
        },
    )


def _build_report(
    config: ValidationConfig,
    dataset: DomainDataset,
    train_path: Path,
    test_path: Path,
    oof_path: Path,
    importance_path: Path,
    report_path: Path,
    pooled_metrics: dict[str, float],
    fold_metrics: list[dict[str, Any]],
    prevalence: float,
    null_log_loss: float,
) -> dict[str, Any]:
    train_rows = int(dataset.metadata.filter(pl.col("domain_label") == 0)
                     .get_column("hourly_row_count").sum())
    test_rows = int(dataset.metadata.filter(pl.col("domain_label") == 1)
                    .get_column("hourly_row_count").sum())
    selection = dataset.selection
    return {
        "schema_version": 1,
        "diagnostic": "Train-versus-test building-domain classification with stratified OOF predictions.",
        "domain_labels": {"train": 0, "test": 1, "positive_class": "test"},
        "configuration": {
            "folds": config.folds,
            "seed": config.seed,
            "splitter": "StratifiedKFold(shuffle=True)",
            "classifier": "RandomForestClassifier",
            "forest_trees": FOREST_TREES,
            "permutation_importance_repeats": IMPORTANCE_REPEATS,
        },
        "counts": {
            "train_buildings": int(np.count_nonzero(dataset.labels == 0)),
            "test_buildings": int(np.count_nonzero(dataset.labels == 1)),
            "total_buildings": int(dataset.labels.size),
            "train_hourly_rows": train_rows,
            "test_hourly_rows": test_rows,
        },
        "null_baselines": {
            "domain_prevalence_p": prevalence,
            "average_precision": prevalence,
            "binary_entropy_log_loss": null_log_loss,
        },
        "metrics": {
            "pooled_oof": pooled_metrics,
            "fold_metrics": fold_metrics,
            "fold_mean_std": _fold_metric_summary(fold_metrics),
            "fold_std_ddof": 0,
        },
        "features": {
            "numeric_source_columns": list(selection.numeric_columns),
            "categorical_source_columns": list(selection.categorical_columns),
            "numeric_summary_columns": list(selection.numeric_summary_columns),
            "categorical_summary_columns": list(selection.categorical_summary_columns),
            "classifier_predictors": list(selection.predictor_columns),
            "metadata_only_columns": [BUILDING_ID_COLUMN, "hourly_row_count"],
            "gte_shift_only_fields": list(selection.gte_shift_only_fields),
            "gte_note": (
                "gte_* results are shift-only diagnostics and are not production feature selection; "
                "unknown provenance remains blocked by the separate feature-availability contract."
            ),
            "importance_use": (
                "Permutation importance is diagnostic only; it does not select, remove, "
                "or filter production features."
            ),
        },
        "excluded_fields": selection.excluded_fields,
        "categorical_mode_tie_break": selection.categorical_mode_tie_break,
        "artifacts": {
            "input_train_parquet": {
                "path": str(train_path.resolve()),
                "sha256": _file_sha256(train_path),
            },
            "input_test_parquet": {
                "path": str(test_path.resolve()),
                "sha256": _file_sha256(test_path),
            },
            "domain_oof_parquet": {
                "path": str(oof_path.resolve()),
                "sha256": _file_sha256(oof_path),
            },
            "fold_averaged_importance_csv": {
                "path": str(importance_path.resolve()),
                "sha256": _file_sha256(importance_path),
            },
            "report_json": {"path": str(report_path.resolve())},
        },
    }


def _collect_schema(frame: pl.DataFrame | pl.LazyFrame) -> pl.Schema:
    return frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema


def _dtype_family(dtype: pl.DataType) -> str | None:
    if dtype.is_numeric() and dtype != pl.Boolean:
        return "numeric"
    if dtype in {pl.String, pl.Boolean, pl.Categorical} or str(dtype).startswith("Enum"):
        return "categorical"
    return None


def _excluded_field_reason(
    column: str,
    train_dtype: pl.DataType | None,
    test_dtype: pl.DataType | None,
) -> str | None:
    normalized = column.casefold()
    if normalized in TARGET_NAMES:
        return "target"
    if normalized in ID_NAMES or normalized.endswith("_id") or normalized.startswith("id_"):
        return "identifier"
    if _is_time_field(normalized, train_dtype, test_dtype):
        return "time_identifier"
    if _is_domain_split_field(normalized):
        return "fold_split_or_domain_label"
    if normalized == "source" or normalized.startswith("source_"):
        return "source_metadata"
    return None


def _is_time_field(
    normalized_name: str,
    train_dtype: pl.DataType | None,
    test_dtype: pl.DataType | None,
) -> bool:
    if normalized_name in TIME_NAMES or normalized_name.endswith(
        ("_time", "_date", "_datetime", "_timestamp")
    ):
        return True
    return any(_is_temporal_dtype(dtype) for dtype in (train_dtype, test_dtype) if dtype is not None)


def _is_temporal_dtype(dtype: pl.DataType) -> bool:
    dtype_name = str(dtype).casefold()
    return dtype_name in {"date", "time"} or "datetime" in dtype_name


def _is_domain_split_field(normalized_name: str) -> bool:
    return (
        normalized_name in DOMAIN_SPLIT_NAMES
        or normalized_name.startswith(("fold_", "split_", "domain_"))
        or normalized_name.endswith(("_fold", "_split", "_domain"))
    )


def _validate_summary_column_names(selection: PredictorSelection) -> None:
    names = selection.predictor_columns
    if len(names) != len(set(names)):
        raise ValueError("Summary feature names collide; rename source columns before validation")


def _assert_no_forbidden_predictors(selection: PredictorSelection) -> None:
    forbidden = {
        column: reason
        for column, reason in selection.excluded_fields.items()
        if reason in {
            "target",
            "identifier",
            "time_identifier",
            "fold_split_or_domain_label",
            "source_metadata",
        }
    }
    allowed_source_columns = set(selection.numeric_columns) | set(selection.categorical_columns)
    leaked = sorted(set(forbidden) & allowed_source_columns)
    if leaked:
        raise AssertionError(f"Forbidden columns leaked into domain predictors: {leaked}")
    if BUILDING_ID_COLUMN in allowed_source_columns:
        raise AssertionError("building_id must remain metadata, never a classifier predictor")


def _assert_disjoint_buildings(train: pl.DataFrame, test: pl.DataFrame) -> None:
    train_ids = set(train.get_column(BUILDING_ID_COLUMN).to_list())
    test_ids = set(test.get_column(BUILDING_ID_COLUMN).to_list())
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Train and test building ID overlap detected: {sorted(overlap)[:5]}")


def _validate_domain_support(labels: np.ndarray, folds: int) -> None:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Domain dataset must contain both train=0 and test=1 buildings")
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    if int(counts.min()) < folds:
        raise ValueError(
            f"Each domain needs at least {folds} buildings for {folds}-fold stratified OOF; "
            f"found train={counts[0]}, test={counts[1]}"
        )


def _validate_oof_artifact(oof: pl.DataFrame, expected_rows: int, folds: int) -> None:
    if oof.height != expected_rows:
        raise ValueError("OOF artifact must contain exactly one row per building")
    if oof.get_column(BUILDING_ID_COLUMN).n_unique() != expected_rows:
        raise ValueError("OOF artifact contains duplicate building IDs")
    probabilities = oof.get_column("domain_test_probability").to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("OOF domain probabilities must be finite and within [0, 1]")
    if set(oof.get_column("fold").to_list()) != set(range(folds)):
        raise ValueError("OOF artifact does not contain every configured fold")


def _validate_config(config: ValidationConfig) -> None:
    if config.folds != DEFAULT_FOLDS:
        raise ValueError(f"This validation contract requires exactly {DEFAULT_FOLDS} folds")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _resolve_project_root(project_root: Path | None) -> Path:
    return (
        Path(project_root).resolve()
        if project_root is not None
        else ProjectPaths.resolve().root
    )


def _resolve_input_directory(prepared_dir: Path, project_root: Path) -> Path:
    path = Path(prepared_dir)
    return (path if path.is_absolute() else project_root / path).resolve()


def _validate_output_directory(output_dir: Path, project_root: Path) -> Path:
    output_root = project_root / "outputs"
    resolved_output_root = output_root.resolve()
    try:
        resolved_output_root.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Project outputs directory resolves outside the project root: {resolved_output_root}"
        ) from error

    path = Path(output_dir)
    resolved_output = (path if path.is_absolute() else project_root / path).resolve()
    try:
        relative_output = resolved_output.relative_to(resolved_output_root)
    except ValueError as error:
        raise ValueError(
            f"Output directory {resolved_output} must resolve under project outputs/ "
            f"({resolved_output_root})"
        ) from error
    if not relative_output.parts:
        raise ValueError("Output directory must be a child of project outputs/")
    return resolved_output


def _require_input_artifacts(train_path: Path, test_path: Path) -> None:
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required prepared input artifact not found: {path}")


def _binary_entropy(prevalence: float) -> float:
    if not 0.0 < prevalence < 1.0:
        raise ValueError("Binary entropy baseline requires both domain labels")
    return float(-prevalence * math.log(prevalence) - (1 - prevalence) * math.log1p(-prevalence))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
