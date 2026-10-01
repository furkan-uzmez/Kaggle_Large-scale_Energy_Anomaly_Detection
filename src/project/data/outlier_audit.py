"""Read-only, development-building outlier diagnostics for observed-hour EDA.

These full-development summaries are descriptive, not causal detectors or
inference features. Flags mean suspected/review-worthy extremes, not established
physical errors. No readings, labels, assignments, or source objects are changed.
"""

from __future__ import annotations

from numbers import Integral

import polars as pl

REQUIRED_COLUMNS = ("building_id", "timestamp", "meter_reading", "anomaly")
QUANTILES = {
    "q001": 0.001, "q01": 0.01, "q25": 0.25,
    "q75": 0.75, "q99": 0.99, "q999": 0.999,
}
COUNT_COLUMNS = (
    "rows", "missing_count", "nan_count", "nonfinite_count", "finite_count",
    "zero_count", "negative_count",
)
STAT_COLUMNS = (
    "min", "q001", "q01", "q25", "median", "q75", "q99", "q999", "max",
    "iqr", "mad",
)
PROFILE_SCHEMA = {
    "feature": pl.String,
    **dict.fromkeys(COUNT_COLUMNS, pl.Int64),
    **dict.fromkeys(STAT_COLUMNS, pl.Float64),
}
IQR_MULTIPLIER = 1.5
MAD_CUTOFF = 3.5
MAD_NORMAL_SCALE = 0.67448975
SAMPLE_SEED = 42


def _validate_frame_schema(frame: pl.LazyFrame) -> pl.Schema:
    if not isinstance(frame, pl.LazyFrame):
        raise TypeError("frame must be a Polars LazyFrame")
    schema = frame.collect_schema()
    missing = set(REQUIRED_COLUMNS).difference(schema)
    if missing:
        raise ValueError(f"Missing required frame columns: {sorted(missing)}")
    if not (schema["building_id"].is_integer() or schema["building_id"] == pl.String):
        raise TypeError("building_id must be integer or String")
    if schema["timestamp"] != pl.Date and not isinstance(schema["timestamp"], pl.Datetime):
        raise TypeError("timestamp must be Date or Datetime, not an unparsed string")
    if not schema["anomaly"].is_integer():
        raise TypeError("anomaly must contain integer binary labels 0/1")
    if not (schema["meter_reading"].is_numeric() or schema["meter_reading"] == pl.Null):
        raise TypeError("meter_reading must be numeric (or Null for all-missing data)")
    return schema


def _validate_manifest(manifest: pl.DataFrame, building_dtype: pl.DataType) -> None:
    if not isinstance(manifest, pl.DataFrame):
        raise TypeError("manifest must be a Polars DataFrame")
    missing = {"building_id", "split", "fold"}.difference(manifest.columns)
    if missing:
        raise ValueError(f"Building manifest is missing assignment columns: {sorted(missing)}")
    if manifest.is_empty():
        raise ValueError("Building manifest must not be empty")
    keys = manifest.get_column("building_id")
    if keys.null_count() or keys.n_unique() != manifest.height:
        raise ValueError("Building manifest building_id assignments must be nonnull and unique")
    if keys.dtype != building_dtype:
        raise TypeError("Building manifest building_id dtype must match the frame")
    if manifest.schema["split"] != pl.String:
        raise TypeError("Building manifest split assignments must be String")
    if manifest.filter(pl.col("split").is_null() | ~pl.col("split").is_in(["dev", "vault"])).height:
        raise ValueError("Building manifest has missing or unknown split assignments")
    if not (manifest.schema["fold"].is_integer() or manifest.schema["fold"] == pl.Null):
        raise TypeError("Building manifest fold must be integer or null")
    invalid_folds = manifest.filter(
        ((pl.col("split") == "dev") & (pl.col("fold").is_null() | (pl.col("fold") < 0)))
        | ((pl.col("split") == "vault") & pl.col("fold").is_not_null())
    )
    if invalid_folds.height:
        raise ValueError("Building manifest requires nonnegative dev folds and null vault folds")


def _safe_numeric_columns(columns: list[str], schema: pl.Schema) -> list[str]:
    if not isinstance(columns, list) or any(not isinstance(c, str) for c in columns):
        raise TypeError("numeric_columns must be a list of column names")
    if len(set(columns)) != len(columns):
        raise ValueError("numeric_columns must not contain duplicate feature names")
    safe = [
        column for column in columns
        if column.casefold() not in {"anomaly", "timestamp"}
        and not column.casefold().endswith("_id")
        and not column.casefold().startswith("gte_")
    ]
    for column in safe:
        if column not in schema:
            raise ValueError(f"Requested numeric feature column is missing: {column}")
        if not (schema[column].is_numeric() or schema[column] == pl.Null):
            raise TypeError(f"Requested numeric feature column is not numeric: {column}")
    return safe


def _development_frame(frame: pl.LazyFrame, manifest: pl.DataFrame) -> pl.LazyFrame:
    # Validate assignments using identifiers only, never vault readings or labels.
    unassigned = (
        frame.select("building_id").unique()
        .join(manifest.lazy().select("building_id"), on="building_id", how="anti")
        .select(pl.len()).collect(engine="streaming").item()
    )
    if unassigned:
        raise ValueError("Frame has missing or unknown building_id assignments")
    return frame.join(
        manifest.filter(pl.col("split") == "dev").lazy().select("building_id"),
        on="building_id", how="semi",
    )


def _validate_development_values(development: pl.LazyFrame) -> None:
    checks = development.select(
        pl.len().alias("rows"),
        (pl.col("anomaly").is_null() | ~pl.col("anomaly").is_in([0, 1]))
        .sum().alias("invalid_labels"),
        pl.col("timestamp").null_count().alias("null_timestamps"),
    ).collect(engine="streaming").row(0, named=True)
    if not checks["rows"]:
        raise ValueError("No development rows are present")
    if checks["invalid_labels"]:
        raise ValueError("Development anomaly labels must be nonnull binary 0/1")
    if checks["null_timestamps"]:
        raise ValueError("Development timestamp values must be nonnull")


def _sum_count(expression: pl.Expr) -> pl.Expr:
    return expression.fill_null(False).sum().cast(pl.Int64)


def _distribution(column: str) -> dict[str, pl.Expr]:
    value = pl.col(column).cast(pl.Float64)
    finite = value.filter(value.is_finite())
    quartiles = [finite.quantile(q, interpolation="linear") for q in (0.25, 0.75)]
    return {
        "rows": pl.len().cast(pl.Int64),
        "missing_count": value.null_count().cast(pl.Int64),
        "nan_count": _sum_count(value.is_nan()),
        "nonfinite_count": _sum_count(~value.is_finite()),
        "finite_count": finite.count().cast(pl.Int64),
        "zero_count": _sum_count(value.is_finite() & (value == 0)),
        "negative_count": _sum_count(value.is_finite() & (value < 0)),
        "min": finite.min(),
        **{name: finite.quantile(q, interpolation="linear") for name, q in QUANTILES.items()},
        "median": finite.median(),
        "max": finite.max(),
        "iqr": quartiles[1] - quartiles[0],
        "mad": (finite - finite.median()).abs().median(),
    }


def _numeric_profile(development: pl.LazyFrame, columns: list[str]) -> pl.DataFrame:
    if not columns:
        return pl.DataFrame(schema=PROFILE_SCHEMA)
    # Materialize a single wide aggregate row, not the observation/feature matrix.
    summary = development.select([
        expression.alias(f"{index}__{name}")
        for index, column in enumerate(columns)
        for name, expression in _distribution(column).items()
    ]).collect(engine="streaming")
    return pl.concat([
        summary.select(
            pl.lit(column).alias("feature"),
            *[pl.col(f"{index}__{name}").alias(name) for name in (*COUNT_COLUMNS, *STAT_COLUMNS)],
        )
        for index, column in enumerate(columns)
    ])


def _meter_summaries(meter: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    statistics = _distribution("meter_reading")
    class_columns = ("rows", "finite_count", "min", "q01", "median", "q99", "max", "zero_count", "negative_count")
    by_class = meter.group_by("anomaly").agg([
        statistics[name].alias(name) for name in class_columns
    ]).collect(engine="streaming").sort("anomaly")
    building_columns = ("rows", "finite_count", "q25", "median", "q75", "iqr", "mad")
    by_building = meter.group_by("building_id").agg(
        *[statistics[name].alias(name) for name in building_columns],
        pl.col("anomaly").sum().cast(pl.Int64).alias("anomaly_rows"),
    ).with_columns(
        ((pl.col("iqr") > 0) & pl.col("iqr").is_finite()).fill_null(False).alias("iqr_available"),
        ((pl.col("mad") > 0) & pl.col("mad").is_finite()).fill_null(False).alias("mad_available"),
    ).collect(engine="streaming").sort("building_id")
    return by_class, by_building


def _flagged_meter(meter: pl.LazyFrame, buildings: pl.DataFrame) -> pl.LazyFrame:
    bounds = buildings.select(
        "building_id", "q25", "median", "q75", "iqr", "mad", "iqr_available", "mad_available",
    )
    value = pl.col("meter_reading").cast(pl.Float64)
    return meter.join(bounds.lazy(), on="building_id", how="inner").with_columns(
        pl.when(value.is_finite() & pl.col("iqr_available"))
        .then((value < pl.col("q25") - IQR_MULTIPLIER * pl.col("iqr"))
              | (value > pl.col("q75") + IQR_MULTIPLIER * pl.col("iqr")))
        .otherwise(None).alias("_iqr_flag"),
        pl.when(value.is_finite() & pl.col("mad_available"))
        .then((value - pl.col("median")).abs() > MAD_CUTOFF * pl.col("mad") / MAD_NORMAL_SCALE)
        .otherwise(None).alias("_mad_flag"),
    )


def _flag_aggregates() -> list[pl.Expr]:
    return [
        expression
        for method in ("iqr", "mad")
        for expression in (
            pl.when(pl.col(f"_{method}_flag").count() > 0)
            .then(pl.col(f"_{method}_flag").sum().cast(pl.Int64))
            .otherwise(pl.lit(None, dtype=pl.Int64)).alias(f"{method}_flag_count"),
            pl.col(f"_{method}_flag").count().cast(pl.Int64).alias(f"{method}_evaluable_count"),
        )
    ]


def _flag_summaries(flagged: pl.LazyFrame, buildings: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    building_flags = flagged.group_by("building_id").agg(_flag_aggregates()).collect(engine="streaming")
    by_building = buildings.join(building_flags, on="building_id").select(
        "building_id", "rows", "finite_count", "anomaly_rows", "q25", "median", "q75", "iqr", "mad",
        "iqr_available", "mad_available", "iqr_flag_count", "mad_flag_count",
        "iqr_evaluable_count", "mad_evaluable_count",
    ).sort("building_id")
    statistics = _distribution("meter_reading")
    by_month = flagged.with_columns(
        pl.col("timestamp").dt.truncate("1mo").cast(pl.Date).alias("month"),
    ).group_by("month").agg(
        statistics["rows"].alias("rows"), statistics["finite_count"].alias("finite_count"),
        pl.col("anomaly").sum().cast(pl.Int64).alias("anomaly_rows"),
        *_flag_aggregates(),
        statistics["zero_count"].alias("zero_count"), statistics["negative_count"].alias("negative_count"),
    ).select(
        "month", "rows", "finite_count", "anomaly_rows", "iqr_flag_count", "mad_flag_count",
        "zero_count", "negative_count", "iqr_evaluable_count", "mad_evaluable_count",
    ).collect(engine="streaming").sort("month")
    return by_building, by_month


def _plot_sample(meter: pl.LazyFrame, by_class: pl.DataFrame, limit: int) -> pl.DataFrame:
    counts = {row["anomaly"]: row["finite_count"] for row in by_class.to_dicts() if row["finite_count"]}
    if len(counts) > limit:
        raise ValueError("sample_limit cannot represent both finite classes; use at least 2")
    if not counts:
        return pl.DataFrame(schema=dict(meter.collect_schema()))
    labels = sorted(counts)
    quotas = {label: min(counts[label], limit // len(labels) + (index < limit % len(labels)))
              for index, label in enumerate(labels)}
    remaining = min(limit, sum(counts.values())) - sum(quotas.values())
    for label in labels:
        extra = min(remaining, counts[label] - quotas[label])
        quotas[label] += extra
        remaining -= extra
    ranked = meter.filter(pl.col("meter_reading").cast(pl.Float64).is_finite()).with_columns(
        pl.struct(REQUIRED_COLUMNS).hash(seed=SAMPLE_SEED).alias("_sample_key"),
    )
    samples = [
        ranked.filter(pl.col("anomaly") == label)
        .bottom_k(quota, by=["_sample_key", *REQUIRED_COLUMNS])
        .select(REQUIRED_COLUMNS).collect(engine="streaming")
        for label, quota in quotas.items()
    ]
    return pl.concat(samples).sort(["building_id", "timestamp", "anomaly", "meter_reading"])


def build_outlier_audit(
    frame: pl.LazyFrame,
    manifest: pl.DataFrame,
    numeric_columns: list[str],
    sample_limit: int = 12000,
) -> dict[str, pl.DataFrame]:
    """Summarize development buildings without consulting vault/test values.

    ``manifest`` requires unique, nonnull building_id plus split (dev/vault) and
    fold (nonnegative for dev, null for vault). All frame identifiers must have
    assignments. Development labels must be nonnull integer 0/1 and timestamps
    Date/Datetime. Requested identifier, target, timestamp and gte_* columns are
    excluded; other requested features must exist and be numeric or all-null.

    Quantiles are exact, linearly interpolated, on finite values only. Null,
    NaN and nonfinite counts are explicit: nonfinite = NaN + +/-Inf, not null.
    Within each development building, IQR = q75 - q25 and
    MAD = median(abs(x - median(x))). IQR flags use strict 1.5-IQR fences;
    MAD flags use abs(x - median) > 3.5 * MAD / 0.67448975. Zero/null/nonfinite
    scales make the method unavailable: flag counts are null, eligible counts
    zero. Monthly counts aggregate these same building-level flags. They are
    full-development descriptions, not seasonal, causal or physical-error tests.

    The finite-only plotting sample is class-balanced with unused quotas
    redistributed, deterministically hash-ranked (seed 42), at most sample_limit
    rows. It is not prevalence-representative and never determines statistics.
    Hash reproducibility is within the installed Polars version. sample_limit
    must be positive and at least 2 when both finite classes exist.

    Only compact aggregates and bounded samples are collected. Exact quantiles
    and MAD can retain column/group values internally; streaming collection does
    not imply constant-memory quantile computation. No input mutation or writes.
    vault_buildings_excluded counts vault assignments in the manifest, including
    assignments absent from the supplied frame.
    """
    schema = _validate_frame_schema(frame)
    _validate_manifest(manifest, schema["building_id"])
    columns = _safe_numeric_columns(numeric_columns, schema)
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, Integral):
        raise TypeError("sample_limit must be a positive integer")
    if sample_limit < 1:
        raise ValueError("sample_limit must be a positive integer")
    projected = frame.select(list(dict.fromkeys([*REQUIRED_COLUMNS, *columns])))
    development = _development_frame(projected, manifest)
    _validate_development_values(development)
    profile = _numeric_profile(development, columns)
    meter = development.select(REQUIRED_COLUMNS)
    by_class, buildings = _meter_summaries(meter)
    by_building, by_month = _flag_summaries(_flagged_meter(meter, buildings), buildings)
    sample = _plot_sample(meter, by_class, int(sample_limit))
    scope = pl.DataFrame({
        "development_rows": [by_building.get_column("rows").sum()],
        "development_buildings": [by_building.height],
        "sample_rows": [sample.height],
        "vault_buildings_excluded": [manifest.filter(pl.col("split") == "vault").height],
    }, schema=dict.fromkeys((
        "development_rows", "development_buildings", "sample_rows", "vault_buildings_excluded",
    ), pl.Int64))
    return {
        "numeric_profile": profile, "meter_by_class": by_class,
        "meter_by_building": by_building, "meter_by_month": by_month,
        "plot_sample": sample, "scope": scope,
    }
