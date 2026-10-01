from __future__ import annotations

import importlib
import importlib.util
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal


def _audit(frame, manifest, numeric_columns=None, sample_limit=12000):
    name = "project.data.outlier_audit"
    assert importlib.util.find_spec(name) is not None, (
        "A development-only outlier audit implementation is required"
    )
    function = importlib.import_module(name).build_outlier_audit
    return function(
        frame, manifest,
        ["meter_reading"] if numeric_columns is None else numeric_columns,
        sample_limit=sample_limit,
    )


def _manifest() -> pl.DataFrame:
    return pl.DataFrame({
        "building_id": [1, 2, 9],
        "split": ["dev", "dev", "vault"],
        "fold": [0, 1, None],
    })


def _frame() -> pl.DataFrame:
    return pl.DataFrame({
        "building_id": [1] * 5 + [2] * 3 + [9],
        "timestamp": [datetime(2024, 1, 31, tzinfo=UTC)] * 4
        + [datetime(2024, 2, 1, tzinfo=UTC)] * 5,
        "meter_reading": [-2.0, 0.0, 2.0, 4.0, 100.0, 10.0, 10.0, 10.0, 1e12],
        "anomaly": [0, 0, 0, 0, 1, 0, 1, 0, 1],
    })


def test_exact_full_development_profile_and_group_flag_counts():
    result = _audit(_frame().lazy(), _manifest())
    assert set(result) == {
        "numeric_profile", "meter_by_class", "meter_by_building",
        "meter_by_month", "plot_sample", "scope",
    }
    profile = result["numeric_profile"].row(0, named=True)
    assert profile["feature"] == "meter_reading"
    expected = {
        "rows": 8, "missing_count": 0, "nan_count": 0, "nonfinite_count": 0,
        "finite_count": 8, "zero_count": 1, "negative_count": 1,
        "min": -2, "q001": -1.986, "q01": -1.86, "q25": 1.5,
        "median": 7, "q75": 10, "q99": 93.7, "q999": 99.37,
        "max": 100, "iqr": 8.5, "mad": 4,
    }
    for field, value in expected.items():
        assert profile[field] == pytest.approx(value), field

    first, second = result["meter_by_building"].to_dicts()
    assert first["building_id"] == 1
    assert (first["q25"], first["median"], first["q75"]) == (0, 2, 4)
    assert first["iqr"] == 4 and first["mad"] == 2
    assert first["iqr_flag_count"] == first["mad_flag_count"] == 1
    assert first["iqr_evaluable_count"] == first["mad_evaluable_count"] == 5
    assert first["anomaly_rows"] == 1
    assert second["iqr"] == second["mad"] == 0
    assert not second["iqr_available"] and not second["mad_available"]
    assert second["iqr_flag_count"] is None and second["mad_flag_count"] is None
    assert second["iqr_evaluable_count"] == second["mad_evaluable_count"] == 0

    normal, anomaly = result["meter_by_class"].to_dicts()
    assert normal["anomaly"] == 0 and normal["rows"] == 6
    assert normal["q01"] == pytest.approx(-1.9)
    assert normal["median"] == 3 and normal["q99"] == 10
    assert anomaly["anomaly"] == 1 and anomaly["rows"] == 2
    assert anomaly["median"] == 55 and anomaly["q01"] == pytest.approx(10.9)
    assert anomaly["q99"] == pytest.approx(99.1)

    january, february = result["meter_by_month"].to_dicts()
    assert result["meter_by_month"].schema["month"] == pl.Date
    assert january["rows"] == 4 and january["iqr_flag_count"] == 0
    assert january["negative_count"] == january["zero_count"] == 1
    assert february["rows"] == 4 and february["anomaly_rows"] == 2
    assert february["iqr_flag_count"] == february["mad_flag_count"] == 1
    assert february["iqr_evaluable_count"] == february["mad_evaluable_count"] == 1
    assert result["scope"].row(0, named=True) == {
        "development_rows": 8, "development_buildings": 2,
        "sample_rows": 8, "vault_buildings_excluded": 1,
    }


def test_vault_values_never_influence_profiles_flags_validation_or_sample():
    original = _frame()
    poisoned = original.with_columns(
        pl.when(pl.col("building_id") == 9).then(float("inf"))
        .otherwise(pl.col("meter_reading")).alias("meter_reading"),
        pl.when(pl.col("building_id") == 9).then(99)
        .otherwise(pl.col("anomaly")).alias("anomaly"),
        pl.when(pl.col("building_id") == 9).then(None)
        .otherwise(pl.col("timestamp")).alias("timestamp"),
    )
    expected = _audit(original.lazy(), _manifest(), sample_limit=4)
    actual = _audit(poisoned.lazy(), _manifest(), sample_limit=4)
    for key in expected:
        assert_frame_equal(actual[key], expected[key])
    assert 9 not in actual["plot_sample"]["building_id"].to_list()


def test_unsafe_features_are_excluded_and_sources_are_unchanged():
    frame = _frame().with_columns(
        pl.lit(1).alias("row_id"), pl.lit(float("inf")).alias("gte_unsafe"),
        pl.lit(-5.0).alias("temperature"),
    )
    snapshot, manifest = frame.clone(), _manifest()
    manifest_snapshot = manifest.clone()
    lazy = frame.lazy()
    plan = lazy.explain()
    result = _audit(lazy, manifest, [
        "building_id", "row_id", "anomaly", "timestamp", "gte_unsafe",
        "meter_reading", "temperature",
    ])
    assert result["numeric_profile"]["feature"].to_list() == [
        "meter_reading", "temperature",
    ]
    assert result["numeric_profile"].row(1, named=True)["negative_count"] == 8
    assert_frame_equal(frame, snapshot)
    assert_frame_equal(lazy.collect(), snapshot)
    assert_frame_equal(manifest, manifest_snapshot)
    assert lazy.explain() == plan


def test_sampling_is_bounded_stratified_reorder_invariant_and_metrics_are_full_dev():
    frame = pl.DataFrame({
        "building_id": [1] * 101,
        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(101)],
        "meter_reading": [float(i) for i in range(101)],
        "anomaly": [0] * 100 + [1],
    })
    small = _audit(frame.lazy(), _manifest(), sample_limit=7)
    reordered = _audit(frame.reverse().lazy(), _manifest(), sample_limit=7)
    large = _audit(frame.lazy(), _manifest(), sample_limit=101)
    assert small["plot_sample"].height == 7
    assert small["plot_sample"].columns == [
        "building_id", "timestamp", "meter_reading", "anomaly",
    ]
    assert set(small["plot_sample"]["anomaly"]) == {0, 1}
    assert small["plot_sample"].filter(pl.col("anomaly") == 1).height == 1
    assert_frame_equal(small["plot_sample"], reordered["plot_sample"])
    for key in ("numeric_profile", "meter_by_class", "meter_by_building", "meter_by_month"):
        assert_frame_equal(small[key], large[key])
    assert small["scope"].item(0, "development_rows") == 101


def test_nonfinite_counts_and_all_missing_groups_remain_explicit():
    values = [None, float("nan"), float("inf"), float("-inf"), 0.0, -3.0]
    frame = pl.DataFrame({
        "building_id": [1] * 6 + [2] * 3,
        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)] * 9,
        "meter_reading": values + [None, float("nan"), float("inf")],
        "anomaly": [0, 0, 1, 1, 0, 1, 0, 1, 0],
        "empty_feature": [None] * 9,
    })
    result = _audit(frame.lazy(), _manifest(), ["meter_reading", "empty_feature"])
    meter, empty = result["numeric_profile"].to_dicts()
    assert (meter["rows"], meter["missing_count"], meter["nan_count"],
            meter["nonfinite_count"], meter["finite_count"]) == (9, 2, 2, 5, 2)
    assert meter["median"] == -1.5 and meter["mad"] == 1.5
    assert empty["missing_count"] == 9 and empty["finite_count"] == 0
    for field in ("min", "q001", "q01", "q25", "median", "q75", "q99", "q999", "max", "iqr", "mad"):
        assert empty[field] is None
    unavailable = result["meter_by_building"].row(1, named=True)
    assert unavailable["rows"] == 3 and unavailable["finite_count"] == 0
    assert unavailable["iqr_flag_count"] is None and unavailable["mad_flag_count"] is None
    assert not unavailable["iqr_available"] and not unavailable["mad_available"]
    assert result["plot_sample"].height == 2
    assert result["plot_sample"]["meter_reading"].is_finite().all()


@pytest.mark.parametrize("values", [[None] * 4, [float("nan")] * 4, [float("inf")] * 4, [5.0] * 4, [5.0]])
def test_degenerate_meter_groups_have_no_invented_flags(values):
    frame = pl.DataFrame({
        "building_id": [1] * len(values),
        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)] * len(values),
        "meter_reading": values,
        "anomaly": [0] * len(values),
    })
    result = _audit(frame.lazy(), _manifest())
    for key in ("meter_by_building", "meter_by_month"):
        summary = result[key].row(0, named=True)
        assert summary["iqr_flag_count"] is None
        assert summary["mad_flag_count"] is None
        assert summary["iqr_evaluable_count"] == summary["mad_evaluable_count"] == 0
    assert result["plot_sample"].height <= len(values)


@pytest.mark.parametrize("manifest", [
    pl.DataFrame({"building_id": [1, 1], "split": ["dev", "vault"], "fold": [0, None]}),
    pl.DataFrame({"building_id": [None], "split": ["dev"], "fold": [0]}),
    _manifest().drop("split"), _manifest().drop("fold"),
    _manifest().with_columns(pl.lit("unknown").alias("split")),
    _manifest().with_columns(pl.lit(None, dtype=pl.String).alias("split")),
    _manifest().with_columns(pl.lit(-1).alias("fold")),
    _manifest().with_columns(pl.lit(None, dtype=pl.Int64).alias("fold")),
])
def test_malformed_manifest_fails_closed(manifest):
    with pytest.raises((ValueError, TypeError), match="manifest|assignment|fold|building_id"):
        _audit(_frame().lazy(), manifest)


@pytest.mark.parametrize("building", [None, 999])
def test_missing_or_unknown_building_assignment_fails_closed(building):
    frame = _frame().with_columns(pl.lit(building, dtype=pl.Int64).alias("building_id"))
    with pytest.raises(ValueError, match="assignment|building_id"):
        _audit(frame.lazy(), _manifest())


def test_no_development_rows_fails_closed():
    with pytest.raises(ValueError, match="development"):
        _audit(_frame().filter(pl.col("building_id") == 9).lazy(), _manifest())


@pytest.mark.parametrize("label", [None, -1, 2])
def test_bad_development_class_labels_fail_closed(label):
    with pytest.raises(ValueError, match="anomaly|binary"):
        _audit(_frame().with_columns(pl.lit(label, dtype=pl.Int64).alias("anomaly")).lazy(), _manifest())


@pytest.mark.parametrize("column,value", [("anomaly", 1.0), ("anomaly", True), ("timestamp", "2024-01-01"), ("meter_reading", "bad")])
def test_bad_input_column_types_fail_closed(column, value):
    with pytest.raises(TypeError, match=column):
        _audit(_frame().with_columns(pl.lit(value).alias(column)).lazy(), _manifest())


def test_null_development_timestamp_fails_closed():
    with pytest.raises(ValueError, match="timestamp"):
        _audit(_frame().with_columns(pl.lit(None, dtype=pl.Datetime).alias("timestamp")).lazy(), _manifest())


@pytest.mark.parametrize("columns", [["missing"], ["meter_reading", "meter_reading"], ["text"]])
def test_missing_duplicate_or_nonnumeric_profile_requests_fail_closed(columns):
    frame = _frame().with_columns(pl.lit("text").alias("text"))
    with pytest.raises((ValueError, TypeError), match="numeric|feature|column"):
        _audit(frame.lazy(), _manifest(), columns)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_bad_sample_limits_fail_closed(limit):
    with pytest.raises((ValueError, TypeError), match="sample_limit"):
        _audit(_frame().lazy(), _manifest(), sample_limit=limit)


def test_sample_limit_one_cannot_silently_omit_a_present_finite_class():
    with pytest.raises(ValueError, match="sample_limit|classes"):
        _audit(_frame().lazy(), _manifest(), sample_limit=1)


def test_empty_safe_feature_list_has_typed_empty_profile():
    result = _audit(_frame().lazy(), _manifest(), ["building_id", "gte_unrequested"])
    assert result["numeric_profile"].is_empty()
    assert result["numeric_profile"].schema["feature"] == pl.String
    assert result["numeric_profile"].schema["rows"] == pl.Int64
    assert result["numeric_profile"].schema["median"] == pl.Float64


def test_wrong_container_types_fail_closed():
    with pytest.raises(TypeError, match="LazyFrame"):
        _audit(_frame(), _manifest())
    with pytest.raises(TypeError, match="DataFrame"):
        _audit(_frame().lazy(), _manifest().lazy())
    with pytest.raises(TypeError, match="numeric_columns"):
        _audit(_frame().lazy(), _manifest(), "meter_reading")


def test_missing_required_frame_column_fails_closed():
    with pytest.raises(ValueError, match="meter_reading"):
        _audit(_frame().drop("meter_reading").lazy(), _manifest())


@pytest.mark.parametrize("values,iqr_count", [
    ([0.0] * 8 + [100.0], None),
    ([0.0] * 6 + [4.0, 4.0, 100.0], 1),
])
def test_zero_mad_is_unavailable_even_with_a_rare_extreme(values, iqr_count):
    frame = pl.DataFrame({
        "building_id": [1] * len(values),
        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)] * len(values),
        "meter_reading": values,
        "anomaly": [0] * (len(values) - 1) + [1],
    })
    result = _audit(frame.lazy(), _manifest())
    building = result["meter_by_building"].row(0, named=True)
    assert building["mad"] == 0 and not building["mad_available"]
    assert building["mad_flag_count"] is None and building["mad_evaluable_count"] == 0
    assert building["iqr_flag_count"] == iqr_count
    assert result["numeric_profile"].item(0, "max") == 100
    assert result["plot_sample"]["meter_reading"].max() == 100


@pytest.mark.parametrize("delta,count", [(0.0, 0), (0.001, 2)])
def test_iqr_fences_use_strict_comparisons(delta, count):
    values = [-6.0 - delta, 0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 10.0 + delta]
    frame = pl.DataFrame({
        "building_id": [1] * len(values),
        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)] * len(values),
        "meter_reading": values,
        "anomaly": [0] * len(values),
    })
    result = _audit(frame.lazy(), _manifest())
    building = result["meter_by_building"].row(0, named=True)
    assert (building["q25"], building["q75"], building["iqr"]) == (0, 4, 4)
    assert building["iqr_flag_count"] == count
    assert building["mad_flag_count"] == 0


def test_returned_schemas_and_date_input_are_stable():
    frame = _frame().with_columns(pl.col("timestamp").cast(pl.Date))
    result = _audit(frame.lazy(), _manifest())
    assert result["numeric_profile"].columns == [
        "feature", "rows", "missing_count", "nan_count", "nonfinite_count",
        "finite_count", "zero_count", "negative_count", "min", "q001", "q01",
        "q25", "median", "q75", "q99", "q999", "max", "iqr", "mad",
    ]
    assert result["meter_by_building"].columns == [
        "building_id", "rows", "finite_count", "anomaly_rows", "q25", "median",
        "q75", "iqr", "mad", "iqr_available", "mad_available", "iqr_flag_count",
        "mad_flag_count", "iqr_evaluable_count", "mad_evaluable_count",
    ]
    assert result["meter_by_month"].columns == [
        "month", "rows", "finite_count", "anomaly_rows", "iqr_flag_count",
        "mad_flag_count", "zero_count", "negative_count", "iqr_evaluable_count",
        "mad_evaluable_count",
    ]
    for table in result.values():
        for column, dtype in table.schema.items():
            if column.endswith(("_count", "_rows")) or column == "rows":
                assert dtype == pl.Int64, column
            if column in {"iqr", "mad", "min", "max", "median", "q001", "q01", "q25", "q75", "q99", "q999"}:
                assert dtype == pl.Float64, column
    assert result["plot_sample"].schema["timestamp"] == pl.Date
