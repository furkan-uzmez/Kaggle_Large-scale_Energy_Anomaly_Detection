from __future__ import annotations

import importlib
import importlib.util
from datetime import UTC, datetime, timedelta
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


def _feature_module(module_name: str) -> ModuleType:
    package_spec = importlib.util.find_spec("project.features")
    assert package_spec is not None, "project.features is required for feature APIs"

    module_path = f"project.features.{module_name}"
    module_spec = importlib.util.find_spec(module_path)
    assert module_spec is not None, f"{module_path} is required for feature APIs"
    return importlib.import_module(module_path)


def _feature_metadata(registry: ModuleType, **overrides: object):
    metadata = {
        "name": "meter_reading",
        "family": "measurement",
        "source_columns": ("meter_reading",),
        "semantic_meaning": "Observed meter reading at the scoring timestamp.",
        "prediction_time_availability": registry.AvailabilityStatus.AVAILABLE,
        "target_dependence": registry.TargetDependence.INDEPENDENT,
        "transform_fit_boundary": registry.TransformFitBoundary.OUTER_TRAIN,
        "fallback_behavior": "Retain native missing values.",
        "resource_cost": registry.ResourceCost.LOW,
        "provenance_status": registry.ProvenanceStatus.VERIFIED,
    }
    metadata.update(overrides)
    return registry.FeatureMetadata(**metadata)


def test_feature_registry_validates_metadata_and_rejects_duplicate_names():
    registry_module = _feature_module("registry")
    definition = _feature_metadata(registry_module)

    registry = registry_module.FeatureRegistry([definition])

    assert registry.names == ("meter_reading",)
    assert definition.source_columns == ("meter_reading",)
    assert definition.semantic_meaning
    assert definition.transform_fit_boundary is registry_module.TransformFitBoundary.OUTER_TRAIN

    with pytest.raises(ValueError, match="duplicate.*meter_reading"):
        registry_module.FeatureRegistry([definition, definition])

    with pytest.raises(ValueError, match="source_columns"):
        _feature_metadata(registry_module, source_columns=())

    with pytest.raises(ValueError, match="resource_cost"):
        _feature_metadata(registry_module, resource_cost="unbounded")


def test_feature_selection_excludes_identifiers_target_fold_and_source_metadata():
    registry_module = _feature_module("registry")
    availability_module = _feature_module("availability")
    prediction_time = datetime(2026, 1, 1, 12, tzinfo=UTC)
    registry = registry_module.FeatureRegistry(
        [
            _feature_metadata(registry_module),
            _feature_metadata(
                registry_module,
                name="source_label",
                family="source_metadata",
                source_columns=("source_label",),
                semantic_meaning="Source lineage label.",
            ),
        ]
    )

    selected = availability_module.select_predictor_columns(
        [
            "row_id",
            "building_id",
            "meter_reading",
            "anomaly",
            "fold",
            "source_file",
            "source_label",
        ],
        registry,
        prediction_time,
    )

    assert selected == ("meter_reading",)


def test_feature_selection_keeps_registered_features_with_target_or_label_substrings():
    registry_module = _feature_module("registry")
    availability_module = _feature_module("availability")
    prediction_time = datetime(2026, 1, 1, 12, tzinfo=UTC)
    registry = registry_module.FeatureRegistry(
        [
            _feature_metadata(
                registry_module,
                name="target_temperature",
                source_columns=("temperature",),
                semantic_meaning="Temperature observed at scoring time.",
            ),
            _feature_metadata(
                registry_module,
                name="label_quality",
                source_columns=("quality_score",),
                semantic_meaning="Quality score available at scoring time.",
            ),
            _feature_metadata(
                registry_module,
                name="target",
                source_columns=("target",),
                semantic_meaning="Reserved target name.",
            ),
            _feature_metadata(
                registry_module,
                name="label",
                source_columns=("label",),
                semantic_meaning="Reserved label name.",
            ),
            _feature_metadata(
                registry_module,
                name="source_temperature",
                source_columns=("source_temperature",),
                semantic_meaning="Source metadata prefix.",
            ),
        ]
    )

    selected = availability_module.select_predictor_columns(
        [
            "target_temperature",
            "label_quality",
            "target",
            "label",
            "source_temperature",
        ],
        registry,
        prediction_time,
    )

    assert selected == ("target_temperature", "label_quality")


def test_unknown_gte_feature_is_rejected_as_unverified_provenance():
    registry_module = _feature_module("registry")
    availability_module = _feature_module("availability")

    with pytest.raises(ValueError, match=r"gte_vendor_feature.*provenance.*pending|unverified"):
        availability_module.select_predictor_columns(
            ["gte_vendor_feature"],
            registry_module.FeatureRegistry(),
            datetime(2026, 1, 1, 12, tzinfo=UTC),
        )


def test_registered_gte_feature_stays_blocked_until_provenance_is_verified():
    registry_module = _feature_module("registry")
    availability_module = _feature_module("availability")
    definition = _feature_metadata(
        registry_module,
        name="gte_unknown_family",
        family="vendor_derived",
        provenance_status=registry_module.ProvenanceStatus.PENDING,
    )
    registry = registry_module.FeatureRegistry([definition])

    with pytest.raises(ValueError, match=r"gte_unknown_family.*provenance.*pending"):
        availability_module.select_predictor_columns(
            ["gte_unknown_family"], registry, datetime(2026, 1, 1, 12, tzinfo=UTC)
        )


def test_feature_availability_rejects_pending_unavailable_and_future_features():
    registry_module = _feature_module("registry")
    availability_module = _feature_module("availability")
    prediction_time = datetime(2026, 1, 1, 12, tzinfo=UTC)
    verified = _feature_metadata(registry_module)

    availability_module.validate_feature_availability(verified, prediction_time)

    pending_provenance = _feature_metadata(
        registry_module,
        provenance_status=registry_module.ProvenanceStatus.PENDING,
    )
    with pytest.raises(ValueError, match=r"meter_reading.*provenance.*pending"):
        availability_module.validate_feature_availability(
            pending_provenance, prediction_time
        )

    unavailable = _feature_metadata(
        registry_module,
        name="late_weather",
        prediction_time_availability=registry_module.AvailabilityStatus.UNAVAILABLE,
    )
    with pytest.raises(ValueError, match=r"late_weather.*unavailable.*prediction time"):
        availability_module.validate_feature_availability(unavailable, prediction_time)

    future = _feature_metadata(
        registry_module,
        name="future_window",
        available_at=prediction_time + timedelta(hours=1),
    )
    with pytest.raises(ValueError, match=r"future_window.*after prediction time"):
        availability_module.validate_feature_availability(future, prediction_time)

    available_at_prediction = _feature_metadata(
        registry_module,
        available_at=prediction_time,
    )
    availability_module.validate_feature_availability(
        available_at_prediction, prediction_time
    )


def test_train_fitted_mask_drops_constants_flags_quasi_constants_and_keeps_order():
    transformers = _feature_module("transformers")
    train = pd.DataFrame(
        {
            "constant": [7, 7, 7, 7],
            "quasi_constant": [0, 0, 0, 1],
            "variable": [1, 2, 3, 4],
        }
    )
    validation = pd.DataFrame(
        {
            "constant": [8, 9],
            "quasi_constant": [5, 5],
            "variable": [6, 7],
        }
    )
    selector = transformers.TrainFoldFeatureMask(quasi_constant_threshold=0.75)

    transformed_train = selector.fit_transform(train)
    transformed_validation = selector.transform(validation)

    assert transformed_train.columns.tolist() == ["quasi_constant", "variable"]
    assert transformed_validation.columns.tolist() == ["quasi_constant", "variable"]
    pd.testing.assert_frame_equal(
        transformed_validation,
        validation[["quasi_constant", "variable"]],
    )
    assert selector.constant_features_ == ("constant",)
    assert selector.quasi_constant_features_ == ("quasi_constant",)


def test_native_nan_preprocessor_preserves_missing_values_and_transforms_validation():
    transformers = _feature_module("transformers")
    train = pd.DataFrame({"signal": [1.0, np.nan, 3.0], "constant": [4, 4, 4]})
    validation = pd.DataFrame({"signal": [np.nan, 5.0], "constant": [4, 4]})
    preprocessor = transformers.make_native_nan_preprocessor()

    transformed_train = preprocessor.fit_transform(train)
    transformed_validation = preprocessor.transform(validation)

    assert transformed_train.columns.tolist() == ["signal"]
    assert transformed_validation.columns.tolist() == ["signal"]
    assert transformed_train["signal"].isna().sum() == 1
    assert transformed_validation["signal"].isna().sum() == 1


@pytest.mark.parametrize(
    "inference_frame",
    [
        pd.DataFrame({"second": [3.0], "first": [4.0]}),
        pd.DataFrame({"first": [4.0], "extra": [5.0]}),
    ],
    ids=["column-order-changed", "columns-changed"],
)
def test_native_preprocessor_rejects_inference_schema_mismatch(inference_frame):
    transformers = _feature_module("transformers")
    preprocessor = transformers.make_native_nan_preprocessor()
    preprocessor.fit(pd.DataFrame({"first": [1.0, 2.0], "second": [2.0, 3.0]}))

    with pytest.raises(ValueError, match="schema.*fit|columns.*fit"):
        preprocessor.transform(inference_frame)


def test_imputation_and_indicators_fit_train_and_ignore_validation_only_patterns():
    transformers = _feature_module("transformers")
    train = pd.DataFrame(
        {
            "reading": [1.0, np.nan, 5.0, 3.0],
            "complete_in_train": [10.0, 11.0, 12.0, 13.0],
            "meter_type": ["A", "B", "A", "B"],
        }
    )
    validation = pd.DataFrame(
        {
            "reading": [100.0, np.nan],
            "complete_in_train": [np.nan, 20.0],
            "meter_type": ["validation_only", None],
        }
    )
    preprocessor = transformers.make_imputation_preprocessor()

    transformed_train = preprocessor.fit_transform(train)
    learned_names = tuple(preprocessor.get_feature_names_out())
    transformed_validation = preprocessor.transform(validation)
    transformed_validation_again = preprocessor.transform(validation)

    assert tuple(preprocessor.get_feature_names_out()) == learned_names
    np.testing.assert_array_equal(transformed_validation, transformed_validation_again)
    np.testing.assert_array_equal(
        preprocessor.transform(train), transformed_train
    )

    feature_names = [name.lower() for name in learned_names]
    reading_index = next(
        index
        for index, name in enumerate(feature_names)
        if "reading" in name and "missingindicator" not in name
    )
    reading_indicator_index = next(
        index
        for index, name in enumerate(feature_names)
        if "reading" in name and "missingindicator" in name
    )
    complete_indicator_index = next(
        index
        for index, name in enumerate(feature_names)
        if "complete_in_train" in name and "missingindicator" in name
    )
    category_indices = [
        index
        for index, name in enumerate(feature_names)
        if "meter_type_" in name and "missingindicator" not in name
    ]

    assert transformed_validation[1, reading_index] == 3.0
    assert transformed_validation[1, reading_indicator_index] == 1.0
    assert transformed_validation[0, complete_indicator_index] == 1.0
    assert category_indices
    np.testing.assert_array_equal(
        transformed_validation[0, category_indices],
        np.zeros(len(category_indices)),
    )


def test_imputation_preprocessor_preserves_sparse_high_cardinality_encoding():
    from scipy import sparse

    transformers = _feature_module("transformers")
    category_count = 64
    train = pd.DataFrame(
        {"meter_type": [f"meter_{index}" for index in range(category_count)]}
    )
    preprocessor = transformers.make_imputation_preprocessor()

    transformed = preprocessor.fit_transform(train)

    assert sparse.issparse(transformed)
    assert transformed.shape == (category_count, category_count)
    assert transformed.nnz == category_count


def test_grouped_lag_and_rolling_features_are_future_invariant_and_never_backfill():
    transformers = _feature_module("transformers")
    frame = pd.DataFrame(
        {
            "row_key": ["A-2", "B-1", "A-0", "A-1", "B-0"],
            "building_id": ["A", "B", "A", "A", "B"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 02:00",
                    "2026-01-01 01:00",
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 00:00",
                ]
            ),
            "meter_reading": [30.0, 200.0, 10.0, 20.0, 100.0],
        },
        index=[11, 12, 13, 14, 15],
    )
    specification = transformers.GroupedTimeFeatureSpec(
        group_columns=("building_id",),
        timestamp_column="timestamp",
        value_column="meter_reading",
        lags=(1, 2),
        rolling_windows=(2,),
    )

    baseline = transformers.add_grouped_shift_rolling_features(frame, specification)
    perturbed_frame = frame.copy()
    prediction_time = pd.Timestamp("2026-01-01 01:00")
    future_rows = (perturbed_frame["building_id"] == "A") & (
        perturbed_frame["timestamp"] > prediction_time
    )
    perturbed_frame.loc[future_rows, "meter_reading"] = 9999.0
    perturbed = transformers.add_grouped_shift_rolling_features(
        perturbed_frame, specification
    )

    baseline_at_time = baseline.loc[baseline["row_key"] == "A-1"].iloc[0]
    perturbed_at_time = perturbed.loc[perturbed["row_key"] == "A-1"].iloc[0]
    assert baseline_at_time["meter_reading_lag_1"] == 10.0
    assert baseline_at_time["meter_reading_rolling_mean_2"] == 10.0
    assert perturbed_at_time["meter_reading_lag_1"] == baseline_at_time[
        "meter_reading_lag_1"
    ]
    assert perturbed_at_time["meter_reading_rolling_mean_2"] == baseline_at_time[
        "meter_reading_rolling_mean_2"
    ]

    first_observation = baseline.loc[baseline["row_key"] == "A-0"].iloc[0]
    assert pd.isna(first_observation["meter_reading_lag_1"])
    assert pd.isna(first_observation["meter_reading_rolling_mean_2"])
    assert baseline["row_key"].tolist() == frame["row_key"].tolist()
    assert baseline.index.equals(frame.index)


def test_grouped_time_features_reject_missing_timestamps():
    transformers = _feature_module("transformers")
    frame = pd.DataFrame(
        {
            "building_id": ["A", "A"],
            "timestamp": pd.to_datetime(["2026-01-01 00:00", None]),
            "meter_reading": [10.0, 20.0],
        }
    )
    specification = transformers.GroupedTimeFeatureSpec(
        group_columns=("building_id",),
        timestamp_column="timestamp",
        value_column="meter_reading",
        lags=(1,),
        rolling_windows=(2,),
    )

    with pytest.raises(ValueError, match="timestamp.*missing"):
        transformers.add_grouped_shift_rolling_features(frame, specification)


def test_grouped_time_features_preserve_missing_readings_and_use_prior_history_only():
    transformers = _feature_module("transformers")
    frame = pd.DataFrame(
        {
            "building_id": ["A", "A", "A", "A"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                    "2026-01-01 03:00",
                ]
            ),
            "meter_reading": [10.0, np.nan, 30.0, 40.0],
        }
    )
    specification = transformers.GroupedTimeFeatureSpec(
        group_columns=("building_id",),
        timestamp_column="timestamp",
        value_column="meter_reading",
        lags=(1,),
        rolling_windows=(2,),
    )

    baseline = transformers.add_grouped_shift_rolling_features(frame, specification)
    perturbed_frame = frame.copy()
    perturbed_frame.loc[perturbed_frame["timestamp"] > pd.Timestamp("2026-01-01 01:00"), "meter_reading"] = 9999.0
    perturbed = transformers.add_grouped_shift_rolling_features(
        perturbed_frame, specification
    )

    missing_time = pd.Timestamp("2026-01-01 01:00")
    after_gap_time = pd.Timestamp("2026-01-01 02:00")
    baseline_at_missing_time = baseline.loc[baseline["timestamp"] == missing_time].iloc[0]
    perturbed_at_missing_time = perturbed.loc[
        perturbed["timestamp"] == missing_time
    ].iloc[0]
    after_gap = baseline.loc[baseline["timestamp"] == after_gap_time].iloc[0]

    assert pd.isna(frame.loc[frame["timestamp"] == missing_time, "meter_reading"]).all()
    assert pd.isna(baseline_at_missing_time["meter_reading"])
    pd.testing.assert_series_equal(
        baseline_at_missing_time[
            ["meter_reading_lag_1", "meter_reading_rolling_mean_2"]
        ],
        perturbed_at_missing_time[
            ["meter_reading_lag_1", "meter_reading_rolling_mean_2"]
        ],
    )
    assert pd.isna(after_gap["meter_reading_lag_1"])
    assert after_gap["meter_reading_rolling_mean_2"] == 10.0


def test_grouped_time_features_reject_duplicate_group_timestamps():
    transformers = _feature_module("transformers")
    timestamp = pd.Timestamp("2026-01-01 00:00")
    frame = pd.DataFrame(
        {
            "building_id": ["A", "A"],
            "timestamp": [timestamp, timestamp],
            "meter_reading": [10.0, 20.0],
        }
    )
    specification = transformers.GroupedTimeFeatureSpec(
        group_columns=("building_id",),
        timestamp_column="timestamp",
        value_column="meter_reading",
        lags=(1,),
        rolling_windows=(2,),
    )

    with pytest.raises(ValueError, match="duplicate.*group.*timestamp"):
        transformers.add_grouped_shift_rolling_features(frame, specification)
