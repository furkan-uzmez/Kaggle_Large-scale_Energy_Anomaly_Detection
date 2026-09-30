from __future__ import annotations

import copy
import importlib
import importlib.util
import pathlib
import warnings
from datetime import UTC, datetime, timedelta
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml


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


# ---------------------------------------------------------------------------
# Task 6 – baseline feature registry
# ---------------------------------------------------------------------------


def test_build_lead_baseline_registry_returns_approved_nine_predictors():
    """build_lead_baseline_registry() returns exactly the nine approved predictors."""
    registry_spec = importlib.util.find_spec("project.features.registry")
    assert registry_spec is not None, "project.features.registry must exist"

    registry_module = importlib.import_module("project.features.registry")

    # build_lead_baseline_registry may not exist yet – that is the red signal
    build_fn = getattr(registry_module, "build_lead_baseline_registry", None)
    assert build_fn is not None, (
        "build_lead_baseline_registry must be defined in project.features.registry"
    )

    registry = build_fn()
    assert isinstance(registry, registry_module.FeatureRegistry)

    expected_order = (
        "meter_reading",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
        "hour",
        "weekday",
        "month",
        "year",
    )
    assert registry.names == expected_order, (
        f"Expected predictor order {expected_order}, got {registry.names}"
    )

    # All must be VERIFIED / AVAILABLE / INDEPENDENT
    for name in expected_order:
        meta = registry.get(name)
        assert meta is not None, f"{name} must be registered"
        assert meta.provenance_status is registry_module.ProvenanceStatus.VERIFIED, (
            f"{name} provenance must be VERIFIED"
        )
        assert (
            meta.prediction_time_availability is registry_module.AvailabilityStatus.AVAILABLE
        ), f"{name} must be AVAILABLE at prediction time"
        assert meta.target_dependence is registry_module.TargetDependence.INDEPENDENT, (
            f"{name} must be INDEPENDENT of the target"
        )

    # Only primary_use is categorical (family == "categorical")
    primary_use_meta = registry.get("primary_use")
    assert primary_use_meta is not None
    assert primary_use_meta.family.casefold() == "categorical", (
        "primary_use must have family 'categorical'"
    )

    numeric_names = [n for n in expected_order if n != "primary_use"]
    for name in numeric_names:
        meta = registry.get(name)
        assert meta is not None
        assert meta.family.casefold() != "categorical", (
            f"{name} must not be categorical"
        )


def test_baseline_registry_select_predictor_columns_rejects_excluded_schema_columns():
    """select_predictor_columns rejects identifiers, target, and unapproved names."""
    registry_module = importlib.import_module("project.features.registry")
    availability_module = importlib.import_module("project.features.availability")

    build_fn = getattr(registry_module, "build_lead_baseline_registry", None)
    assert build_fn is not None, (
        "build_lead_baseline_registry must be defined in project.features.registry"
    )
    registry = build_fn()

    from datetime import UTC, datetime

    prediction_time = datetime(2022, 1, 2, 0, tzinfo=UTC)

    # Full realistic schema including target/IDs/fold/weather/lag/group-aggregate/alias columns
    full_schema = [
        # Approved predictors
        "meter_reading",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
        "hour",
        "weekday",
        "month",
        "year",
        # Identifiers and target – must be excluded
        "row_id",
        "building_id",
        "site_id",
        "anomaly",
        "fold",
        # Unapproved: raw timestamp
        "timestamp",
        # Unapproved: weather
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
        # Unapproved: is_holiday
        "is_holiday",
        # Unapproved: lag/rolling
        "meter_reading_lag_1",
        "meter_reading_rolling_mean_24",
        # Unapproved: group aggregates
        "building_meter_mean",
        "building_meter_std",
        # Unapproved: cyclical aliases
        "hour_sin",
        "hour_cos",
        # Source metadata
        "source_file",
        "source_label",
    ]

    selected = availability_module.select_predictor_columns(
        full_schema, registry, prediction_time
    )

    approved = (
        "meter_reading",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
        "hour",
        "weekday",
        "month",
        "year",
    )
    assert selected == approved, (
        f"Only the nine approved predictors should be selected, got {selected}"
    )


def test_baseline_registry_unknown_gte_fails_closed():
    """select_predictor_columns must fail closed for unknown gte_* names."""
    registry_module = importlib.import_module("project.features.registry")
    availability_module = importlib.import_module("project.features.availability")

    build_fn = getattr(registry_module, "build_lead_baseline_registry", None)
    assert build_fn is not None, (
        "build_lead_baseline_registry must be defined in project.features.registry"
    )
    registry = build_fn()

    from datetime import UTC, datetime

    with pytest.raises(ValueError, match=r"gte_vendor_x.*provenance.*pending|unverified"):
        availability_module.select_predictor_columns(
            ["gte_vendor_x"],
            registry,
            datetime(2022, 1, 1, tzinfo=UTC),
        )


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


# ---------------------------------------------------------------------------
# Task 6 – baseline estimator factory
# ---------------------------------------------------------------------------


def _load_baseline_config() -> dict[str, Any]:
    config_path = pathlib.Path(__file__).parents[2] / "configs" / "experiments" / "baselines.yaml"
    assert config_path.exists(), (
        f"configs/experiments/baselines.yaml must exist, not found at {config_path}"
    )
    with config_path.open() as fh:
        config = yaml.safe_load(fh)
    assert isinstance(config, dict), "baselines.yaml must parse as a mapping"
    return config


def _extract_estimator(est: Any, cls: type) -> Any:
    """Return the underlying estimator, unwrapping a Pipeline if needed."""
    from sklearn.pipeline import Pipeline

    if isinstance(est, Pipeline):
        for _, step in est.steps:
            if isinstance(step, cls):
                return step
        return est.steps[-1][1]
    return est


def test_build_baseline_estimators_returns_expected_keys_and_types():
    """build_baseline_estimators(config) returns exactly the three required estimators."""
    mod_spec = importlib.util.find_spec("project.models.baselines")
    assert mod_spec is not None, (
        "project.models.baselines must be created for Task 6"
    )

    mod = importlib.import_module("project.models.baselines")
    fn = getattr(mod, "build_baseline_estimators", None)
    assert fn is not None, (
        "build_baseline_estimators must be defined in project.models.baselines"
    )

    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    config = _load_baseline_config()
    estimators = fn(config)

    assert isinstance(estimators, dict), "build_baseline_estimators must return a dict"
    assert set(estimators.keys()) == {"dummy_prior", "logistic_l2", "hist_gradient_boosting"}, (
        f"Expected keys dummy_prior/logistic_l2/hist_gradient_boosting, got {set(estimators.keys())}"
    )

    # Dummy: strategy must be "prior"
    dummy = estimators["dummy_prior"]
    dummy_core = _extract_estimator(dummy, DummyClassifier)
    assert isinstance(dummy_core, DummyClassifier), (
        "dummy_prior must ultimately be a DummyClassifier"
    )
    assert dummy_core.strategy == "prior", (
        f"DummyClassifier strategy must be 'prior', got {dummy_core.strategy!r}"
    )
    assert dummy_core.random_state == 42, (
        f"random_state must be 42, got {dummy_core.random_state}"
    )

    # Logistic regression (assert l1_ratio=0.0 for L2, not deprecated penalty='l2')
    lr = estimators["logistic_l2"]
    lr_core = _extract_estimator(lr, LogisticRegression)
    assert isinstance(lr_core, LogisticRegression), (
        "logistic_l2 must ultimately be a LogisticRegression"
    )
    assert lr_core.l1_ratio == 0.0, (
        f"l1_ratio must be 0.0 for L2 regularization in sklearn 1.9+, got {lr_core.l1_ratio!r}"
    )
    assert lr_core.C == pytest.approx(1.0), f"C must be 1.0, got {lr_core.C}"
    assert lr_core.class_weight is None, (
        f"class_weight must be None, got {lr_core.class_weight!r}"
    )
    assert lr_core.max_iter == 1000, f"max_iter must be 1000, got {lr_core.max_iter}"
    assert lr_core.random_state == 42, f"random_state must be 42, got {lr_core.random_state}"

    # HistGradientBoostingClassifier
    hgb = estimators["hist_gradient_boosting"]
    hgb_core = _extract_estimator(hgb, HistGradientBoostingClassifier)
    assert isinstance(hgb_core, HistGradientBoostingClassifier), (
        "hist_gradient_boosting must ultimately be a HistGradientBoostingClassifier"
    )
    assert hgb_core.early_stopping is False, (
        f"early_stopping must be False, got {hgb_core.early_stopping!r}"
    )
    assert hgb_core.random_state == 42, (
        f"random_state must be 42, got {hgb_core.random_state}"
    )


def test_build_baseline_estimators_fit_and_predict_on_tiny_dataframe_with_missings():
    """All three baseline estimators fit and produce valid [0,1] probabilities without FutureWarning."""
    mod_spec = importlib.util.find_spec("project.models.baselines")
    assert mod_spec is not None, (
        "project.models.baselines must be created for Task 6"
    )

    mod = importlib.import_module("project.models.baselines")
    fn = getattr(mod, "build_baseline_estimators", None)
    assert fn is not None, (
        "build_baseline_estimators must be defined in project.models.baselines"
    )

    config = _load_baseline_config()
    numeric_columns = list(config["features"]["numeric"])
    categorical_columns = list(config["features"]["categorical"])
    all_feature_columns = numeric_columns + categorical_columns

    estimators = fn(config)

    # Build a tiny DataFrame with missing numeric and a missing/unknown categorical value
    train = pd.DataFrame(
        {
            "meter_reading": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "square_feet":   [1000.0, 2000.0, np.nan, 4000.0, 5000.0, 6000.0],
            "year_built":    [1990, 2000, 2010, 2020, 2015, 2005],
            "floor_count":   [1, 2, 3, 4, np.nan, 6],
            "hour":          [8, 12, 16, 20, 6, 22],
            "weekday":       [0, 1, 2, 3, 4, 5],
            "month":         [1, 3, 6, 9, 11, 12],
            "year":          [2022, 2022, 2022, 2022, 2022, 2022],
            "primary_use":   ["Office", "Retail", None, "Education", "Office", "Retail"],
            "anomaly":       [0, 1, 0, 1, 0, 1],
        }
    )
    X_train = train[all_feature_columns]
    y_train = train["anomaly"].to_numpy()

    val = pd.DataFrame(
        {
            "meter_reading": [2.0, np.nan],
            "square_feet":   [np.nan, 3000.0],
            "year_built":    [1985, 2018],
            "floor_count":   [2, np.nan],
            "hour":          [10, 14],
            "weekday":       [2, 4],
            "month":         [4, 8],
            "year":          [2022, 2022],
            "primary_use":   ["UNKNOWN_CATEGORY", None],  # unseen + missing
        }
    )
    X_val = val[all_feature_columns]

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        for name, estimator in estimators.items():
            estimator.fit(X_train, y_train)
            probas = estimator.predict_proba(X_val)
            assert probas.shape == (len(X_val), 2), (
                f"{name}: predict_proba must return shape ({len(X_val)}, 2)"
            )
            assert np.isfinite(probas).all(), f"{name}: probabilities must be finite"
            assert (probas >= 0).all() and (probas <= 1).all(), (
                f"{name}: probabilities must be in [0, 1]"
            )
            assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6), (
                f"{name}: probability rows must sum to 1"
            )

        future_warnings = [
            w for w in recorded_warnings if issubclass(w.category, FutureWarning)
        ]
        assert len(future_warnings) == 0, (
            f"Expected no FutureWarning during estimator fit/predict, got: "
            f"{[str(w.message) for w in future_warnings]}"
        )


def test_build_baseline_estimators_reflects_modified_config_hyperparameters():
    """Modifying LogisticRegression C and top-level seed in config propagates to estimators."""
    mod_spec = importlib.util.find_spec("project.models.baselines")
    assert mod_spec is not None, "project.models.baselines must exist"

    mod = importlib.import_module("project.models.baselines")
    fn = getattr(mod, "build_baseline_estimators", None)
    assert fn is not None, "build_baseline_estimators must be defined"

    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    config = copy.deepcopy(_load_baseline_config())
    config["seed"] = 123
    config["models"]["logistic_l2"]["params"]["C"] = 0.25

    estimators = fn(config)

    lr_core = _extract_estimator(estimators["logistic_l2"], LogisticRegression)
    assert lr_core.C == pytest.approx(0.25), f"Expected C=0.25, got {lr_core.C}"
    assert lr_core.random_state == 123, f"Expected random_state=123, got {lr_core.random_state}"

    dummy_core = _extract_estimator(estimators["dummy_prior"], DummyClassifier)
    assert dummy_core.random_state == 123, f"Expected random_state=123, got {dummy_core.random_state}"

    hgb_core = _extract_estimator(estimators["hist_gradient_boosting"], HistGradientBoostingClassifier)
    assert hgb_core.random_state == 123, f"Expected random_state=123, got {hgb_core.random_state}"


def test_build_baseline_estimators_config_matches_baselines_yaml():
    """configs/experiments/baselines.yaml exists and matches the approved design."""
    config = _load_baseline_config()

    # Seed
    assert config.get("seed") == 42, f"seed must be 42, got {config.get('seed')!r}"

    # Folds
    assert config.get("folds") == 4, f"folds must be 4, got {config.get('folds')!r}"
    assert config.get("split_manifest") == "outputs/splits/building_folds.parquet"

    # One repeat total; saved OOF repeat indices are zero-based.
    assert config.get("repeats") == 1, (
        f"repeats must be 1, got {config.get('repeats')!r}"
    )
    assert "repeat" not in config, "repeat index belongs in OOF rows, not run config"

    # Target
    assert config.get("target") == "anomaly", (
        f"target must be 'anomaly', got {config.get('target')!r}"
    )

    # Positive label
    assert config.get("positive_label") == 1, (
        f"positive_label must be 1, got {config.get('positive_label')!r}"
    )

    # Primary metric
    assert config.get("primary_metric") == "roc_auc", (
        f"primary_metric must be 'roc_auc', got {config.get('primary_metric')!r}"
    )

    # Require single canonical nested features.numeric and features.categorical list
    assert "features" in config, "features mapping must exist in config"
    features_block = config["features"]
    assert isinstance(features_block, dict), "features must be a mapping"
    assert "numeric" in features_block, "features.numeric must be defined"
    assert "categorical" in features_block, "features.categorical must be defined"
    assert "numeric_features" not in config, "avoid duplicate numeric feature declarations"
    assert "categorical_features" not in config, "avoid duplicate categorical feature declarations"
    assert "positive_class" not in config, "positive_label is the canonical target setting"

    expected_numeric = [
        "meter_reading",
        "square_feet",
        "year_built",
        "floor_count",
        "hour",
        "weekday",
        "month",
        "year",
    ]
    expected_categorical = ["primary_use"]
    assert features_block["numeric"] == expected_numeric, (
        f"features.numeric mismatch: expected {expected_numeric}, got {features_block['numeric']}"
    )
    assert features_block["categorical"] == expected_categorical, (
        f"features.categorical mismatch: expected {expected_categorical}, got {features_block['categorical']}"
    )

    # Exactly three candidates in models mapping
    models = config.get("models")
    assert isinstance(models, dict), "models must be a mapping"
    assert len(models) == 3, f"Expected exactly three candidate models, got {len(models)}"
    expected_models = {"dummy_prior", "logistic_l2", "hist_gradient_boosting"}
    assert set(models.keys()) == expected_models, (
        f"Expected models {expected_models}, got {set(models.keys())}"
    )
    assert all("random_state" not in model["params"] for model in models.values()), (
        "random_state must be governed only by the top-level seed"
    )

    # Check model hyperparameters
    assert models["dummy_prior"]["params"]["strategy"] == "prior"

    assert models["logistic_l2"]["params"]["C"] == 1.0
    assert models["logistic_l2"]["params"]["l1_ratio"] == 0.0
    assert models["logistic_l2"]["params"]["class_weight"] is None
    assert models["logistic_l2"]["params"]["max_iter"] == 1000

    assert models["hist_gradient_boosting"]["params"]["early_stopping"] is False
