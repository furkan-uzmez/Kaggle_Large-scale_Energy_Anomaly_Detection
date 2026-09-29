"""Fold-local feature metadata, availability, and transformation primitives."""

from project.features.availability import (
    select_predictor_columns,
    validate_feature_availability,
)
from project.features.registry import (
    AvailabilityStatus,
    FeatureMetadata,
    FeatureRegistry,
    ProvenanceStatus,
    ResourceCost,
    TargetDependence,
    TransformFitBoundary,
)
from project.features.transformers import (
    GroupedTimeFeatureSpec,
    TrainFoldFeatureMask,
    add_grouped_shift_rolling_features,
    make_imputation_preprocessor,
    make_native_nan_preprocessor,
)

__all__ = [
    "AvailabilityStatus",
    "FeatureMetadata",
    "FeatureRegistry",
    "GroupedTimeFeatureSpec",
    "ProvenanceStatus",
    "ResourceCost",
    "TargetDependence",
    "TrainFoldFeatureMask",
    "TransformFitBoundary",
    "add_grouped_shift_rolling_features",
    "make_imputation_preprocessor",
    "make_native_nan_preprocessor",
    "select_predictor_columns",
    "validate_feature_availability",
]
