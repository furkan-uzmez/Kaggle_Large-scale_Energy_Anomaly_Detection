"""Enforce prediction-time and provenance rules for candidate predictors."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from project.features.registry import (
    AvailabilityStatus,
    FeatureMetadata,
    FeatureRegistry,
    ProvenanceStatus,
    TargetDependence,
)

_NON_PREDICTOR_NAMES = {"anomaly", "fold", "label", "target", "y"}


def validate_feature_availability(
    feature_meta: FeatureMetadata,
    prediction_time: datetime,
) -> None:
    """Raise when metadata does not prove a feature is safe at prediction time."""

    if feature_meta.provenance_status is not ProvenanceStatus.VERIFIED:
        raise ValueError(
            f"{feature_meta.name} provenance is {feature_meta.provenance_status.value}"
        )
    if feature_meta.prediction_time_availability is not AvailabilityStatus.AVAILABLE:
        raise ValueError(
            f"{feature_meta.name} is {feature_meta.prediction_time_availability.value} "
            "at prediction time"
        )
    if feature_meta.target_dependence is not TargetDependence.INDEPENDENT:
        raise ValueError(f"{feature_meta.name} is target-dependent and cannot be a predictor")
    if feature_meta.available_at is not None:
        try:
            is_future = feature_meta.available_at > prediction_time
        except TypeError as error:
            raise ValueError(
                f"{feature_meta.name} and prediction_time must use compatible timezones"
            ) from error
        if is_future:
            raise ValueError(f"{feature_meta.name} is available after prediction time")


def select_predictor_columns(
    columns: Sequence[str],
    registry: FeatureRegistry,
    prediction_time: datetime,
) -> tuple[str, ...]:
    """Return registered, available predictors in the input schema's order."""

    if len(columns) != len(set(columns)):
        raise ValueError("columns must not contain duplicates")

    selected: list[str] = []
    for column in columns:
        if not isinstance(column, str) or not column.strip():
            raise ValueError("columns must contain non-empty strings")

        normalized_name = column.casefold()
        if normalized_name.startswith("gte_") and registry.get(column) is None:
            raise ValueError(f"{column} provenance is pending/unverified")
        if _is_reserved_predictor_name(normalized_name):
            continue

        feature_meta = registry.get(column)
        if feature_meta is None:
            continue

        normalized_family = feature_meta.family.casefold().replace("-", "_")
        if normalized_family in {"source_metadata", "identifier", "target"}:
            continue

        validate_feature_availability(feature_meta, prediction_time)
        selected.append(column)

    return tuple(selected)


def _is_reserved_predictor_name(name: str) -> bool:
    return (
        name in _NON_PREDICTOR_NAMES
        or name == "id"
        or name.endswith("_id")
        or name.startswith(("source_", "fold_"))
    )
