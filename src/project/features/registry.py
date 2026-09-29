"""Typed feature metadata and an insertion-ordered feature registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AvailabilityStatus(str, Enum):
    """Whether a feature value is available at scoring time."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PENDING = "pending"


class TargetDependence(str, Enum):
    """Whether a feature is derived from the prediction target."""

    INDEPENDENT = "independent"
    TARGET_DERIVED = "target_derived"
    UNKNOWN = "unknown"


class TransformFitBoundary(str, Enum):
    """Data boundary allowed when fitting a feature transformation."""

    OUTER_TRAIN = "outer_train"
    FIXED = "fixed"
    NONE = "none"


class ResourceCost(str, Enum):
    """Approximate cost of computing a feature family."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProvenanceStatus(str, Enum):
    """Whether a feature's source and lineage have been verified."""

    VERIFIED = "verified"
    PENDING = "pending"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class FeatureMetadata:
    """Describe a feature's lineage, availability, and transformation contract."""

    name: str
    family: str
    source_columns: tuple[str, ...]
    semantic_meaning: str
    prediction_time_availability: AvailabilityStatus
    target_dependence: TargetDependence
    transform_fit_boundary: TransformFitBoundary
    fallback_behavior: str
    resource_cost: ResourceCost
    provenance_status: ProvenanceStatus
    available_at: datetime | None = None

    def __post_init__(self) -> None:
        _validate_nonempty_text("name", self.name)
        _validate_nonempty_text("family", self.family)
        _validate_nonempty_text("semantic_meaning", self.semantic_meaning)
        _validate_nonempty_text("fallback_behavior", self.fallback_behavior)

        if not self.source_columns:
            raise ValueError("source_columns must contain at least one column")
        if not isinstance(self.source_columns, tuple):
            raise TypeError("source_columns must be a tuple of strings")
        if any(not isinstance(column, str) for column in self.source_columns):
            raise TypeError("source_columns must contain strings")
        if any(not column.strip() for column in self.source_columns):
            raise ValueError("source_columns must contain non-empty strings")
        if len(set(self.source_columns)) != len(self.source_columns):
            raise ValueError("source_columns must not contain duplicates")

        _validate_enum("prediction_time_availability", self.prediction_time_availability, AvailabilityStatus)
        _validate_enum("target_dependence", self.target_dependence, TargetDependence)
        _validate_enum("transform_fit_boundary", self.transform_fit_boundary, TransformFitBoundary)
        _validate_enum("resource_cost", self.resource_cost, ResourceCost)
        _validate_enum("provenance_status", self.provenance_status, ProvenanceStatus)
        if self.available_at is not None and not isinstance(self.available_at, datetime):
            raise ValueError("available_at must be a datetime or None")


class FeatureRegistry:
    """Store validated feature definitions, preserving their registration order."""

    def __init__(self, features: Iterable[FeatureMetadata] = ()) -> None:
        definitions: dict[str, FeatureMetadata] = {}
        for feature in features:
            if not isinstance(feature, FeatureMetadata):
                raise TypeError("features must contain FeatureMetadata instances")
            if feature.name in definitions:
                raise ValueError(f"duplicate feature name: {feature.name}")
            definitions[feature.name] = feature
        self._features = definitions

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in insertion order."""

        return tuple(self._features)

    def get(self, name: str) -> FeatureMetadata | None:
        """Return one feature definition, or ``None`` if it is unregistered."""

        return self._features.get(name)


def _validate_nonempty_text(field_name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_enum(field_name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        if isinstance(value, str):
            try:
                enum_type(value)
            except ValueError as error:
                raise ValueError(
                    f"{field_name} must be a {enum_type.__name__} value"
                ) from error
        raise TypeError(f"{field_name} must be a {enum_type.__name__} value")
