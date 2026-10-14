"""Resolve explicitly evidenced supplied raw feature pools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from project.experiments.candidate_contracts import (
    NON_PREDICTORS,
    BlockedExperimentError,
)
from project.features.registry import (
    ABSOLUTE_CALENDAR_YEAR,
    AvailabilityStatus,
    FeatureMetadata,
    FeatureRegistry,
    ProvenanceStatus,
    ResourceCost,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)

_HISTORY_PREFIXES = ("history_", "lag_", "rolling_", "group_", "building_history")
_MAX_EVIDENCE_TEXT = 240

WEATHER_FEATURES: tuple[str, ...] = (
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
)


@dataclass(frozen=True)
class ProvidedSourceEvidence:
    """Bounded immutable evidence for one repository-provided raw feature."""

    source_notebook: str
    source_cell: str
    producer: str

    def __post_init__(self) -> None:
        for label, value in (
            ("source_notebook", self.source_notebook),
            ("source_cell", self.source_cell),
            ("producer", self.producer),
        ):
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{label} must be a nonempty unpadded string")
            if len(value) > _MAX_EVIDENCE_TEXT:
                raise ValueError(f"{label} exceeds the bounded evidence length")
        if not self.source_notebook.endswith(".ipynb"):
            raise ValueError("source_notebook must identify a notebook")


def resolve_provided_pool(
    core: Sequence[str],
    requested: Sequence[str],
    authoritative_registry: FeatureRegistry,
    *,
    year_built_encoding: str | None = None,
) -> tuple[str, ...]:
    """Return an ordered raw pool whose additions have verified source evidence.

    The repository producer definition is the trust boundary. The production
    definition is intentionally empty until a real source producer and bounded
    notebook/cell evidence are verified. Tests may replace that boundary with a
    typed synthetic producer definition; caller metadata alone never grants
    admission.
    """
    _validate_inputs(core, requested, authoritative_registry)
    core_names = tuple(core)
    requested_names = tuple(requested)
    _validate_requested_order(core_names, requested_names)

    if year_built_encoding is None:
        year_meta = authoritative_registry.get("year_built")
        if year_meta is not None and "lead_year_offset_1900_uint8_v1" in year_meta.semantic_meaning:
            year_built_encoding = "lead_year_offset_1900_uint8_v1"
        else:
            year_built_encoding = ABSOLUTE_CALENDAR_YEAR
    repository_core = build_lead_baseline_registry(year_built_encoding=year_built_encoding)
    for name in core_names:
        expected = repository_core.get(name)
        if expected is None:
            raise BlockedExperimentError("unknown_core_feature", f"unknown repository core feature: {name}")
        metadata = authoritative_registry.get(name)
        if metadata is None:
            raise BlockedExperimentError("unknown_core_feature", f"core feature is not authoritative: {name}")
        _validate_core_metadata(metadata)
        if metadata != expected:
            raise BlockedExperimentError("core_metadata_mismatch", f"core metadata differs from repository definition: {name}")

    additional_names = requested_names[len(core_names):]
    if not additional_names:
        raise BlockedExperimentError(
            "no_verified_additional_features",
            "provided pool contains no verified additional raw feature",
        )

    producer_definitions = _validated_producer_definitions()
    for name in additional_names:
        _validate_unsafe_name(name)
        metadata = authoritative_registry.get(name)
        if metadata is None:
            raise BlockedExperimentError("unknown_provided_feature", f"unknown provided feature: {name}")
        _validate_additional_metadata(metadata)
        evidence = producer_definitions.get(name)
        if evidence is None:
            raise BlockedExperimentError("missing_source_evidence", f"source evidence is missing: {name}")
        _validate_evidence_binding(name, metadata, evidence)

    return requested_names


def _repository_producer_definitions() -> Mapping[str, ProvidedSourceEvidence]:
    """Return verified repository producers."""
    return MappingProxyType({
        name: ProvidedSourceEvidence(
            source_notebook="notebooks/01_data_understanding.ipynb",
            source_cell="lead-weather-load",
            producer="weather_physical",
        )
        for name in WEATHER_FEATURES
    })


def build_provided_feature_registry(year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> FeatureRegistry:
    """Return the 16-feature registry: 9 core baseline + 7 verified physical weather measurements."""
    baseline = build_lead_baseline_registry(year_built_encoding=year_built_encoding)
    weather_meta = [
        FeatureMetadata(
            name=name,
            family="weather_physical",
            source_columns=(name,),
            semantic_meaning=f"Hourly physical weather observation for {name}.",
            prediction_time_availability=AvailabilityStatus.AVAILABLE,
            target_dependence=TargetDependence.INDEPENDENT,
            transform_fit_boundary=TransformFitBoundary.OUTER_TRAIN,
            fallback_behavior="Retain native missing values for median imputation.",
            resource_cost=ResourceCost.LOW,
            provenance_status=ProvenanceStatus.VERIFIED,
        )
        for name in WEATHER_FEATURES
    ]
    return FeatureRegistry([*(baseline.get(name) for name in baseline.names), *weather_meta])

def _validated_producer_definitions() -> Mapping[str, ProvidedSourceEvidence]:
    definitions = _repository_producer_definitions()
    if not isinstance(definitions, Mapping):
        raise BlockedExperimentError("invalid_source_evidence", "producer definitions must be a mapping")

    copied: dict[str, ProvidedSourceEvidence] = {}
    for name, evidence in definitions.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(evidence, ProvidedSourceEvidence):
            raise BlockedExperimentError("invalid_source_evidence", "producer definitions contain invalid evidence")
        copied[name] = evidence
    return MappingProxyType(copied)


def _validate_inputs(
    core: Sequence[str], requested: Sequence[str], registry: FeatureRegistry,
) -> None:
    if not isinstance(registry, FeatureRegistry):
        raise TypeError("authoritative_registry must be a FeatureRegistry")
    for label, names in (("core", core), ("requested", requested)):
        if not isinstance(names, tuple):
            raise TypeError(f"{label} must be an immutable tuple of feature names")
        if not names or any(not isinstance(name, str) or name != name.strip() or not name for name in names):
            raise ValueError(f"{label} must contain nonempty feature names")
        if len(names) != len(set(names)):
            raise BlockedExperimentError("duplicate_feature_names", f"{label} contains duplicate feature names")


def _validate_requested_order(core: tuple[str, ...], requested: tuple[str, ...]) -> None:
    if len(requested) < len(core) or requested[:len(core)] != core:
        raise BlockedExperimentError(
            "feature_pool_order_mismatch",
            "requested pool must preserve the explicit core order before additions",
        )


def _validate_core_metadata(metadata: FeatureMetadata) -> None:
    _validate_unsafe_name(metadata.name)
    if metadata.source_columns != (metadata.name,):
        raise BlockedExperimentError("derived_core_feature", f"core feature is not a raw source: {metadata.name}")
    if metadata.provenance_status is not ProvenanceStatus.VERIFIED:
        raise BlockedExperimentError("unverified_provenance", f"core feature provenance is not verified: {metadata.name}")
    if metadata.prediction_time_availability is not AvailabilityStatus.AVAILABLE:
        raise BlockedExperimentError("unavailable_core_feature", f"core feature is unavailable: {metadata.name}")
    if metadata.target_dependence is not TargetDependence.INDEPENDENT:
        raise BlockedExperimentError("target_dependent_feature", f"core feature is target-dependent: {metadata.name}")
    if metadata.available_at is not None:
        raise BlockedExperimentError("unbounded_prediction_time", f"core feature has an unbound availability timestamp: {metadata.name}")


def _validate_additional_metadata(metadata: FeatureMetadata) -> None:
    if metadata.source_columns != (metadata.name,):
        raise BlockedExperimentError("derived_provided_feature", f"provided feature is not a raw source: {metadata.name}")
    if metadata.family.casefold().replace("-", "_") in {"target", "identifier", "source_metadata"}:
        raise BlockedExperimentError("derived_provided_feature", f"derived metadata feature is blocked: {metadata.name}")
    if metadata.provenance_status is not ProvenanceStatus.VERIFIED:
        raise BlockedExperimentError("unverified_provenance", f"provided feature provenance is not verified: {metadata.name}")
    if metadata.prediction_time_availability is not AvailabilityStatus.AVAILABLE:
        raise BlockedExperimentError("unavailable_provided_feature", f"provided feature is unavailable: {metadata.name}")
    if metadata.target_dependence is not TargetDependence.INDEPENDENT:
        raise BlockedExperimentError("target_dependent_feature", f"provided feature is target-dependent: {metadata.name}")
    if metadata.transform_fit_boundary not in (TransformFitBoundary.OUTER_TRAIN, TransformFitBoundary.NONE):
        raise BlockedExperimentError("invalid_fit_boundary", f"provided feature fit boundary is invalid: {metadata.name}")
    if metadata.available_at is not None:
        raise BlockedExperimentError("unbounded_prediction_time", f"provided feature has an unbound availability timestamp: {metadata.name}")


def _validate_evidence_binding(
    name: str,
    metadata: FeatureMetadata,
    evidence: ProvidedSourceEvidence,
) -> None:
    if evidence.producer != metadata.family:
        raise BlockedExperimentError("producer_evidence_mismatch", f"producer evidence does not match: {name}")
    if evidence.source_notebook == "" or evidence.source_cell == "":
        raise BlockedExperimentError("invalid_source_evidence", f"source roots are missing: {name}")


def _validate_unsafe_name(name: str) -> None:
    lowered = name.casefold()
    if lowered in NON_PREDICTORS or lowered.endswith("_id"):
        raise BlockedExperimentError("forbidden_provided_feature", f"target or identifier feature is blocked: {name}")
    if lowered.startswith(("gte_", "source_", "fold_")) or lowered.startswith(_HISTORY_PREFIXES):
        raise BlockedExperimentError("forbidden_provided_feature", f"derived or history feature is blocked: {name}")
