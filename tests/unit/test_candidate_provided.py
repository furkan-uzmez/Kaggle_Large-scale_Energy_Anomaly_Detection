"""TDD contracts for fail-closed supplied-feature pool resolution."""

from dataclasses import replace
from importlib.util import find_spec

import pytest

from project.experiments.candidate_contracts import (
    NON_PREDICTORS,
    BlockedExperimentError,
)
from project.features.registry import (
    AvailabilityStatus,
    FeatureMetadata,
    FeatureRegistry,
    ProvenanceStatus,
    ResourceCost,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)

CORE = build_lead_baseline_registry().names


def _verified_feature(name: str = "temperature_mean") -> FeatureMetadata:
    return FeatureMetadata(
        name=name,
        family="synthetic_fixture",
        source_columns=(name,),
        semantic_meaning="Synthetic source feature available at scoring time.",
        prediction_time_availability=AvailabilityStatus.AVAILABLE,
        target_dependence=TargetDependence.INDEPENDENT,
        transform_fit_boundary=TransformFitBoundary.OUTER_TRAIN,
        fallback_behavior="No fallback; source value is required.",
        resource_cost=ResourceCost.LOW,
        provenance_status=ProvenanceStatus.VERIFIED,
    )


def _resolver():
    from project.features.provided import resolve_provided_pool

    return resolve_provided_pool


def _synthetic_producers():
    from project.features.provided import ProvidedSourceEvidence

    return {
        "temperature_mean": ProvidedSourceEvidence(
            source_notebook="tests/fixtures/provided_sources.ipynb",
            source_cell="synthetic-producer-01",
            producer="synthetic_fixture",
        )
    }


def _registry_with(*features: FeatureMetadata) -> FeatureRegistry:
    baseline = build_lead_baseline_registry()
    return FeatureRegistry([*(baseline.get(name) for name in baseline.names), *features])

def test_provided_resolver_module_is_available():
    """The supplied-feature resolver is a project feature boundary."""
    assert find_spec("project.features.provided") is not None


def test_no_additional_pool_is_blocked_not_a_duplicate_control():
    registry = build_lead_baseline_registry()

    with pytest.raises(BlockedExperimentError) as error:
        _resolver()(registry.names, registry.names, registry)

    assert error.value.code == "no_verified_additional_features"


def test_verified_synthetic_source_is_returned_in_requested_order(monkeypatch):
    registry = _registry_with(_verified_feature())
    monkeypatch.setattr(
        "project.features.provided._repository_producer_definitions",
        _synthetic_producers,
    )

    result = _resolver()(CORE, (*CORE, "temperature_mean"), registry)

    assert result == (*CORE, "temperature_mean")
    assert result == tuple(dict.fromkeys(result))


@pytest.mark.parametrize(
    ("name", "requested"),
    [
        ("unknown", (*CORE, "unknown")),
        ("gte", (*CORE, "gte_vendor_feature")),
        ("id", (*CORE, "building_id")),
        ("target", (*CORE, "anomaly")),
        ("history", (*CORE, "building_history_mean")),
    ],
)
def test_unsafe_requested_names_are_explicitly_blocked(name, requested):
    del name

    with pytest.raises(BlockedExperimentError):
        _resolver()(CORE, requested, build_lead_baseline_registry())


def test_forged_verified_metadata_without_source_evidence_is_blocked():
    registry = _registry_with(_verified_feature())

    with pytest.raises(BlockedExperimentError) as error:
        _resolver()(CORE, (*CORE, "temperature_mean"), registry)

    assert error.value.code == "missing_source_evidence"


def test_production_like_additional_registry_without_repository_producer_is_blocked():
    registry = _registry_with(_verified_feature("temperature_mean"))

    with pytest.raises(BlockedExperimentError) as error:
        _resolver()(CORE, (*CORE, "temperature_mean"), registry)

    assert error.value.code == "missing_source_evidence"


def test_forged_expected_producer_cannot_approve_unregistered_source():
    forged_registry = build_lead_baseline_registry()

    with pytest.raises(BlockedExperimentError) as error:
        _resolver()(CORE, (*CORE, "temperature_mean"), forged_registry)

    assert error.value.code == "unknown_provided_feature"


def test_pending_feature_is_blocked_even_with_matching_evidence(monkeypatch):
    pending = replace(_verified_feature(), provenance_status=ProvenanceStatus.PENDING)
    registry = _registry_with(pending)
    monkeypatch.setattr(
        "project.features.provided._repository_producer_definitions",
        _synthetic_producers,
    )

    with pytest.raises(BlockedExperimentError) as error:
        _resolver()(CORE, (*CORE, "temperature_mean"), registry)

    assert error.value.code == "unverified_provenance"


def test_duplicate_and_core_order_mismatch_are_blocked():
    registry = build_lead_baseline_registry()

    with pytest.raises(BlockedExperimentError, match="duplicate"):
        _resolver()(CORE, ("meter_reading", "meter_reading", *CORE[1:]), registry)

    with pytest.raises(BlockedExperimentError, match="order"):
        _resolver()(CORE, (*CORE[1:], CORE[0]), registry)


def test_synthetic_producer_evidence_is_immutable(monkeypatch):
    from project.features.provided import ProvidedSourceEvidence

    registry = _registry_with(_verified_feature())
    definitions = _synthetic_producers()
    monkeypatch.setattr(
        "project.features.provided._repository_producer_definitions",
        lambda: definitions,
    )
    result = _resolver()(CORE, (*CORE, "temperature_mean"), registry)

    assert result[-1] == "temperature_mean"
    with pytest.raises(AttributeError):
        definitions["temperature_mean"].producer = "forged_after_call"
    assert isinstance(definitions["temperature_mean"], ProvidedSourceEvidence)


@pytest.mark.parametrize("field,value", [
    ("family", "forged"), ("semantic_meaning", "Forged meaning"),
    ("transform_fit_boundary", TransformFitBoundary.NONE),
    ("fallback_behavior", "Forged fallback"), ("resource_cost", ResourceCost.HIGH),
])
def test_core_metadata_must_match_repository_definition(monkeypatch, field, value):
    baseline = build_lead_baseline_registry()
    registry = FeatureRegistry([
        replace(baseline.get(name), **{field: value}) if name == "meter_reading"
        else baseline.get(name) for name in CORE
    ] + [_verified_feature()])
    monkeypatch.setattr("project.features.provided._repository_producer_definitions", _synthetic_producers)
    with pytest.raises(BlockedExperimentError):
        _resolver()(CORE, (*CORE, "temperature_mean"), registry)


@pytest.mark.parametrize("name", ["timestamp", "repeat", "fold", "seed", "fabricated_core"])
@pytest.mark.parametrize("role", ["core", "addition"])
def test_fabricated_core_and_reserved_additions_are_blocked(monkeypatch, name, role):
    from project.features.provided import ProvidedSourceEvidence

    registry = _registry_with(_verified_feature(name), _verified_feature())
    definitions = _synthetic_producers()
    definitions[name] = ProvidedSourceEvidence("tests/fixtures/source.ipynb", "cell", "synthetic_fixture")
    monkeypatch.setattr("project.features.provided._repository_producer_definitions", lambda: definitions)
    if role == "core":
        with pytest.raises(BlockedExperimentError):
            _resolver()((*CORE, name), (*CORE, name, "temperature_mean"), registry)
    elif name != "fabricated_core":
        with pytest.raises(BlockedExperimentError, match="blocked"):
            _resolver()(CORE, (*CORE, name), registry)
    else:
        assert _resolver()(CORE, (*CORE, name), registry) == (*CORE, name)


@pytest.mark.parametrize("name", sorted(NON_PREDICTORS))
def test_all_canonical_reserved_additions_are_blocked_despite_verified_evidence(monkeypatch, name):
    from project.features.provided import ProvidedSourceEvidence

    registry = _registry_with(_verified_feature(name))
    evidence = ProvidedSourceEvidence("tests/fixtures/source.ipynb", "cell", "synthetic_fixture")
    monkeypatch.setattr("project.features.provided._repository_producer_definitions", lambda: {name: evidence})
    with pytest.raises(BlockedExperimentError) as error:
        _resolver()(CORE, (*CORE, name), registry)
    assert error.value.code == "forbidden_provided_feature"
