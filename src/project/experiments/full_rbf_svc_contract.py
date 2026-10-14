"""Pure exact-RBF route contracts; construction is not launch admission."""

from __future__ import annotations

from dataclasses import dataclass, replace

from project.experiments.candidate_contracts import CandidateSpec, _is_sha256
from project.experiments.candidate_dispatch import (
    CandidateArm,
    _configuration,
    build_experiment_arms,
)
from project.experiments.full_input_schema import CORE
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    FullResourcePolicy,
    FullRunSpec,
    SourceBinding,
)
from project.experiments.full_snapshot_codec import configuration_digest
from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1
from project.features.registry import build_lead_baseline_registry

EXPERIMENT_ID = "dim_pca_vs_passthrough"
ROUTE_ID = "exact_rbf_svc_ranking_v1"
ARM_IDS = ("control", "challenger")
_FROZEN_SVC_PARAMETERS = (
    ("C", 1.0), ("kernel", "rbf"), ("degree", 3), ("gamma", "scale"),
    ("coef0", 0.0), ("shrinking", True), ("probability", "deprecated"),
    ("tol", .001), ("cache_size", 200), ("class_weight", None),
    ("verbose", False), ("max_iter", -1), ("decision_function_shape", "ovr"),
    ("break_ties", False), ("random_state", 42),
)
CANCELLATION_SEMANTICS = (
    "explicit_parent_request_only_no_training_deadline",
    "terminate_process_group_and_verify_descendant_cleanup_before_next_fit",
    "append_cancelled_attempt_preserve_debit_and_quarantine_partial_artifacts",
    "stop_queue_pending_reconciliation_no_n03_or_probability_pool_eligibility",
    "retry_requires_explicit_request_fresh_attempt_and_unchanged_bindings",
)


def validate_full_rbf_svc_snapshot(
    snapshot: FullCandidateSnapshot, *, year_built_encoding: str,
) -> None:
    """Reject unreviewed inputs before debit without constructing estimators."""
    if type(snapshot) is not FullCandidateSnapshot or type(snapshot.spec) is not CandidateSpec:
        raise TypeError("exact full candidate snapshot and CandidateSpec required")
    requested = snapshot.spec
    if (requested.experiment_id != EXPERIMENT_ID or requested.model_family != "svm"
            or requested.feature_pool != CORE or snapshot.arms != ARM_IDS
            or snapshot.selector_arms or snapshot.study is not None
            or snapshot.study_method is not None
            or year_built_encoding != LEAD_YEAR_OFFSET_1900_UINT8_V1):
        raise ValueError("RBF exact identity, canonical core and frozen protocol required")
    methods = dict(requested.method_params)
    if any(name != "pca_components" or type(value) is not float or value != .95
           for name, value in methods.items()):
        raise ValueError("RBF PCA method must match frozen variance fraction .95")
    frozen = dict(_FROZEN_SVC_PARAMETERS)
    if any(name not in frozen or type(value) is not type(frozen[name]) or value != frozen[name]
           for name, value in requested.model_params):
        raise ValueError("RBF model parameters must match typed frozen defaults")


def build_full_rbf_svc_arms(
    snapshot: FullCandidateSnapshot, *, year_built_encoding: str,
) -> dict[str, CandidateArm]:
    """Construct the existing train-only recipe; never grants launch admission."""
    validate_full_rbf_svc_snapshot(snapshot, year_built_encoding=year_built_encoding)
    requested = snapshot.spec
    registry = build_lead_baseline_registry(year_built_encoding=year_built_encoding)
    reference = CandidateSpec(EXPERIMENT_ID, CORE, "svm", (), (("pca_components", .95),))
    actual = build_experiment_arms(requested, registry, year_built_encoding=year_built_encoding)
    expected = build_experiment_arms(reference, registry, year_built_encoding=year_built_encoding)
    result = {}
    for name, key in zip(ARM_IDS, ("passthrough", "pca"), strict=True):
        arm, baseline = actual[key], expected[key]
        if (arm.score_kind != "decision_margin"
                or configuration_digest(_configuration(arm)) != configuration_digest(_configuration(baseline))):
            raise ValueError("RBF resolved configuration differs from frozen recipe")
        result[name] = replace(arm, arm_id=name, candidate_id=f"{EXPERIMENT_ID}_{name}")
    return result


@dataclass(frozen=True)
class RBFDevelopmentCardinality:
    """Declared dev-only aggregates bound to sources, not authenticated evidence.

    Parent data preflight must independently derive/recheck these twelve counts.
    Never assume building-disjoint folds have equal row counts. No file is read
    here, and knowing source hashes alone does not authenticate caller counts.
    """

    spec_sha256: str
    data_source: SourceBinding
    split_source: SourceBinding
    development_rows: int
    validation_rows: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.spec_sha256) or self.spec_sha256 == "0" * 64:
            raise ValueError("non-placeholder spec binding required")
        for source, role in ((self.data_source, "data"), (self.split_source, "split")):
            if (type(source) is not SourceBinding or source.role != role
                    or source.sha256 == "0" * 64):
                raise ValueError("exact non-placeholder data/split source binding required")
        if type(self.development_rows) is not int or self.development_rows < 4:
            raise ValueError("positive exact development cardinality required")
        if type(self.validation_rows) is not tuple or len(self.validation_rows) != 3:
            raise ValueError("all three repeat counts required")
        for counts in self.validation_rows:
            if (type(counts) is not tuple or len(counts) != 4
                    or any(type(count) is not int or not 0 < count < self.development_rows for count in counts)
                    or sum(counts) != self.development_rows):
                raise ValueError("four positive exact validation counts must cover development per repeat")


@dataclass(frozen=True)
class RBFFeasibilityEstimate:
    """Deterministic conceptual scaling estimate, not actual RAM or an ETA."""

    spec_sha256: str
    data_source: SourceBinding
    split_source: SourceBinding
    configuration_sha256: tuple[tuple[str, str], ...]
    largest_training_rows: int
    conceptual_dense_gram_bytes: int
    oof_rows_per_arm: int
    resource_policy: FullResourcePolicy
    planned_model_fits: int = 24
    planned_selector_fits: int = 0
    kernel_cache_mib: float = 200.0
    route_id: str = ROUTE_ID
    risk: str = "severe_compute_risk"
    score_kind: str = "decision_margin"
    preprocessing_fit_boundary: str = "outer_train"
    probability_guardrail_status: str = "requires_calibrated_probability"
    execution_authorized: bool = False
    wall_seconds_estimate: None = None
    cancellation_semantics: tuple[str, ...] = CANCELLATION_SEMANTICS


def estimate_full_rbf_svc(
    spec: FullRunSpec, cardinality: RBFDevelopmentCardinality,
) -> RBFFeasibilityEstimate:
    """Check declared lineage and estimate n²*8 without reading or fitting.

    libsvm evaluates/caches kernel entries; this conceptual float64 dense Gram
    size is NOT a claim about its actual allocation. The 200 MiB frozen kernel
    cache is an estimator parameter, NOT a campaign RSS cap or total-RAM bound.
    Kernel training can scale worse than quadratically; PCA reduces numeric
    width, not sample count. No wall-time prediction or feasibility guarantee.
    """
    if type(spec) is not FullRunSpec or type(cardinality) is not RBFDevelopmentCardinality:
        raise TypeError("exact full spec and declared cardinality required")
    # Revalidate frozen objects rather than trusting externally mutated fields.
    spec = replace(spec)
    cardinality = replace(cardinality)
    sources = {role: tuple(item for item in spec.sources if item.role == role)
               for role in ("data", "split")}
    if (sources["data"] != (cardinality.data_source,)
            or sources["split"] != (cardinality.split_source,)
            or cardinality.spec_sha256 != spec.binding_sha256
            or cardinality.development_rows != spec.development_rows):
        raise ValueError("development source/cardinality binding mismatch")
    snapshot = next((item for item in spec.candidates if item.spec.experiment_id == EXPERIMENT_ID), None)
    if snapshot is None:
        raise ValueError("RBF snapshot absent from run")
    arms = build_full_rbf_svc_arms(snapshot, year_built_encoding=LEAD_YEAR_OFFSET_1900_UINT8_V1)
    largest = max(spec.development_rows - count
                  for counts in cardinality.validation_rows for count in counts)
    return RBFFeasibilityEstimate(
        spec.binding_sha256, cardinality.data_source, cardinality.split_source,
        tuple((name, configuration_digest(_configuration(arm))) for name, arm in arms.items()),
        largest, largest * largest * 8, spec.development_rows * spec.repeats, spec.resource_policy,
    )
