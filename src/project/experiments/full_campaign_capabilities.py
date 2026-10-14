"""Authenticated fixed ordinary transport route, never comparative permission."""

import hashlib
import hmac
import secrets
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from project.experiments.candidate_contracts import _validate_managed_path
from project.experiments.full_run_contracts import SourceBinding, _file_hash, _json

ORDINARY_ROUTE_ID = "reviewed_ordinary_transport_codec_v1"
ORDINARY_EXPERIMENTS = (
    "feat_missingness_indicators",
    "feat_categorical_interactions",
    "feat_frequency_encoding",
    "feat_meter_deviation_ratio",
    "feat_histogram_quantile",
    "feat_missingness_bitmask",
    "feat_safe_calendar_numeric_transforms",
    "dim_semantic_block_drop_calendar",
    "dim_semantic_block_drop_measurement",
    "dim_semantic_block_drop_building_metadata",
    "dim_fold_local_selector_vs_passthrough",
    "regime_natural_vs_class_weighted",
    "feat_missingness_passthrough_comparison",
    "model_extratrees_core_features",
)
TE_ROUTE_ID = "reviewed_target_encoding_transport_v1"
HPO_ROUTE_ID = "reviewed_full_hpo_transport_v1"
AG_ROUTE_ID = "reviewed_autogluon_xt_transport_v1"
PCA_ROUTE_ID = "exact_rbf_svc_ranking_v1"
PROVIDED_ROUTE_ID = "reviewed_provided_weather_transport_v1"
PREPARED_TE_EXPERIMENTS = ("feat_conditional_cross_fitted_target_encoding",)
_ROUTES = {
    ORDINARY_ROUTE_ID: ORDINARY_EXPERIMENTS,
    TE_ROUTE_ID: PREPARED_TE_EXPERIMENTS,
    PCA_ROUTE_ID: ("dim_pca_vs_passthrough",),
    HPO_ROUTE_ID: ("regime_same_family_hpo_vs_default",),
    AG_ROUTE_ID: ("model_autogluon_core_features",),
    PROVIDED_ROUTE_ID: ("feat_core_vs_provided",),
}
_KEY = secrets.token_bytes(32)
_REFERENCE_ROOT = Path(__file__).resolve().parents[3]
# Every promoted execution module belongs to the authenticated closure.
_UNPROMOTED_MODULES = set()
_ROUTE_FILES = (
    "docs/superpowers/specs/2026-10-04-full-campaign-decisions.md",
    "src/project/experiments/full_campaign_capabilities.py",
    "src/project/experiments/full_campaign_preflight.py",
    "src/project/experiments/full_campaign_evidence.py",
    "src/project/experiments/full_candidate_runner.py",
    "src/project/experiments/full_campaign_runtime.py",
    "src/project/experiments/full_campaign_worker.py",
    "src/project/experiments/full_snapshot_codec.py",
    "src/project/experiments/full_run_contracts.py",
    "src/project/experiments/full_campaign_admission.py",
    "src/project/experiments/full_candidate_io.py",
    "src/project/experiments/candidate_dispatch.py",
    "src/project/experiments/fit_accounting.py",
    "src/project/models/candidate_artifacts.py",
    "src/project/models/full_candidate_artifacts.py",
    "src/project/experiments/baseline_budget.py",
    "src/project/experiments/candidate_contracts.py",
    "src/project/experiments/full_input_schema.py",
    "src/project/experiments/inner_cv.py",
    "src/project/experiments/nested_tuning.py",
    "src/project/__init__.py",
    "src/project/config.py",
    "src/project/paths.py",
    "scripts/train.py",
)
_ROUTE_FILES += (
    "src/project/experiments/full_rbf_svc_contract.py",
    "src/project/experiments/full_rbf_svc_worker.py",
    "src/project/experiments/full_nested_tuning.py",
    "src/project/experiments/full_hpo_lifecycle.py",
    "src/project/experiments/full_hpo_worker.py",
    "src/project/experiments/full_hpo_campaign.py",
    "src/project/experiments/full_autogluon_contract.py",
    "src/project/experiments/full_autogluon_svc_worker.py",
    "src/project/experiments/full_campaign_cli.py",
)
_ROUTE_FILES = tuple(sorted(set(_ROUTE_FILES) | {
    str(path.relative_to(_REFERENCE_ROOT))
    for package in ("features", "models", "data")
    for path in (_REFERENCE_ROOT / "src/project" / package).rglob("*.py")
    if str(path.relative_to(_REFERENCE_ROOT)) not in _UNPROMOTED_MODULES
}))
# Bind the running installation independently of the requested root. Merely
# copying names, changing a file and rehashing it, or constructing this report
# cannot substitute an implementation for the fixed route.
_REFERENCE_BINDINGS = tuple(SourceBinding("implementation", name,
                             _file_hash(_REFERENCE_ROOT / name)) for name in _ROUTE_FILES)


def build_reviewed_ordinary_arms(snapshot, *, year_built_encoding: str):
    """Recompute exact untuned comparison graphs at each authenticated boundary.

    Capability membership alone is not configuration equivalence. Compare the
    requested dispatch against a separately constructed frozen-baseline spec,
    including transformer/registry/preprocessing parameters and score semantics.
    This constructs unfitted estimators only; it grants no fit or completion.
    """
    from project.experiments.candidate_contracts import CandidateSpec
    from project.experiments.candidate_dispatch import (
        _configuration,
        build_experiment_arms,
    )
    from project.experiments.full_input_schema import canonical_feature_order
    from project.experiments.full_run_contracts import (
        SEMANTIC_VARIANTS,
        FullCandidateSnapshot,
        source_matrix_identity,
    )
    from project.experiments.full_snapshot_codec import configuration_digest
    from project.features.registry import build_lead_baseline_registry

    if type(snapshot) is not FullCandidateSnapshot:
        raise TypeError("exact ordinary candidate snapshot required")
    requested = snapshot.spec
    histogram = requested.experiment_id == "feat_histogram_quantile"
    trees = requested.experiment_id == "model_extratrees_core_features"
    selector = requested.experiment_id == "dim_fold_local_selector_vs_passthrough"
    semantic = requested.experiment_id in SEMANTIC_VARIANTS
    target_encoding = requested.experiment_id in PREPARED_TE_EXPERIMENTS
    family = "logistic_l2" if selector else "extra_trees" if trees else "hist_gradient_boosting"
    methods = (("n_bins", 8), ("n_quantiles", 64)) if histogram else ()
    if target_encoding:
        methods = (("smoothing", 10.0), ("n_splits", 3), ("seed", 42), ("max_categories", 4096))
        if (requested.method_params != methods
                or any(type(actual) is not type(expected)
                       for (_, actual), (_, expected) in zip(requested.method_params, methods, strict=True))):
            raise ValueError("target-encoding methods are not exactly reviewed")
    if semantic:
        methods = (("drop_block", requested.experiment_id.removeprefix("dim_semantic_block_drop_")),)
    arms = ("control", "bin", "quantile") if histogram else ("control", "challenger")
    if (requested.experiment_id not in ORDINARY_EXPERIMENTS + PREPARED_TE_EXPERIMENTS + ("feat_core_vs_provided",)
            or requested.model_family != family
            or any(name not in dict(methods) or type(value) is not type(dict(methods)[name])
                   or value != dict(methods)[name] for name, value in requested.method_params)
            or snapshot.study is not None
            or snapshot.study_method is not None or snapshot.selector_arms != (("challenger",) if selector else ())
            or semantic and requested.method_params != methods
            or snapshot.arms != arms):
        raise ValueError("ordinary recipe/configuration is not reviewed")
    canonical_feature_order(requested.feature_pool)
    parameters = (("n_estimators", 64), ("max_depth", 12), ("min_samples_leaf", 2),
                  ("max_features", "sqrt"), ("bootstrap", False), ("class_weight", None),
                  ("n_jobs", 1), ("random_state", 42)) if trees else (
                      ("early_stopping", False), ("random_state", 42))
    if selector:
        parameters = ()
    reference = CandidateSpec(source_matrix_identity(requested.experiment_id), requested.feature_pool,
                              family, parameters, methods)
    if requested.experiment_id == "feat_core_vs_provided":
        from project.features.provided import build_provided_feature_registry
        registry = build_provided_feature_registry(year_built_encoding=year_built_encoding)
    else:
        registry = build_lead_baseline_registry(year_built_encoding=year_built_encoding)
    actual = build_experiment_arms(replace(requested, experiment_id=source_matrix_identity(requested.experiment_id)),
                                   registry=registry, year_built_encoding=year_built_encoding)
    expected = build_experiment_arms(reference, registry=registry, year_built_encoding=year_built_encoding)
    if selector or semantic:
        keys = ("passthrough", "selector" if selector else "drop_" + dict(methods)["drop_block"])
        def identified(items):
            return {name: replace(items[key], arm_id=name,
                    candidate_id=f"{requested.experiment_id}_{name}")
                    for name, key in zip(snapshot.arms, keys, strict=True)}
        actual, expected = identified(actual), identified(expected)
    if tuple(actual) != snapshot.arms or tuple(expected) != snapshot.arms:
        raise ValueError("ordinary exact arm contract mismatch")
    for name in snapshot.arms:
        arm, baseline = actual[name], expected[name]
        if (arm.score_kind != "positive_probability" or arm.score_kind != baseline.score_kind
                or configuration_digest(_configuration(arm)) != configuration_digest(_configuration(baseline))):
            raise ValueError("ordinary resolved configuration differs from frozen recipe")
    return actual

def route_for_experiments(experiment_ids: tuple[str, ...] | None = None) -> str:
    """Keep ordinary defaults; separately reviewed recipes cannot be mixed."""
    selected = ORDINARY_EXPERIMENTS if experiment_ids is None else experiment_ids
    if type(selected) is not tuple or not selected or len(set(selected)) != len(selected):
        raise ValueError("unsupported route selection")
    if set(selected) <= set(ORDINARY_EXPERIMENTS):
        return ORDINARY_ROUTE_ID
    for route, identities in _ROUTES.items():
        if selected == identities:
            return route
    raise ValueError("unsupported or mixed reviewed route selection")


def build_reviewed_route_arms(snapshot, *, year_built_encoding: str):
    """Reconstruct the selected exact recipe at every trust boundary."""
    route = route_for_experiments((snapshot.spec.experiment_id,))
    if route in (ORDINARY_ROUTE_ID, TE_ROUTE_ID, PROVIDED_ROUTE_ID):
        return build_reviewed_ordinary_arms(snapshot, year_built_encoding=year_built_encoding)
    if route == PCA_ROUTE_ID:
        from project.experiments.full_rbf_svc_contract import build_full_rbf_svc_arms
        return build_full_rbf_svc_arms(snapshot, year_built_encoding=year_built_encoding)
    if route == AG_ROUTE_ID:
        from project.experiments.full_autogluon_contract import build_reviewed_autogluon_arms
        return build_reviewed_autogluon_arms(snapshot, year_built_encoding=year_built_encoding)
    from dataclasses import asdict
    from project.experiments.full_nested_tuning import full_hpo_study_spec, validate_full_hpo_factory
    from project.experiments.nested_tuning import DEFAULT_HGB_PARAMS, build_hgb_tuning_arm
    from project.experiments.full_snapshot_codec import canonical_json
    from project.features.registry import build_lead_baseline_registry
    if (snapshot.study is None or snapshot.study_method != "random_search"
            or canonical_json(asdict(snapshot.study)) != canonical_json(asdict(full_hpo_study_spec()))):
        raise ValueError("HPO requires exact typed frozen T8 K3 seed42 finite space")
    validate_full_hpo_factory(snapshot)
    from project.experiments.candidate_dispatch import CandidateArm
    if snapshot.spec.method_params:
        raise ValueError("HPO exact methods reside only in frozen study snapshot")
    registry = build_lead_baseline_registry(year_built_encoding=year_built_encoding)
    return {name: CandidateArm(f"{snapshot.spec.experiment_id}_{name}", name,
        snapshot.spec.feature_pool, build_hgb_tuning_arm(DEFAULT_HGB_PARAMS), registry,
        "positive_probability") for name in snapshot.arms}


@dataclass(frozen=True)
class FullCampaignCapability:
    """Read-only capability report, not a launch grant."""

    source_root: str
    available: bool = False
    blocker_codes: tuple[str, ...] = ("full_dispatch_unavailable",)
    route_id: str | None = None
    supported_experiment_ids: tuple[str, ...] = ()
    implementation_sources: tuple[SourceBinding, ...] = ()
    seal: str = ""


def _seal(evidence: FullCampaignCapability) -> str:
    payload = asdict(evidence)
    payload.pop("seal")
    return hmac.new(_KEY, _json(payload).encode(), hashlib.sha256).hexdigest()


def _verify_route_sources(root: Path) -> None:
    for binding in _REFERENCE_BINDINGS:
        binding.verify(_REFERENCE_ROOT)
        binding.verify(root)


def resolve_full_campaign_capability(root: str | Path, *, experiment_ids: tuple[str, ...] | None = None) -> FullCampaignCapability:
    """Resolve repository capability without writes, imports of workers or fits."""
    root_path = _validate_managed_path(root, "capability source root", directory=True)
    route = route_for_experiments(experiment_ids)
    try:
        _verify_route_sources(root_path)
    except (OSError, ValueError):
        return FullCampaignCapability(str(root_path))
    evidence = FullCampaignCapability(str(root_path), True, (), route,
                                     _ROUTES[route], _REFERENCE_BINDINGS)
    return replace(evidence, seal=_seal(evidence))


def verify_full_campaign_capability(root: str | Path, evidence: FullCampaignCapability) -> None:
    """Fail closed; future resolver-issued grants need authenticated bindings."""
    if type(evidence) is not FullCampaignCapability:
        raise TypeError("repository-resolved capability evidence required")
    if type(evidence.seal) is not str or not hmac.compare_digest(evidence.seal, _seal(evidence)):
        raise ValueError("forged or stale capability evidence")
    root_path = _validate_managed_path(root, "capability source root", directory=True)
    if (evidence.source_root != str(root_path) or evidence.route_id not in _ROUTES
            or evidence.implementation_sources != _REFERENCE_BINDINGS
            or evidence.supported_experiment_ids != _ROUTES[evidence.route_id]
            or evidence.blocker_codes):
        raise ValueError("reviewed route capability binding mismatch")
    if evidence.available is not True:
        raise ValueError("full_dispatch_unavailable: subprocess worker/journal/shard route required")
    _verify_route_sources(root_path)
