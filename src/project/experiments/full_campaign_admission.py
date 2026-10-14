"""Read-only campaign inspection; optional audit authentication, never launch permission.

Without reviewed development evidence this remains source-only. When immutable
development evidence exists, its separate gate streams bound input hashes and
reads split metadata only; it never analyzes prepared labels/features or fits.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import yaml

from project.experiments.candidate_contracts import BlockedExperimentError, CandidateSpec
from project.experiments.full_input_schema import CORE, canonical_feature_order
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    FullResourcePolicy,
    FullRunSpec,
    FullStatusRecord,
    SourceBinding,
)

MATRIX_PATH = "configs/experiments/feature_campaign.yaml"
BASELINE_PATH = "configs/experiments/baselines.yaml"
NOTEBOOK_PATH = "notebooks/02_baseline_and_candidate_design.ipynb"
PRODUCER_PATH = "src/project/features/registry.py"
SPLIT_MANIFEST_PATH = "outputs/splits/building_folds_repeated.parquet"
MATRIX_IDS = (
    "feat_core_vs_provided", "feat_missingness_indicators",
    "feat_preserve_raw_vs_invalid_sentinel", "feat_categorical_interactions",
    "feat_frequency_encoding", "feat_conditional_cross_fitted_target_encoding",
    "feat_group_aggregates_blocked", "feat_meter_deviation_ratio",
    "feat_histogram_quantile", "feat_missingness_bitmask",
    "feat_safe_calendar_numeric_transforms", "dim_pca_vs_passthrough",
    "dim_semantic_block_drop", "dim_fold_local_selector_vs_passthrough",
    "regime_natural_vs_class_weighted", "regime_same_family_hpo_vs_default",
    "prohibited_opaque_building_id_digit_extraction", "blocked_unverified_gte_family",
    "feat_missingness_passthrough_comparison", "model_extratrees_core_features",
    "model_autogluon_core_features",
)
FIXED_BLOCKERS = {
    "feat_core_vs_provided": ("no_verified_additional_features",),
    "feat_preserve_raw_vs_invalid_sentinel": ("sentinel_evidence_unverified",),
    "feat_group_aggregates_blocked": ("blocked_group_aggregates",),
    "blocked_unverified_gte_family": ("blocked_unverified_gte",),
    "regime_same_family_hpo_vs_default": ("hpo_dispatch_missing", "hpo_method_space_unapproved"),
    "model_autogluon_core_features": ("full_lease_unsupported", "autogluon_full_profile_missing"),
    "prohibited_opaque_building_id_digit_extraction": ("prohibited_experiment",),
}


@dataclass(frozen=True)
class FullCampaignIntent:
    """Resolved subset plus explicit user intent; omitted rows remain inventoried."""

    spec: FullRunSpec
    explicit_full_request: bool
    sequential: bool = True
    resource_opt_out_record: str | None = None


@dataclass(frozen=True)
class FullAdmissionRow:
    record: FullStatusRecord
    model_family: str | None
    arms: tuple[str, ...] = ()
    planned_model_fits: int | None = None
    planned_selector_fits: int | None = None
    planned_trials: int | None = None
    encoder_objects: int | None = None
    scientific_disposition: Literal["unverified", "data_ready_pending_runtime_proof", "not_applicable"] = "unverified"
    disposition_reason: str | None = None
    development_evidence_sha256: str | None = None


@dataclass(frozen=True)
class FullCampaignAdmission:
    """Detached source audit metadata is canonical JSON, never model configs.

    ``deferred_sources`` are unverified data/split declarations. Only actual
    source-byte checks enter ``authenticated_sources``. No field is a launch
    token or completed empirical evidence.
    """

    rows: tuple[FullAdmissionRow, ...]
    blocker_codes: tuple[str, ...]
    authenticated_sources: tuple[SourceBinding, ...]
    audit_families: tuple[tuple[str, str], ...]
    context_anchors: tuple[tuple[str, str], ...]
    requested_spec_sha256: str
    resource_policy: FullResourcePolicy
    deferred_sources: tuple[SourceBinding, ...]
    audit_source_metadata: tuple[tuple[str, str], ...]
    context_source_metadata: tuple[tuple[str, str], ...]
    full_request_recorded: bool
    partition_count: int = field(default=12, init=False)
    vault_excluded: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)
    execution_authorized: bool = field(default=False, init=False)
    n03_eligible: bool = field(default=False, init=False)


def _read_source(root: Path, relative: str, role: str) -> tuple[bytes, SourceBinding]:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative:
        raise ValueError("source_path_invalid")
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("source_path_invalid")
    if not current.is_file() or current.stat().st_nlink != 1:
        raise ValueError("source_path_invalid")
    payload = current.read_bytes()
    return payload, SourceBinding(role, relative, hashlib.sha256(payload).hexdigest())


def _authenticate(root: Path, spec: FullRunSpec):
    """Only source assets are read; dataset/split digests remain declarations."""
    payloads, authenticated, codes = {}, [], []
    for source in spec.sources:
        if source.role in ("data", "split"):
            expected = {"data": "outputs/prepared/train_joined.parquet",
                        "split": SPLIT_MANIFEST_PATH}
            if source.path != expected[source.role]:
                codes.append(f"{source.role}_source_path_mismatch")
            continue
        allowed = {
            "matrix": (MATRIX_PATH,), "baseline": (BASELINE_PATH,),
            "notebook": (NOTEBOOK_PATH, "notebooks/01_data_understanding.ipynb"),
            "producer": (PRODUCER_PATH,),
            "encoding": ("docs/evidence/year_built_encoding_v1.json",),
            "autogluon_profile": ("docs/evidence/autogluon_full_profile_v1.json",),
        }
        if source.path not in allowed.get(source.role, ()):
            codes.append("unsupported_producer" if source.role == "producer" else "source_path_invalid")
            continue
        try:
            payload, actual = _read_source(root, source.path, source.role)
            if actual != source:
                codes.append("source_hash_mismatch")
                continue
            payloads[source.path] = payload
            authenticated.append(actual)
        except (OSError, ValueError):
            codes.append("source_path_invalid")
    if BASELINE_PATH not in payloads:
        codes.append("baseline_source_missing")
    if MATRIX_PATH not in payloads:
        codes.append("matrix_source_missing")
    return payloads, tuple(authenticated), codes


def _protocol_valid(config: dict, baseline: dict, sources: tuple[SourceBinding, ...]) -> bool:
    if tuple(source.path for source in sources if source.role == "split") != (SPLIT_MANIFEST_PATH,):
        return False
    expected = {"folds": 4, "repeats": 3, "positive_label": 1}
    for mapping in (config.get("prediction_contract", {}), baseline):
        if any(type(mapping.get(k)) is not int or mapping.get(k) != v for k, v in expected.items()):
            return False
        if mapping.get("target") != "anomaly" or mapping.get("split_seeds") != [42, 43, 44]:
            return False
        if any(type(seed) is not int for seed in mapping["split_seeds"]):
            return False
        if mapping.get("primary_metric") != "roc_auc":
            return False
        if mapping.get("guardrails") != ["average_precision", "log_loss", "brier_score"]:
            return False
        if mapping.get("split_manifest") != SPLIT_MANIFEST_PATH:
            return False
    expected_models = {
        "dummy_prior": {"model_family": "dummy", "recipe_id": "recipe_dummy_prior",
                        "model_config_id": "cfg_dummy_prior", "params": {"strategy": "prior"}},
        "logistic_l2": {"model_family": "linear_model", "recipe_id": "recipe_logistic_l2",
                        "model_config_id": "cfg_logistic_l2", "params": {
                            "C": 1.0, "l1_ratio": 0.0, "class_weight": None, "max_iter": 1000}},
        "hist_gradient_boosting": {"model_family": "ensemble_gbdt",
                                   "recipe_id": "recipe_hist_gradient_boosting",
                                   "model_config_id": "cfg_hist_gradient_boosting",
                                   "params": {"early_stopping": False}},
    }
    # JSON comparison distinguishes exact bool/int and frozen numeric profiles.
    return (type(baseline.get("seed")) is int and baseline["seed"] == 42
            and json.dumps(baseline.get("models"), sort_keys=True) == json.dumps(expected_models, sort_keys=True))


def _cell_codes(candidate: FullCandidateSnapshot, notebooks: dict) -> list[str]:
    notebook = notebooks.get(candidate.source_notebook, {})
    cells = [c for c in notebook.get("cells", []) if c.get("id") == candidate.source_cell_id]
    if len(cells) != 1:
        return ["source_cell_missing"]
    source = cells[0].get("source", [])
    text = source if isinstance(source, str) else "".join(source)
    # Current source renders all rows from the independently hashed config.
    linked = (candidate.source_notebook == NOTEBOOK_PATH
              and candidate.source_cell_id == "cand-matrix-render"
              and cells[0].get("cell_type") == "code"
              and "campaign['experiment_matrix']" in text)
    return [] if linked or candidate.spec.experiment_id in text else ["source_cell_linkage_mismatch"]


def _methods_valid(spec: CandidateSpec) -> bool:
    methods = dict(spec.method_params)
    allowlists = {
        "feat_histogram_quantile": {"n_bins", "n_quantiles"},
        "feat_conditional_cross_fitted_target_encoding": {"smoothing", "n_splits", "cv_splits", "seed", "max_categories"},
        "dim_pca_vs_passthrough": {"pca_components"},
        "dim_semantic_block_drop": {"drop_block"},
        "feat_preserve_raw_vs_invalid_sentinel": {"rule", "sentinel_rule", "column", "invalid_values", "replacement_value", "evidence_reference", "evidence_sha256", "evidence_root"},
        "regime_same_family_hpo_vs_default": {"study_spec", "study_dir"},
    }
    if set(methods) - allowlists.get(spec.experiment_id, set()):
        return False
    bounds = {"n_bins": (2, 64), "n_quantiles": (2, 256), "n_splits": (2, 5),
              "cv_splits": (2, 5), "seed": (0, 2**32 - 1), "max_categories": (1, 4096)}
    for key, value in methods.items():
        if key in bounds and (type(value) is not int or not bounds[key][0] <= value <= bounds[key][1]):
            return False
        if key == "smoothing" and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1000):
            return False
        if key == "pca_components" and (type(value) is not float or not 0 < value < 1):
            return False
    return not {"n_splits", "cv_splits"} <= set(methods)


def _dependency_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _model_parameters_valid(spec: CandidateSpec) -> bool:
    """Inspect fixed estimator signatures, never instantiate or fit a model."""
    if not spec.model_params:
        return True
    if spec.model_family == "autogluon":
        return False  # Existing bounded settings cannot express full opt-out.
    if spec.model_family == "extra_trees":
        defaults = {"n_estimators": 64, "max_depth": 12, "min_samples_leaf": 2,
                    "max_features": "sqrt", "bootstrap": False, "class_weight": None,
                    "n_jobs": 1, "random_state": 42}
    else:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
        except ImportError:
            return False
        estimator = {"svm": SVC, "logistic_l2": LogisticRegression}.get(spec.model_family, HistGradientBoostingClassifier)
        defaults = {key: p.default for key, p in inspect.signature(estimator).parameters.items() if p.default is not inspect.Parameter.empty}
        if spec.model_family == "svm":
            defaults.update(C=1., kernel="rbf", gamma="scale", random_state=42)
        elif spec.model_family == "logistic_l2":
            defaults.update(C=1., l1_ratio=0., solver="lbfgs", max_iter=1000, random_state=42)
        else:
            defaults.update(early_stopping=False, random_state=42)
        if spec.experiment_id in ("feat_missingness_passthrough_comparison", "regime_natural_vs_class_weighted"):
            defaults = {key: defaults[key] for key in ("early_stopping", "random_state")}
    return all(key in defaults and type(value) is type(defaults[key]) and value == defaults[key]
               for key, value in spec.model_params)


def _candidate_row(row: dict, context: tuple) -> FullAdmissionRow:
    snapshots, notebooks, core, global_codes = context
    recipe = row["id"]
    candidate = snapshots.get(recipe)
    codes = list(global_codes) + list(FIXED_BLOCKERS.get(recipe, ()))
    arms = ()
    fits = selectors = trials = encoders = None
    if candidate is None:
        codes.append("candidate_configuration_pending")
    else:
        codes.extend(_cell_codes(candidate, notebooks))
        if candidate.spec.model_family != row["model_family"]:
            codes.append("model_family_mismatch")
        try:
            canonical_pool = canonical_feature_order(candidate.spec.feature_pool)
        except ValueError:
            canonical_pool = None
        from project.features.provided import WEATHER_FEATURES
        expected_pool = (core + WEATHER_FEATURES) if recipe == "feat_core_vs_provided" else core
        if canonical_pool != expected_pool:
            codes.append("raw_feature_pool_unverified")
        if not _model_parameters_valid(candidate.spec):
            codes.append("model_profile_unverified")
        if not _methods_valid(candidate.spec):
            codes.append("invalid_candidate_parameters")
        methods = dict(candidate.spec.method_params)
        arms = candidate.arms
        expected_arms = (("challenger",) if recipe == "model_autogluon_core_features" else
                         ("control", "bin", "quantile") if recipe == "feat_histogram_quantile" else ("control", "challenger"))
        if arms != expected_arms:
            codes.append("arm_contract_mismatch")
        expected_selectors = ("challenger",) if recipe == "dim_fold_local_selector_vs_passthrough" else ()
        if candidate.selector_arms != expected_selectors:
            codes.append("selector_accounting_mismatch")
        units = candidate.planned_units
        selectors = sum(unit.endswith("/selector") for unit in units)
        fits = len(units) - selectors
        trials = 12 * candidate.study.n_trials if candidate.study else 0
        encoders = 0
        if recipe == "feat_conditional_cross_fitted_target_encoding":
            splits = methods.get("n_splits", methods.get("cv_splits", 3))
            if type(splits) is not int or not 2 <= splits <= 5:
                codes.append("invalid_encoder_splits")
                encoders = None
            else:
                encoders = 12 * (splits + 1)
            codes.append("inner_class_support_unverified")
    if recipe == "dim_semantic_block_drop":
        methods = dict(candidate.spec.method_params) if candidate else {}
        if methods.get("drop_block") not in ("calendar", "measurement", "building_metadata"):
            codes.append("missing_drop_block")
    if recipe == "regime_same_family_hpo_vs_default" and candidate is None:
        codes.extend(("hpo_config_pending", "hpo_cost_unknown"))
    if candidate is not None and recipe == "feat_core_vs_provided":
        try:
            from project.features.provided import WEATHER_FEATURES, build_provided_feature_registry, resolve_provided_pool
            resolve_provided_pool(core, candidate.spec.feature_pool, build_provided_feature_registry())
        except (ValueError, BlockedExperimentError):
            pass
        else:
            if candidate.spec.feature_pool == core + WEATHER_FEATURES:
                codes = [code for code in codes if code != "no_verified_additional_features"]
    if candidate is not None and recipe == "regime_same_family_hpo_vs_default":
        try:
            from project.experiments.full_campaign_capabilities import build_reviewed_route_arms
            from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1
            build_reviewed_route_arms(candidate, year_built_encoding=LEAD_YEAR_OFFSET_1900_UINT8_V1)
        except (ValueError, TypeError):
            codes.append("hpo_method_space_unapproved")
        else:
            codes = [code for code in codes if code not in ("hpo_dispatch_missing", "hpo_method_space_unapproved")]
            codes.append("inner_class_support_unverified")
    if candidate is not None and recipe == "model_autogluon_core_features":
        try:
            from project.experiments.full_autogluon_contract import validate_full_autogluon_snapshot
            from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1
            validate_full_autogluon_snapshot(candidate, year_built_encoding=LEAD_YEAR_OFFSET_1900_UINT8_V1)
        except (ValueError, TypeError):
            codes.append("autogluon_full_profile_missing")
        else:
            codes = [code for code in codes if code not in ("full_lease_unsupported", "autogluon_full_profile_missing")]
    if set(codes) & {"invalid_candidate_parameters", "model_profile_unverified",
                     "arm_contract_mismatch", "selector_accounting_mismatch",
                     "model_family_mismatch", "raw_feature_pool_unverified"}:
        fits = selectors = trials = encoders = None
    status = "blocked" if codes else "pending_implementation"
    if recipe == "prohibited_opaque_building_id_digit_extraction":
        status = "prohibited"
    codes.extend(("full_dispatch_unavailable", "real_preflight_pending"))
    record = FullStatusRecord(recipe, status, tuple(dict.fromkeys(codes)))
    return FullAdmissionRow(record, row["model_family"], arms, fits, selectors, trials, encoders)


def inspect_full_campaign(root: str | Path, intent: FullCampaignIntent) -> FullCampaignAdmission:
    """Authenticate source bytes and inventory all 21 rows without real data IO.

    A snapshot is declarative input, not an admitted run. Dataset/split content,
    class support and a reviewed full dispatch are unresolved preflight gates.
    All report eligibility/authorization fields remain false.
    """
    if not isinstance(intent, FullCampaignIntent) or not isinstance(intent.spec, FullRunSpec):
        raise TypeError("typed FullCampaignIntent and FullRunSpec required")
    root = Path(root).absolute()
    if ".." in root.parts or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("source_path_invalid")
    payloads, authenticated, codes = _authenticate(root, intent.spec)
    # Inventory comes from the repository matrix even when the supplied digest
    # is forged. A malformed/non-authoritative matrix fails closed. When the
    # binding authenticated, parse exactly the authenticated bytes so report
    # content can never diverge from the digest in ``authenticated_sources``.
    if MATRIX_PATH in payloads:
        matrix_bytes = payloads[MATRIX_PATH]
    else:
        matrix_bytes, _ = _read_source(root, MATRIX_PATH, "matrix")
    campaign = yaml.safe_load(matrix_bytes)
    matrix = campaign["experiment_matrix"]
    if tuple(r["id"] for r in matrix) != MATRIX_IDS:
        raise ValueError("authoritative_matrix_mismatch")
    for row in matrix:
        recipe = row["id"]
        family = {"dim_pca_vs_passthrough": "svm",
                  "dim_fold_local_selector_vs_passthrough": "logistic_l2",
                  "model_extratrees_core_features": "extra_trees",
                  "model_autogluon_core_features": "autogluon",
                  "prohibited_opaque_building_id_digit_extraction": None,
                  "blocked_unverified_gte_family": None}.get(recipe, "hist_gradient_boosting")
        if row.get("model_family") != family:
            codes.append("source_matrix_mismatch")
    baseline = yaml.safe_load(payloads.get(BASELINE_PATH, b"{}"))
    if not _protocol_valid(campaign, baseline, intent.spec.sources):
        codes.append("prediction_contract_mismatch")
    if intent.explicit_full_request is not True:
        codes.append("explicit_full_request_missing")
    if intent.sequential is not True:
        codes.append("sequential_execution_required")
    snapshots = {c.spec.experiment_id: c for c in intent.spec.candidates}
    from project.experiments.full_run_contracts import (
        SEMANTIC_VARIANTS,
        source_matrix_identity,
    )
    if set(snapshots) - set(MATRIX_IDS) - set(SEMANTIC_VARIANTS):
        codes.append("experiment_not_in_source_matrix")
    ordered = tuple(identity for row in MATRIX_IDS for identity in (
        SEMANTIC_VARIANTS if row == "dim_semantic_block_drop" and set(snapshots) & set(SEMANTIC_VARIANTS)
        else (row,)) if identity in snapshots)
    if ordered != tuple(snapshots):
        codes.append("candidate_source_order_mismatch")
    semantic = tuple(identity for identity in SEMANTIC_VARIANTS if identity in snapshots)
    if semantic and (semantic != SEMANTIC_VARIANTS or "dim_semantic_block_drop" in snapshots):
        codes.append("semantic_variant_coverage_pending")
    notebooks = {}
    for path, payload in payloads.items():
        if path.endswith(".ipynb"):
            try:
                notebooks[path] = json.loads(payload)
            except (ValueError, UnicodeError):
                codes.append("source_notebook_invalid")
    core = tuple(baseline.get("features", {}).get("numeric", [])) + tuple(baseline.get("features", {}).get("categorical", []))
    if core != CORE:
        codes.append("baseline_feature_contract_mismatch")
    if any(not _dependency_available(name) for name in ("sklearn", "numpy", "pandas", "polars", "pyarrow")):
        codes.append("required_dependency_unavailable")
    global_codes = tuple(dict.fromkeys(codes))
    context = (snapshots, notebooks, core, global_codes)
    rows = []
    for row in matrix:
        if row["id"] == "dim_semantic_block_drop" and semantic:
            variants = []
            for identity in semantic:
                candidate = snapshots[identity]
                block = identity.removeprefix("dim_semantic_block_drop_")
                if candidate.spec.method_params != (("drop_block", block),):
                    raise ValueError("semantic variant identity/drop_block mismatch")
                linked = replace(candidate, spec=replace(candidate.spec, experiment_id=source_matrix_identity(identity)))
                variants.append(_candidate_row(row, ({row["id"]: linked}, notebooks, core, global_codes)))
            first = variants[0]
            blockers = tuple(dict.fromkeys(code for v in variants for code in v.record.blocker_codes))
            rows.append(replace(first, record=replace(first.record, blocker_codes=blockers,
                status="blocked" if any(v.record.status == "blocked" for v in variants) else first.record.status),
                planned_model_fits=None if any(v.planned_model_fits is None for v in variants)
                else sum(v.planned_model_fits for v in variants)))
        else:
            rows.append(_candidate_row(row, context))
    # This disposition is data evidence only. It never adds target encoding to
    # ORDINARY_EXPERIMENTS or clears runtime/launch/completion gates. Retain all
    # source matrix rows, including a scientifically non-applicable sentinel.
    from project.experiments.full_development_readiness import (
        EVIDENCE_PATH,
        SENTINEL_ID,
        TARGET_ID,
        inspect_development_readiness,
    )
    if (root / EVIDENCE_PATH).exists():
        try:
            audit_spec = intent.spec
            profile_sources = tuple(s for s in audit_spec.sources if s.role == "autogluon_profile")
            if profile_sources:
                from project.experiments.full_autogluon_contract import verify_autogluon_profile_source
                verify_autogluon_profile_source(audit_spec, root)
                audit_spec = replace(audit_spec, sources=tuple(s for s in audit_spec.sources
                                                             if s.role != "autogluon_profile"))
            development = inspect_development_readiness(root, audit_spec)
        except (OSError, ValueError, TypeError, KeyError) as error:
            reason = "audit_refresh_required" if "audit_refresh_required" in str(error) else "development_evidence_unverified"
            rows = [replace(row, record=replace(row.record,
                blocker_codes=tuple(dict.fromkeys((*row.record.blocker_codes, reason)))))
                if row.record.experiment_id in (TARGET_ID, SENTINEL_ID, "regime_same_family_hpo_vs_default") else row for row in rows]
        else:
            updated = []
            for row in rows:
                if row.record.experiment_id in (TARGET_ID, "regime_same_family_hpo_vs_default") and row.record.experiment_id in snapshots:
                    codes = tuple(code for code in row.record.blocker_codes if code != "inner_class_support_unverified")
                    row = replace(row, record=replace(row.record, status="blocked", blocker_codes=codes),
                        scientific_disposition="data_ready_pending_runtime_proof",
                        disposition_reason="genuine_worker_fit_proof_pending",
                        development_evidence_sha256=development.evidence_sha256)
                elif row.record.experiment_id == SENTINEL_ID and development.sentinel_disposition == "not_applicable":
                    row = replace(row, record=replace(row.record, status="blocked",
                        blocker_codes=tuple(dict.fromkeys((*global_codes, "no_observed_invalid_values")))),
                        scientific_disposition="not_applicable", disposition_reason="no_observed_invalid_values",
                        development_evidence_sha256=development.evidence_sha256,
                        planned_model_fits=0, planned_selector_fits=0, planned_trials=0, encoder_objects=0, arms=())
                updated.append(row)
            rows = updated
    return FullCampaignAdmission(
        tuple(rows), global_codes, authenticated,
        tuple((name, "pending_coverage") for name in campaign["model_family_audit"]),
        tuple((name, value["status"]) for name, value in campaign["context_baselines"].items()),
        intent.spec.binding_sha256, intent.spec.resource_policy,
        tuple(s for s in intent.spec.sources if s.role in ("data", "split")),
        tuple((name, json.dumps(value, sort_keys=True, allow_nan=False)) for name, value in campaign["model_family_audit"].items()),
        tuple((name, json.dumps(value, sort_keys=True, allow_nan=False)) for name, value in campaign["context_baselines"].items()),
        intent.explicit_full_request is True,
    )
