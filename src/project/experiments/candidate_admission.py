"""Task 12a: pure configuration readiness, never an execution permission.

``source_design`` is an externally supplied mapping with ``campaign`` (parsed
feature_campaign YAML), its canonical-JSON ``campaign_sha256``, and
``candidate_links`` keyed by requested experiment ID. Each link must contain
the run's notebook/cell and that hash. Diagnostic requests additionally require
``diagnostic``: generator, max_rows, groups, folds, repeats, cpu_threads.
An optional ``optional_capabilities`` mapping can describe AutoGluon as unknown,
unavailable, unsupported, or available. These are observations, not grants.

Hash/cell checks establish internal consistency ONLY. This module neither opens
source files nor proves notebook provenance, generates rows, checks evidence
bytes, constructs estimators, reserves/debits counters, fits, or creates outputs.
The future repository-owned loader must authenticate source bindings; the future
runner must generate its own rows and establish wall/RSS supervision separately.
No field in this report grants dataset access or execution, including full mode.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from project.experiments.candidate_contracts import CandidateSpec, FitBudget, RunSpec

_FAMILIES = {
    "feat_core_vs_provided": "hist_gradient_boosting",
    "feat_preserve_raw_vs_invalid_sentinel": "hist_gradient_boosting",
    "feat_meter_deviation_ratio": "hist_gradient_boosting",
    "feat_histogram_quantile": "hist_gradient_boosting",
    "feat_conditional_cross_fitted_target_encoding": "hist_gradient_boosting",
    "feat_missingness_passthrough_comparison": "hist_gradient_boosting",
    "regime_natural_vs_class_weighted": "hist_gradient_boosting",
    "model_extratrees_core_features": "extra_trees",
    "model_autogluon_core_features": "autogluon",
    "regime_same_family_hpo_vs_default": "hist_gradient_boosting",
    "feat_missingness_indicators": "hist_gradient_boosting",
    "feat_safe_calendar_numeric_transforms": "hist_gradient_boosting",
    "feat_categorical_interactions": "hist_gradient_boosting",
    "feat_missingness_bitmask": "hist_gradient_boosting",
    "feat_frequency_encoding": "hist_gradient_boosting",
    "dim_pca_vs_passthrough": "svm",
    "dim_fold_local_selector_vs_passthrough": "logistic_l2",
    "dim_semantic_block_drop": "hist_gradient_boosting",
}
_BLOCKED_IDS = {
    "prohibited_opaque_building_id_digit_extraction": "prohibited_experiment",
    "blocked_unverified_gte_family": "blocked_unverified_gte",
    "feat_group_aggregates_blocked": "blocked_group_aggregates",
}
_CORE = ("meter_reading", "square_feet", "year_built", "floor_count", "hour",
         "weekday", "month", "year", "primary_use")
_SOURCE_KEYS = {"campaign", "campaign_sha256", "candidate_links", "diagnostic", "optional_capabilities"}
_NOTEBOOKS = {"notebooks/01_data_understanding.ipynb", "notebooks/02_baseline_and_candidate_design.ipynb"}


@dataclass(frozen=True)
class AdmissionBlocker:
    """Immutable diagnostic reason, scoped to design or execution readiness."""
    code: str
    reason: str
    experiment_id: str | None = None
    scope: str = "readiness"


@dataclass(frozen=True)
class CandidateReadiness:
    experiment_id: str
    disposition: str
    model_fits_per_partition: int | None
    trials_per_partition: int | None
    learned_preprocessing_boundary: str = "outer_train"
    target_encoder_objects_per_partition: int = 0


@dataclass(frozen=True)
class CandidateAdmissionReport:
    design_valid: bool = False
    diagnostic_ready: bool = False
    blockers: tuple[AdmissionBlocker, ...] = ()
    planned_model_fits: int | None = None
    planned_trials: int | None = None
    candidates: tuple[CandidateReadiness, ...] = ()
    partition_count: int = 0
    diagnostic_max_rows: int | None = None
    campaign_sha256: str | None = None
    executed: bool = field(default=False, init=False)
    execution_authorized: bool = field(default=False, init=False)
    data_access_authorized: bool = field(default=False, init=False)
    n03_eligible: bool = field(default=False, init=False)
    source_provenance: str = "externally_supplied_unverified"

    def to_dict(self) -> dict[str, Any]:
        """Return detached JSON-safe data; never retain caller mappings."""
        return asdict(self)


def _integer(value: object, minimum: int = 1, maximum: int | None = None) -> bool:
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def _budget_valid(budget: object) -> bool:
    if type(budget) is not FitBudget:
        return False
    try:
        wall = budget.max_wall_seconds
        return (all(_integer(getattr(budget, k)) for k in ("max_model_fits", "max_rss_bytes", "cpu_threads"))
                and _integer(budget.max_trials, 0)
                and type(wall) in (int, float) and math.isfinite(wall) and wall > 0)
    except (AttributeError, OverflowError):
        return False


def _family(value: str) -> str:
    return "extra_trees" if value == "extratrees" else value


def _pairs(value: object) -> dict:
    """Revalidate caller-owned tuple configuration without shared internals."""
    if (type(value) is not tuple or any(type(pair) is not tuple or len(pair) != 2
                                      or type(pair[0]) is not str or not pair[0].strip() for pair in value)
            or len({pair[0] for pair in value}) != len(value)):
        raise ValueError("parameters must be unique immutable named tuple pairs")
    return dict(value)


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate_candidate_parameters(candidate: CandidateSpec) -> None:
    """Static dispatch contract checks; inspect signatures but construct no models."""
    recipe = candidate.experiment_id
    pool = candidate.feature_pool
    if (type(pool) is not tuple or not pool or any(type(n) is not str or not n.strip() for n in pool)
            or len(set(pool)) != len(pool)):
        raise ValueError("unique nonempty raw predictor names required")
    if recipe not in ("feat_core_vs_provided", "dim_pca_vs_passthrough", "dim_fold_local_selector_vs_passthrough", "dim_semantic_block_drop") and set(pool) != set(_CORE):
        raise ValueError("source core-feature recipe requires all nine raw core predictors")
    if recipe == "dim_pca_vs_passthrough" and not any(n != "primary_use" for n in pool):
        raise ValueError("PCA requires numeric raw predictors")
    methods, params = _pairs(candidate.method_params), _pairs(candidate.model_params)
    allowlists = {
        "feat_histogram_quantile": {"n_bins", "n_quantiles"},
        "feat_conditional_cross_fitted_target_encoding": {"smoothing", "n_splits", "cv_splits", "seed", "max_categories"},
        "dim_pca_vs_passthrough": {"pca_components"},
        "dim_semantic_block_drop": {"drop_block"},
        "model_autogluon_core_features": {"time_limit", "max_rss_bytes", "num_cpus"},
        "regime_same_family_hpo_vs_default": {"study_spec", "study_dir"},
        "feat_preserve_raw_vs_invalid_sentinel": {"rule", "sentinel_rule", "column", "invalid_values", "replacement_value", "evidence_reference", "evidence_sha256", "evidence_root"},
    }
    if set(methods) - allowlists.get(recipe, set()):
        raise ValueError("unknown method keys for recipe")
    bounds = {"n_bins": (2, 64), "n_quantiles": (2, 256), "n_splits": (2, 5), "cv_splits": (2, 5), "seed": (0, 2**32 - 1), "max_categories": (1, 4096)}
    for key, value in methods.items():
        if key in bounds and not _integer(value, *bounds[key]):
            raise ValueError(f"invalid typed/bounded {key}")
        if key == "smoothing" and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1000):
            raise ValueError("invalid smoothing")
        if key == "pca_components" and (type(value) is not float or not 0 < value < 1):
            raise ValueError("pca_components must be a float in (0,1)")
        if key in ("study_dir", "evidence_root") and (type(value) is not str or not value.strip()):
            raise ValueError("configuration path must be a nonempty string")
    if {"n_splits", "cv_splits"} <= set(methods):
        raise ValueError("duplicate split aliases")
    if recipe == "dim_semantic_block_drop":
        blocks = {"measurement": {"meter_reading"}, "calendar": {"hour", "weekday", "month", "year"},
                  "building_metadata": {"primary_use", "square_feet", "year_built", "floor_count"}}
        block = methods.get("drop_block")
        if type(block) is not str or block not in blocks or not blocks[block] & set(pool) or set(pool) <= blocks[block]:
            raise ValueError("semantic drop requires an explicit present block and nonempty retained pool")
    if recipe == "feat_preserve_raw_vs_invalid_sentinel":
        aliases = {"rule", "sentinel_rule"} & set(methods)
        scalar_keys = {"column", "invalid_values", "replacement_value", "evidence_reference", "evidence_sha256"}
        if len(aliases) > 1 or aliases and scalar_keys & set(methods):
            raise ValueError("ambiguous sentinel rule")
        if aliases:
            value = methods[next(iter(aliases))]
            if type(value) is not tuple or len(value) != 5:
                raise ValueError("sentinel alias requires five immutable scalar/tuple fields")
            rule = dict(zip(("column", "invalid_values", "replacement_value", "evidence_reference", "evidence_sha256"), value, strict=True))
        else:
            rule = methods
        if scalar_keys & set(rule):
            if not scalar_keys <= set(rule):
                raise ValueError("incomplete sentinel rule")
            values = rule["invalid_values"]
            if (rule["column"] not in _CORE or rule["column"] == "primary_use"
                    or type(values) is not tuple or not values
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                    or type(rule["replacement_value"]) not in (int, float) or not math.isfinite(rule["replacement_value"])
                    or type(rule["evidence_reference"]) is not str or not rule["evidence_reference"].strip()
                    or type(rule["evidence_sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", rule["evidence_sha256"]) is None):
                raise ValueError("invalid sentinel scalar fields")
    family = _family(candidate.model_family)
    if family == "autogluon":
        if set(params) - {"time_limit", "max_rss_bytes", "num_cpus"} or set(params) & set(methods):
            raise ValueError("unknown/duplicate AutoGluon settings")
        settings = {"time_limit": 10., "max_rss_bytes": 1024**3, "num_cpus": 1, **methods, **params}
        if (not _positive(settings["time_limit"]) or not _integer(settings["max_rss_bytes"])
                or type(settings["num_cpus"]) is not int or settings["num_cpus"] != 1):
            raise ValueError("invalid bounded AutoGluon settings")
        return
    if family == "extra_trees":
        defaults = {"n_estimators": 64, "max_depth": 12, "min_samples_leaf": 2,
                    "max_features": "sqrt", "bootstrap": False, "class_weight": None,
                    "n_jobs": 1, "random_state": 42}
        tree_bounds = {"n_estimators": (1, 256), "max_depth": (1, 24), "min_samples_leaf": (1, 128)}
        for key, value in params.items():
            if key in tree_bounds:
                if not _integer(value, *tree_bounds[key]):
                    raise ValueError("ExtraTrees setting outside bounds")
            elif key not in defaults or type(value) is not type(defaults[key]) or value != defaults[key]:
                raise ValueError("ExtraTrees setting differs from fixed profile")
        return
    cls = SVC if family == "svm" else LogisticRegression if family == "logistic_l2" else HistGradientBoostingClassifier
    defaults = {key: p.default for key, p in inspect.signature(cls).parameters.items() if p.default is not inspect.Parameter.empty}
    if family == "svm":
        defaults.update(C=1., kernel="rbf", gamma="scale", random_state=42)
    elif family == "logistic_l2":
        defaults.update(C=1., l1_ratio=0., solver="lbfgs", max_iter=1000, random_state=42)
    else:
        defaults.update(early_stopping=False, random_state=42)
    if recipe in ("feat_missingness_passthrough_comparison", "regime_natural_vs_class_weighted"):
        defaults = {k: defaults[k] for k in ("early_stopping", "random_state")}
    if any(key not in defaults or type(value) is not type(defaults[key]) or value != defaults[key] for key, value in params.items()):
        raise ValueError("explicit model parameters must match the frozen profile")


def _diagnostic_valid(value: object) -> bool:
    keys = {"generator", "max_rows", "groups", "folds", "repeats", "cpu_threads"}
    return (isinstance(value, Mapping) and set(value) == keys
            and value["generator"] == "candidate_fixtures_v1"
            and _integer(value["max_rows"], 8, 128)
            and _integer(value["groups"], 4, value["max_rows"] // 2)
            and all(type(value[k]) is int and value[k] == expected
                    for k, expected in (("folds", 2), ("repeats", 1), ("cpu_threads", 1))))


def _json_payload(value: object) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("JSON configuration keys must be strings")
            _json_payload(child)
    elif type(value) is list:
        for child in value:
            _json_payload(child)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("configuration values must be JSON primitives")


def _source_checks(run: RunSpec, source: object, blockers: list[AdmissionBlocker]) -> tuple[dict, str | None]:
    def reject(code: str, reason: str):
        blockers.append(AdmissionBlocker(code, reason, scope="design"))
    if not isinstance(source, Mapping) or not {"campaign", "campaign_sha256", "candidate_links"} <= set(source):
        reject("missing_source_design", "Parsed campaign, canonical hash and per-candidate notebook/cell bindings required")
        return {}, None
    if set(source) - _SOURCE_KEYS:
        reject("unsupported_source_design_fields", "Caller data/split paths and execution flags are not admission inputs")
    campaign = source["campaign"]
    digest = source["campaign_sha256"]
    if not isinstance(campaign, dict):
        reject("invalid_source_design", "campaign must be a JSON-safe parsed configuration mapping")
        return {}, None
    try:
        _json_payload(campaign)
        encoded = json.dumps(campaign, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        actual_hash = hashlib.sha256(encoded).hexdigest()
    except (TypeError, ValueError, OverflowError, RecursionError):
        reject("invalid_source_design", "campaign contains non-JSON or nonfinite values")
        return {}, None
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None or digest != actual_hash:
        reject("source_hash_mismatch", "campaign_sha256 must match canonical JSON of supplied campaign")
    if (type(campaign.get("schema_version")) is not int or campaign.get("schema_version") != 1
            or campaign.get("campaign_id") != "task7_feature_campaign"
            or campaign.get("mode") != "design_only"
            or campaign.get("execution_authorized") is not False
            or campaign.get("baseline_config") != "configs/experiments/baselines.yaml"
            or campaign.get("resource_budget") != "configs/resource_budget.yaml"):
        reject("invalid_campaign_contract", "Only the current design_only, non-authorized Task 7 contract is supported")
    contract = campaign.get("prediction_contract", {})
    if (not isinstance(contract, dict) or contract.get("target") != "anomaly"
            or any(type(contract.get(k)) is not int or contract.get(k) != v
                   for k, v in (("positive_label", 1), ("folds", 4), ("repeats", 3)))
            or contract.get("split_seeds") != [42, 43, 44]
            or any(type(v) is not int for v in contract.get("split_seeds", []))):
        reject("prediction_contract_mismatch", "Source contract must preserve the frozen 4-fold/3-repeat binary protocol")
    rows = campaign.get("experiment_matrix")
    if (not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or type(row.get("id")) is not str for row in rows)
            or len({row["id"] for row in rows}) != len(rows)):
        reject("invalid_experiment_matrix", "Source matrix requires unique explicit experiment IDs")
        matrix = {}
    else:
        matrix = {row["id"]: row for row in rows}
    links = source["candidate_links"]
    if not isinstance(links, dict) or set(links) != {c.experiment_id for c in run.candidates}:
        reject("source_linkage_mismatch", "Exact per-request candidate source bindings required")
        links = {}
    for candidate in run.candidates:
        recipe = candidate.experiment_id
        expected = {"source_notebook": run.source_notebook, "source_cell_id": run.source_cell_id, "campaign_sha256": digest}
        if links.get(recipe) != expected:
            reject("source_linkage_mismatch", f"Notebook/cell/config linkage differs for {recipe}")
        row = matrix.get(recipe)
        if row is None:
            reject("experiment_not_in_source_matrix", f"Requested ID absent from supplied source matrix: {recipe}")
        elif recipe in _FAMILIES and (_family(row.get("model_family", "")) != _FAMILIES[recipe]
                                     or row.get("status") not in ("planned", "conditional")
                                     or row.get("execution_status") != "blocked_unapproved_compute"):
            reject("source_matrix_mismatch", f"Source row family/status is inconsistent: {recipe}")
    return matrix, digest if digest == actual_hash else None


def _candidate_readiness(candidate: CandidateSpec, capabilities: Mapping, blockers: list[AdmissionBlocker]) -> CandidateReadiness:
    recipe = candidate.experiment_id
    start = len(blockers)
    def reject(code: str, reason: str, scope: str = "readiness"):
        blockers.append(AdmissionBlocker(code, reason, recipe, scope))
    fits, trials, encoder_objects = 2, 0, 0
    if recipe in _BLOCKED_IDS:
        reject(_BLOCKED_IDS[recipe], "Source experiment remains prohibited/blocked", "design")
        fits = None
    elif recipe not in _FAMILIES:
        reject("unknown_experiment", "No registered candidate recipe", "design")
        fits = None
    elif _family(candidate.model_family) != _FAMILIES[recipe]:
        reject("model_family_mismatch", "Model family differs from the registered source recipe", "design")
    if recipe in _FAMILIES:
        try:
            _validate_candidate_parameters(candidate)
        except (ValueError, TypeError, OverflowError) as error:
            reject("invalid_candidate_parameters", str(error), "design")
    for name in candidate.feature_pool:
        if type(name) is not str:
            reject("unverified_raw_source", "Raw feature names must be strings", "design")
            continue
        if name.startswith("gte_"):
            reject("blocked_unverified_gte", "GTE raw source not verified", "design")
        elif name.startswith(("lag_", "rolling_", "group_", "history_", "building_history")):
            reject("blocked_group_aggregates", "History/group source lacks point-in-time approval", "design")
        elif name not in _CORE:
            reject("unverified_raw_source", "Raw source is not an approved target-independent core predictor", "design")
    try:
        methods = _pairs(candidate.method_params)
    except (ValueError, TypeError):
        reject("invalid_candidate_parameters", "Malformed method parameter packet", "design")
        return CandidateReadiness(recipe, "blocked", None, None)
    if any(k in methods for k in ("data_path", "split_path", "prepared_data", "execution_authorized", "trust_me_synthetic")):
        reject("external_data_forbidden", "Candidate methods cannot select data/splits or grant execution", "design")
    if recipe == "feat_core_vs_provided":
        reject("no_verified_additional_features", "Verified additional repository producer evidence is not available")
    elif recipe == "feat_preserve_raw_vs_invalid_sentinel":
        required = {"column", "invalid_values", "replacement_value", "evidence_reference", "evidence_sha256"}
        present = required <= set(methods) or bool({"rule", "sentinel_rule"} & set(methods))
        reject("sentinel_evidence_unverified" if present else "missing_evidence_rule",
               "Admission cannot verify evidence bytes; loader/producer verification remains required")
    elif recipe == "feat_histogram_quantile":
        fits = 3
    elif recipe == "dim_fold_local_selector_vs_passthrough":
        fits = 3  # control L2 + challenger L1 selector + challenger L2
    elif recipe == "dim_semantic_block_drop" and methods.get("drop_block") not in ("calendar", "measurement", "building_metadata"):
        reject("missing_drop_block", "Explicit registered semantic block required", "design")
    elif recipe == "feat_conditional_cross_fitted_target_encoding":
        splits = methods.get("n_splits", methods.get("cv_splits", 3))
        if not _integer(splits, 2, 5):
            reject("invalid_encoder_splits", "Cross-fitted encoder requires 2..5 integer splits", "design")
        else:
            encoder_objects = splits + 1
    elif recipe == "model_autogluon_core_features":
        capability = capabilities.get("autogluon", "unknown")
        if capability != "available":
            code = "optional_dependency_unavailable" if capability == "unavailable" else "optional_capability_unsupported" if capability == "unsupported" else "optional_capability_unverified"
            reject(code, "AutoGluon capability must be established by a future repository-owned probe")
        reject("supervisor_unavailable", "No live repository-owned wall/RSS supervisor lease exists")
    elif recipe == "regime_same_family_hpo_vs_default":
        reject("hpo_not_ready", "Parent-approved Task 9 accounting/resume integration remains required")
        study = methods.get("study_spec")
        try:
            study_values = dict(study) if isinstance(study, tuple) else {}
        except (TypeError, ValueError):
            study_values = {}
        count, folds = study_values.get("n_trials"), study_values.get("inner_folds")
        if _integer(count) and _integer(folds, 2, 5):
            fits, trials = count * folds + 2, count
        else:
            fits, trials = None, None
            reject("hpo_cost_unknown", "Explicit integer trial/fold counts required to plan T*K+2 fits")
    return CandidateReadiness(recipe, "blocked" if len(blockers) > start else "diagnostic_config_ready",
                              fits, trials, target_encoder_objects_per_partition=encoder_objects)


def validate_candidate_admission(
    run_spec: RunSpec, source_design: Mapping[str, Any], budget: FitBudget,
) -> CandidateAdmissionReport:
    """Assess a supplied design, with zero IO/fits and no permission inference.

    ``diagnostic_ready`` means configuration/planned budget checks passed for a
    generated diagnostic request only. It does not attest rows, group/class
    support, wall/RSS enforcement, source authenticity, or candidate fitting.
    Non-tuning max_trials may be zero. All other caps are positive exact typed.
    """
    blockers: list[AdmissionBlocker] = []
    if type(run_spec) is not RunSpec:
        return CandidateAdmissionReport(blockers=(AdmissionBlocker("invalid_run_spec", "Typed RunSpec required", scope="design"),))
    full = run_spec.execution_mode == "full"
    if full:
        blockers.append(AdmissionBlocker(
            "production_execution_blocked", "Full execution is unconditionally blocked; no approved production budget/entrypoint exists",
        ))
    if (run_spec.execution_mode not in ("full", "synthetic_smoke") or type(run_spec.version) is not int or run_spec.version != 1
            or run_spec.source_notebook not in _NOTEBOOKS or type(run_spec.source_cell_id) is not str or not run_spec.source_cell_id.strip()
            or not isinstance(run_spec.candidates, tuple) or not run_spec.candidates
            or any(type(c) is not CandidateSpec for c in run_spec.candidates)
            or len({c.experiment_id for c in run_spec.candidates}) != len(run_spec.candidates)):
        return CandidateAdmissionReport(blockers=(AdmissionBlocker("invalid_run_spec", "Invalid diagnostic RunSpec fields", scope="design"),))
    valid_budget = _budget_valid(budget) and _budget_valid(run_spec.budget)
    if not valid_budget:
        blockers.append(AdmissionBlocker("invalid_budget", "Positive finite exact typed runtime/fit/RSS/thread caps required", scope="design"))
    elif budget != run_spec.budget:
        blockers.append(AdmissionBlocker("budget_mismatch", "Supplied budget must equal RunSpec caps", scope="design"))
    if valid_budget and not full and budget.cpu_threads != 1:
        blockers.append(AdmissionBlocker("invalid_diagnostic_protocol", "Diagnostic budget requires one CPU", scope="design"))
    _, digest = _source_checks(run_spec, source_design, blockers)
    diagnostic = source_design.get("diagnostic") if isinstance(source_design, Mapping) else None
    diagnostic_valid = _diagnostic_valid(diagnostic)
    if full and isinstance(source_design, Mapping) and "diagnostic" in source_design:
        blockers.append(AdmissionBlocker("mixed_execution_protocol", "Full declarations cannot carry a generated diagnostic protocol", scope="design"))
    elif not full and not diagnostic_valid:
        blockers.append(AdmissionBlocker("invalid_diagnostic_protocol", "Only generated <=128-row, >=4-group, 2-fold/1-repeat/CPU1 diagnostics supported", scope="design"))
    capabilities = source_design.get("optional_capabilities", {}) if isinstance(source_design, Mapping) else {}
    if (not isinstance(capabilities, dict) or set(capabilities) - {"autogluon"}
            or any(v not in ("unknown", "available", "unavailable", "unsupported") for v in capabilities.values())):
        blockers.append(AdmissionBlocker("invalid_capability_descriptor", "Only explicit AutoGluon capability observations accepted", scope="design"))
        capabilities = {}
    candidates = tuple(_candidate_readiness(c, capabilities, blockers) for c in run_spec.candidates)
    if not full and diagnostic_valid:
        for candidate in run_spec.candidates:
            if candidate.experiment_id == "feat_conditional_cross_fitted_target_encoding":
                try:
                    methods = _pairs(candidate.method_params)
                except (ValueError, TypeError):
                    continue
                splits = methods.get("n_splits", methods.get("cv_splits", 3))
                if _integer(splits, 2, 5) and diagnostic["groups"] // 2 < splits:
                    blockers.append(AdmissionBlocker("insufficient_inner_groups", "Declared groups cannot support encoder splits in both balanced whole-building outer partitions", candidate.experiment_id))
    partitions = 12 if full else 2 if diagnostic_valid else 0
    fits = (sum(c.model_fits_per_partition for c in candidates) * partitions
            if partitions and all(c.model_fits_per_partition is not None for c in candidates) else None)
    trials = (sum(c.trials_per_partition for c in candidates) * partitions
              if partitions and all(c.trials_per_partition is not None for c in candidates) else None)
    if valid_budget and fits is not None and fits > budget.max_model_fits:
        blockers.append(AdmissionBlocker("fit_budget_exhausted", "Total cross-fold arm fit plan exceeds the declared cap"))
    if valid_budget and trials is not None and trials > budget.max_trials:
        blockers.append(AdmissionBlocker("trial_budget_exhausted", "Total cross-fold trial plan exceeds the declared cap"))
    return CandidateAdmissionReport(
        design_valid=not any(b.scope == "design" for b in blockers),
        diagnostic_ready=not full and not blockers, blockers=tuple(blockers),
        planned_model_fits=fits, planned_trials=trials, candidates=candidates,
        partition_count=partitions, diagnostic_max_rows=diagnostic["max_rows"] if not full and diagnostic_valid else None,
        campaign_sha256=digest,
    )
